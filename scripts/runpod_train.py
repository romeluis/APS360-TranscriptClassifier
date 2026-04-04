"""RunPod remote execution script.

Provisions a RunPod GPU pod, copies the project and training data,
runs training or evaluation remotely, and downloads resulting artifacts.

Flow:
1) Create pod (with SSH on public IP)
2) Immediately verify torch.cuda.is_available() after SSH is ready
3) Copy project files via SCP
4) Archive and copy training data
5) Install uv, create venv, install torch (cu128) + deps, run selected task
6) Download model and logs via SCP
7) Terminate pod (unless --no-terminate)
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPU_PREFERRED_ORDER = [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA L40",
    "NVIDIA H100 80GB HBM3",
]

# Base Docker image with CUDA 12.8.1 runtime.  PyTorch is installed
# explicitly into a uv-managed venv during run_training() to avoid
# version conflicts with the image's system packages.
DOCKER_IMAGES = [
    "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
]

REMOTE_WORKSPACE = "/workspace/simplify"
DEFAULT_POD_HEALTHCHECK_RETRIES = 3


# ---------------------------------------------------------------------------
# RunPodTrainer
# ---------------------------------------------------------------------------


class RunPodTrainer:
    """Provision a RunPod pod and run remote tasks."""

    def __init__(
        self,
        api_key: str,
        ssh_key_path: str,
        gpu_type: str = "NVIDIA GeForce RTX 5090",
    ):
        self.api_key = api_key
        self.ssh_key_path = os.path.expanduser(ssh_key_path)
        self.gpu_type = gpu_type
        self.gpu_display_name = gpu_type
        self.pod_id: str | None = None
        self.pod_host_id: str | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int = 22
        self.ssh_user: str = "root"

        # Configure RunPod SDK.
        import runpod as _runpod

        _runpod.api_key = self.api_key

        key_path = Path(self.ssh_key_path)
        if not key_path.exists():
            raise FileNotFoundError(
                f"SSH key not found at {key_path}. Set RUNPOD_SSH_KEY_PATH to a valid private key."
            )

    # ------------------------------------------------------------------
    # RunPod API helpers (via SDK)
    # ------------------------------------------------------------------

    def list_available_gpus(self) -> list[str]:
        """Return full GPU IDs currently available on secure cloud."""
        import runpod

        all_gpus = runpod.get_gpus()
        available: list[str] = []
        for gpu in all_gpus:
            gpu_id = gpu["id"]
            try:
                detail = runpod.get_gpu(gpu_id)
                if detail.get("secureCloud"):
                    available.append(gpu_id)
            except Exception:
                continue
        return available

    def create_pod(self, disk_size: int = 50, volume_size: int = 0) -> str:
        """Create a secure-cloud US pod, falling back to alternate GPUs as needed."""
        import runpod

        # Build fallback list: selected GPU first, then preferred order, then
        # any other secure-cloud GPUs discovered dynamically.
        seen: set[str] = {self.gpu_type}
        candidates: list[str] = [self.gpu_type]
        for gpu_name in GPU_PREFERRED_ORDER:
            if gpu_name not in seen:
                candidates.append(gpu_name)
                seen.add(gpu_name)

        last_error: str | None = None

        for gpu_name in candidates:
            if gpu_name != self.gpu_type:
                print(f"Requested GPU unavailable, trying fallback {gpu_name}...", flush=True)

            for image in DOCKER_IMAGES:
                try:
                    pod = runpod.create_pod(
                        name="simplify-training",
                        image_name=image,
                        gpu_type_id=gpu_name,
                        cloud_type="SECURE",
                        country_code="US",
                        support_public_ip=True,
                        start_ssh=True,
                        gpu_count=1,
                        volume_in_gb=volume_size,
                        container_disk_in_gb=disk_size,
                        ports="22/tcp",
                    )

                    self.gpu_display_name = gpu_name
                    self.pod_id = pod["id"]
                    self.pod_host_id = (pod.get("machine") or {}).get("podHostId")
                    print(
                        f"Pod created: {self.pod_id} (GPU: {gpu_name}, image: {image})",
                        flush=True,
                    )
                    return self.pod_id

                except Exception as exc:
                    last_error = str(exc)
                    msg = last_error.lower()

                    # Capacity problem → try next GPU type.
                    if "resources" in msg or "no available" in msg or "capacity" in msg:
                        break

                    # Image problem → try next image.
                    if "layer" in msg or "image" in msg:
                        print(f"Image {image} unavailable, trying next...", flush=True)
                        continue

                    raise

        raise RuntimeError(
            f"No secure-cloud US GPUs available across all fallback types. Last error: {last_error}"
        )

    def wait_for_ready(self, timeout: int = 600, poll_interval: int = 10) -> None:
        """Wait for pod to be running, then allow a short SSH warmup window."""
        import runpod

        if not self.pod_id:
            raise RuntimeError("Cannot wait for pod readiness before creating a pod.")

        print("Waiting for pod to become ready...", flush=True)
        start = time.time()
        last_status = None

        while time.time() - start < timeout:
            pod = runpod.get_pod(self.pod_id)

            if pod:
                status = pod.get("desiredStatus")
                runtime = pod.get("runtime")
                uptime = pod.get("uptimeSeconds") or 0

                if status != last_status:
                    print(f"  Pod status: {status}", flush=True)
                    last_status = status

                if status == "RUNNING" and runtime is not None:
                    self.pod_host_id = pod.get("machineId")
                    self._resolve_ssh_endpoint(runtime)

                    # Wait until a public-IP SSH endpoint is available.
                    # The runtime object appears before ports are mapped,
                    # so we keep polling until _resolve_ssh_endpoint finds
                    # the port-22 mapping.
                    if self.ssh_host is None:
                        elapsed = int(time.time() - start)
                        print(
                            f"  Pod running but SSH port not mapped yet ({elapsed}s)...",
                            flush=True,
                        )
                        time.sleep(poll_interval)
                        continue

                    print(
                        f"Pod is running (uptime={uptime}s). "
                        "Waiting 10s for SSH to settle...",
                        flush=True,
                    )
                    time.sleep(10)
                    print(
                        f"SSH target ready: {self._ssh_target}:{self.ssh_port} (mode: {self._ssh_mode})",
                        flush=True,
                    )
                    return

            time.sleep(poll_interval)

        raise TimeoutError(f"Pod did not become ready within {timeout} seconds.")

    def _run_immediate_cuda_check(self) -> bool:
        """Run torch.cuda.is_available() right after pod connection is ready."""
        print("Running immediate CUDA health check on pod...", flush=True)
        result = self._run_ssh_status(
            "python3 -c \"import torch, sys; ok = bool(torch.cuda.is_available()); "
            "print(f'torch.cuda.is_available()={ok}'); sys.exit(0 if ok else 2)\""
        )

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            print(stdout, flush=True)
        if stderr:
            print(stderr, file=sys.stderr, flush=True)

        return result.returncode == 0

    def create_pod_with_cuda_check(
        self,
        disk_size: int = 50,
        volume_size: int = 0,
        max_attempts: int = DEFAULT_POD_HEALTHCHECK_RETRIES,
    ) -> str:
        """Create a pod, verify CUDA immediately after connect, and retry if unhealthy."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            print(f"Provisioning pod attempt {attempt}/{max_attempts}...", flush=True)

            try:
                self.create_pod(disk_size=disk_size, volume_size=volume_size)
                self.wait_for_ready()

                if self._run_immediate_cuda_check():
                    print("Pod passed immediate CUDA health check.", flush=True)
                    if not self.pod_id:
                        raise RuntimeError("Pod ID is missing after successful health check.")
                    return self.pod_id

                last_error = "torch.cuda.is_available() returned False"
                print(
                    "CUDA health check failed on this pod; terminating and retrying...",
                    flush=True,
                )
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"Pod attempt {attempt} failed before training: {exc}",
                    flush=True,
                )

            if self.pod_id:
                try:
                    self.terminate_pod()
                except Exception as term_exc:
                    print(f"Warning: failed to terminate unhealthy pod: {term_exc}", flush=True)

        raise RuntimeError(
            "Unable to provision a healthy pod after "
            f"{max_attempts} attempt(s). Last error: {last_error}"
        )

    def terminate_pod(self) -> None:
        """Terminate the current pod to stop billing."""
        if not self.pod_id:
            return

        import runpod

        pod_id = self.pod_id
        print(f"Terminating pod {pod_id}...", flush=True)
        runpod.terminate_pod(pod_id)
        print("Pod terminated.", flush=True)

        # Clear connection state so retries always re-resolve a fresh endpoint.
        self.pod_id = None
        self.pod_host_id = None
        self.ssh_host = None
        self.ssh_port = 22
        self.ssh_user = "root"

    # ------------------------------------------------------------------
    # SSH / SCP plumbing
    # ------------------------------------------------------------------

    @property
    def _ssh_target(self) -> str:
        if self.ssh_host:
            return f"{self.ssh_user}@{self.ssh_host}"
        if not self.pod_id and not self.pod_host_id:
            raise RuntimeError("SSH target is not available before pod creation.")
        username = self.pod_host_id or self.pod_id
        return f"{username}@ssh.runpod.io"

    @property
    def _ssh_mode(self) -> str:
        if self.ssh_host:
            return "public-ip"
        return "proxy"

    def _resolve_ssh_endpoint(self, runtime: dict[str, Any]) -> None:
        """Prefer direct SSH endpoint (public IP + mapped SSH port) when available."""
        ports = runtime.get("ports") or []
        for port in ports:
            private_port = port.get("privatePort")
            public_port = port.get("publicPort")
            ip = port.get("ip")
            if private_port == 22 and public_port and ip:
                self.ssh_host = str(ip)
                self.ssh_port = int(public_port)
                self.ssh_user = "root"
                return

        # Fallback to RunPod proxy endpoint.
        self.ssh_host = None
        self.ssh_port = 22

    def _ssh_base_cmd(self) -> list[str]:
        return [
            "ssh",
            "-T",
            "-p",
            str(self.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-i",
            self.ssh_key_path,
            self._ssh_target,
        ]

    def _scp_base_cmd(self) -> list[str]:
        return [
            "scp",
            "-O",
            "-P",
            str(self.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-i",
            self.ssh_key_path,
        ]

    def _run_ssh_status(self, remote_cmd: str) -> subprocess.CompletedProcess[str]:
        cmd = self._ssh_base_cmd() + [f"bash -lc {shlex.quote(remote_cmd)}"]
        return subprocess.run(cmd, capture_output=True, text=True)

    def _run_ssh(self, remote_cmd: str, stream_output: bool = False) -> None:
        if stream_output:
            cmd = self._ssh_base_cmd() + [f"bash -lc {shlex.quote(remote_cmd)}"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_lines: list[str] = []
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output_lines.append(line)

            returncode = process.wait()
            if returncode != 0:
                combined = "".join(output_lines).strip()
                raise RuntimeError(
                    "Remote command failed (%s): %s\noutput: %s"
                    % (returncode, remote_cmd, combined)
                )
            return

        result = self._run_ssh_status(remote_cmd)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            raise RuntimeError(
                "Remote command failed (%s): %s\nstdout: %s\nstderr: %s"
                % (result.returncode, remote_cmd, stdout, stderr)
            )

    def _scp_upload(
        self,
        local_path: Path,
        remote_path: str,
        recursive: bool = True,
        show_progress: bool = False,
    ) -> None:
        if not local_path.exists():
            raise FileNotFoundError(f"Upload source does not exist: {local_path}")

        cmd = self._scp_base_cmd()
        if recursive:
            cmd.append("-r")
        cmd += [str(local_path), f"{self._ssh_target}:{remote_path}"]

        if show_progress:
            result = subprocess.run(cmd)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (getattr(result, "stderr", "") or "").strip()
            stdout = (getattr(result, "stdout", "") or "").strip()
            raise RuntimeError(
                "SCP upload failed (%s): %s -> %s\nstdout: %s\nstderr: %s"
                % (result.returncode, local_path, remote_path, stdout, stderr)
            )

    def _scp_download(self, remote_path: str, local_path: Path, recursive: bool = True) -> None:
        cmd = self._scp_base_cmd()
        if recursive:
            cmd.append("-r")
        cmd += [f"{self._ssh_target}:{remote_path}", str(local_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            raise RuntimeError(
                "SCP download failed (%s): %s -> %s\nstdout: %s\nstderr: %s"
                % (result.returncode, remote_path, local_path, stdout, stderr)
            )

    # ------------------------------------------------------------------
    # Project & data sync
    # ------------------------------------------------------------------

    def sync_project(self, project_dir: str = ".") -> None:
        """Upload project files via SCP, excluding local environment/cache dirs."""
        root = Path(project_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Project directory does not exist: {root}")

        print("Syncing project files...", flush=True)
        # Clean and recreate remote workspace.
        self._run_ssh(f"rm -rf {REMOTE_WORKSPACE} && mkdir -p {REMOTE_WORKSPACE}")

        excluded = {
            ".git",
            ".venv",
            ".DS_Store",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "data",
        }

        upload_items = [
            item
            for item in sorted(root.iterdir(), key=lambda p: p.name)
            if not item.name.startswith(".") and item.name not in excluded
        ]

        total_items = len(upload_items)
        print(f"Uploading {total_items} project item(s)...", flush=True)

        for idx, item in enumerate(upload_items, 1):
            item_type = "dir" if item.is_dir() else "file"
            print(f"  [{idx}/{total_items}] Uploading {item_type}: {item.name}", flush=True)
            self._scp_upload(item, f"{REMOTE_WORKSPACE}/{item.name}", recursive=item.is_dir())

        print("Project sync complete.", flush=True)

    # Backward-compatible alias used by menu.py.
    sync_code = sync_project

    def sync_data(self, data_dir: str = "data/output") -> None:
        """Upload training data as a single zip archive to remote data/output path."""
        local_data = Path(data_dir).resolve()
        if not local_data.exists() or not local_data.is_dir():
            raise FileNotFoundError(f"Data directory does not exist or is not a directory: {local_data}")

        print("Syncing training data...", flush=True)
        files = [path for path in local_data.rglob("*") if path.is_file()]
        if not files:
            raise RuntimeError(f"No files found under data directory: {local_data}")

        total_files = len(files)
        total_bytes = sum(path.stat().st_size for path in files)
        print(
            f"Preparing data archive: {total_files} files, {total_bytes / (1024 * 1024):.1f} MB",
            flush=True,
        )

        tmp = tempfile.NamedTemporaryFile(prefix="simplify_data_", suffix=".zip", delete=False)
        archive_path = Path(tmp.name)
        tmp.close()

        try:
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for idx, src in enumerate(files, 1):
                    rel = src.relative_to(local_data)
                    archive.write(src, arcname=str(Path("output") / rel))
                    if idx % 500 == 0 or idx == total_files:
                        pct = (idx / total_files) * 100
                        print(f"  Archive progress: {idx}/{total_files} files ({pct:.1f}%)", flush=True)

            archive_mb = archive_path.stat().st_size / (1024 * 1024)
            print(f"Archive created: {archive_mb:.1f} MB ({archive_path})", flush=True)

            remote_archive = f"/tmp/simplify_data_{self.pod_id or 'upload'}.zip"
            print("Uploading data archive...", flush=True)
            self._scp_upload(archive_path, remote_archive, recursive=False, show_progress=True)

            print("Extracting data archive on pod...", flush=True)
            self._run_ssh(
                " && ".join(
                    [
                        f"rm -rf {REMOTE_WORKSPACE}/data/output",
                        f"mkdir -p {REMOTE_WORKSPACE}/data",
                        (
                            "python3 -c \"import zipfile; "
                            f"z=zipfile.ZipFile('{remote_archive}'); "
                            f"z.extractall('{REMOTE_WORKSPACE}/data'); z.close()\""
                        ),
                        f"rm -f {remote_archive}",
                    ]
                )
            )
        finally:
            if archive_path.exists():
                archive_path.unlink()

        print("Data sync complete.", flush=True)

    # ------------------------------------------------------------------
    # Remote task execution
    # ------------------------------------------------------------------

    def _setup_remote_environment(self, allow_cpu: bool = False) -> str:
        """Set up uv venv, install deps, and run CUDA preflight.

        This sequence must stay exactly aligned with training setup.
        """
        venv_python = f"{REMOTE_WORKSPACE}/.venv/bin/python"

        print("Setting up uv venv and installing dependencies on pod...", flush=True)
        self._run_ssh(
            " && ".join(
                [
                    f"cd {REMOTE_WORKSPACE}",
                    # 1. Install uv
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    # Make uv available in this shell session
                    'export PATH="$HOME/.local/bin:$PATH"',
                    # 2. Create venv
                    "uv venv",
                    # 3. Install torch from official cu128 wheel index
                    "uv pip install torch --index-url https://download.pytorch.org/whl/cu128",
                    # 4. Install remaining project dependencies
                    "uv pip install -r requirements.txt",
                ]
            ),
            stream_output=True,
        )

        print("Running remote CUDA preflight...", flush=True)
        self._run_ssh(
            " && ".join(
                [
                    f"cd {REMOTE_WORKSPACE}",
                    # Force NVIDIA driver init before importing torch
                    "nvidia-smi > /dev/null 2>&1",
                    (
                        f"{venv_python} -c \""
                        "import json, sys, torch; "
                        f"allow_cpu = {str(allow_cpu)}; "
                        "cuda_ok = torch.cuda.is_available(); "
                        "count = torch.cuda.device_count() if cuda_ok else 0; "
                        "name = torch.cuda.get_device_name(0) if cuda_ok and count > 0 else ''; "
                        "info = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
                        "'cuda_available': cuda_ok, 'device_count': count, 'gpu_name': name}; "
                        "print('CUDA preflight:', json.dumps(info)); "
                        "sys.exit(0 if (cuda_ok or allow_cpu) else 2)\""
                    ),
                ]
            ),
            stream_output=True,
        )

        return venv_python

    def run_training(self, train_args: str, allow_cpu: bool = False) -> None:
        """Create a uv-managed venv, install PyTorch + deps, and run training."""
        venv_python = self._setup_remote_environment(allow_cpu=allow_cpu)

        print("Starting remote training...", flush=True)
        self._run_ssh(
            f"cd {REMOTE_WORKSPACE} && "
            f"nvidia-smi > /dev/null 2>&1 && "
            f"TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
            f"{venv_python} -m scripts.train {train_args}",
            stream_output=True,
        )
        print("Remote training finished.", flush=True)

    def run_evaluation(
        self,
        eval_args: str,
        allow_cpu: bool = False,
        remote_report_path: str = f"{REMOTE_WORKSPACE}/model/logs/evaluation_remote.txt",
    ) -> None:
        """Create a uv-managed venv, install PyTorch + deps, and run evaluation."""
        venv_python = self._setup_remote_environment(allow_cpu=allow_cpu)

        print("Starting remote evaluation...", flush=True)
        self._run_ssh(
            " && ".join(
                [
                    "set -o pipefail",
                    f"cd {REMOTE_WORKSPACE}",
                    "mkdir -p model/logs",
                    "nvidia-smi > /dev/null 2>&1",
                    (
                        f"TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
                        f"{venv_python} -m scripts.evaluate {eval_args} "
                        f"2>&1 | tee {shlex.quote(remote_report_path)}"
                    ),
                ]
            ),
            stream_output=True,
        )
        print("Remote evaluation finished.", flush=True)

    def _setup_remote_ollama(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 11434,
    ) -> str:
        """Install/start Ollama on pod and ensure the requested model is available."""
        base_url = f"http://{host}:{port}"
        host_port = f"{host}:{port}"
        quoted_model = shlex.quote(model)

        print("Setting up Ollama on pod...", flush=True)
        self._run_ssh(
            "\n".join(
                [
                    'export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:$PATH"',
                    "if ! command -v ollama >/dev/null 2>&1; then",
                    "  curl -fsSL https://ollama.com/install.sh | sh",
                    "fi",
                    f"if ! curl -fsS {shlex.quote(base_url + '/api/tags')} >/dev/null 2>&1; then",
                    f"  nohup env OLLAMA_HOST={shlex.quote(host_port)} ollama serve >/tmp/ollama.log 2>&1 &",
                    "fi",
                    "for i in $(seq 1 90); do",
                    f"  if curl -fsS {shlex.quote(base_url + '/api/tags')} >/dev/null 2>&1; then",
                    "    break",
                    "  fi",
                    "  sleep 2",
                    "done",
                    f"curl -fsS {shlex.quote(base_url + '/api/tags')} >/dev/null",
                    f"env OLLAMA_HOST={shlex.quote(host_port)} ollama --version",
                    f"env OLLAMA_HOST={shlex.quote(host_port)} ollama pull {quoted_model}",
                ]
            ),
            stream_output=True,
        )

        print(f"Ollama ready at {base_url} with model {model}", flush=True)
        return base_url

    def _ensure_remote_java(self) -> None:
        """Ensure Java 11+ exists on the remote pod (required by OpenDataLoader)."""
        print("Checking Java runtime on pod...", flush=True)
        self._run_ssh(
            "\n".join(
                [
                    'export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:$PATH"',
                    "if command -v java >/dev/null 2>&1; then",
                    "  java -version",
                    "  exit 0",
                    "fi",
                    "if command -v apt-get >/dev/null 2>&1; then",
                    "  apt-get update",
                    "  DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jre-headless",
                    "fi",
                    "java -version",
                ]
            ),
            stream_output=True,
        )

    def _setup_remote_opendataloader_hybrid(
        self,
        host: str = "127.0.0.1",
        port: int = 5002,
    ) -> str:
        """Ensure OpenDataLoader hybrid server is running on the remote pod."""
        base_url = f"http://{host}:{port}"
        health_url = f"{base_url}/health"
        server_cmd = (
            f"{REMOTE_WORKSPACE}/.venv/bin/opendataloader-pdf-hybrid "
            f"--host {shlex.quote(host)} --port {int(port)}"
        )

        print("Setting up OpenDataLoader hybrid server on pod...", flush=True)
        self._run_ssh(
            "\n".join(
                [
                    'export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:$PATH"',
                    f"if ! curl -fsS {shlex.quote(health_url)} >/dev/null 2>&1; then",
                    f"  nohup {server_cmd} >/tmp/opendataloader-hybrid.log 2>&1 &",
                    "fi",
                    "for i in $(seq 1 90); do",
                    f"  if curl -fsS {shlex.quote(health_url)} >/dev/null 2>&1; then",
                    "    break",
                    "  fi",
                    "  sleep 2",
                    "done",
                    f"curl -fsS {shlex.quote(health_url)} >/dev/null",
                ]
            ),
            stream_output=True,
        )

        print(f"OpenDataLoader hybrid ready at {base_url}", flush=True)
        return base_url

    def _build_marker_llm_env(self) -> dict[str, str]:
        """Return marker/Ollama environment settings for remote commands."""
        defaults = {
            "MARKER_LLM_TIMEOUT": "900",
            "MARKER_LLM_MAX_RETRIES": "5",
            "MARKER_LLM_MAX_CONCURRENCY": "1",
            "MARKER_LLM_IMAGE_MIN_SIDE": "56",
            "MARKER_LLM_IMAGE_MAX_SIDE": "1536",
            "MARKER_LLM_ERROR_BODY_CHARS": "600",
            "MARKER_LLM_FAIL_FAST_KNOWN_ERRORS": "true",
            "MARKER_LLM_TEXT_ONLY_FALLBACK_ON_INVALID_IMAGES": "true",
        }
        return {key: os.environ.get(key, default) for key, default in defaults.items()}

    def _upload_single_file(self, local_path: Path, remote_path: str) -> None:
        """Upload one local file to an exact remote destination path."""
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"File does not exist: {local_path}")

        remote_dir = remote_path.rsplit("/", 1)[0]
        self._run_ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        self._scp_upload(local_path, remote_path, recursive=False)

    def _download_single_file(self, remote_path: str, local_path: Path) -> None:
        """Download one remote file to an exact local destination path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._scp_download(remote_path, local_path, recursive=False)

    def run_preprocess_single_remote(
        self,
        local_pdf_path: str,
        local_output_path: str,
        backend: str = "docling",
        batch_multiplier: int = 2,
        disable_table_recognition: bool = False,
        ollama_model: str = "qwen2.5vl:7b",
        ollama_host: str = "127.0.0.1",
        ollama_port: int = 11434,
        opendataloader_hybrid: bool = False,
        opendataloader_hybrid_mode: str | None = None,
        opendataloader_hybrid_url: str | None = None,
        opendataloader_hybrid_timeout: str | None = None,
        opendataloader_hybrid_fallback: bool = False,
        allow_cpu: bool = False,
    ) -> None:
        """Run single-PDF preprocessing remotely and download markdown."""
        venv_python = self._setup_remote_environment(allow_cpu=allow_cpu)
        backend = backend.strip().lower()

        marker_env_assignments = ""
        if backend == "marker":
            ollama_base_url = self._setup_remote_ollama(
                model=ollama_model,
                host=ollama_host,
                port=ollama_port,
            )

            marker_env = self._build_marker_llm_env()
            marker_env_assignments = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in marker_env.items()
            )
        elif backend == "opendataloader":
            self._ensure_remote_java()
            if opendataloader_hybrid:
                if opendataloader_hybrid_url:
                    from urllib.parse import urlparse

                    parsed = urlparse(opendataloader_hybrid_url)
                    host = (parsed.hostname or "").strip().lower()
                    if host in {"", "127.0.0.1", "localhost", "0.0.0.0"}:
                        port = parsed.port or 5002
                        opendataloader_hybrid_url = self._setup_remote_opendataloader_hybrid(
                            host="127.0.0.1",
                            port=port,
                        )
                else:
                    opendataloader_hybrid_url = self._setup_remote_opendataloader_hybrid(
                        host="127.0.0.1",
                        port=5002,
                    )
        elif backend == "docling":
            pass
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported backends: marker, opendataloader, docling"
            )

        local_pdf = Path(local_pdf_path).expanduser().resolve()
        local_output = Path(local_output_path).expanduser().resolve()
        remote_input = f"{REMOTE_WORKSPACE}/tmp/input_preprocess.pdf"
        remote_output = f"{REMOTE_WORKSPACE}/tmp/output_preprocess.md"

        print("Uploading source PDF...", flush=True)
        self._upload_single_file(local_pdf, remote_input)

        cmd_parts = [
            f"cd {REMOTE_WORKSPACE}",
            "mkdir -p tmp",
            "nvidia-smi > /dev/null 2>&1",
            (
                "TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
                + (
                    f"OLLAMA_BASE_URL={shlex.quote(ollama_base_url)} "
                    f"OLLAMA_MODEL={shlex.quote(ollama_model)} "
                    f"{marker_env_assignments} "
                    if backend == "marker"
                    else ""
                )
                +
                f"{venv_python} -m scripts.preprocess_single "
                f"--pdf {shlex.quote(remote_input)} "
                f"--output {shlex.quote(remote_output)} "
                f"--backend {shlex.quote(backend)} "
                f"--batch-multiplier {int(batch_multiplier)}"
            ),
        ]
        if backend == "marker" and disable_table_recognition:
            cmd_parts[-1] += " --disable-table-recognition"
        if backend == "opendataloader" and opendataloader_hybrid:
            cmd_parts[-1] += " --opendataloader-hybrid"
            if opendataloader_hybrid_mode:
                cmd_parts[-1] += (
                    f" --opendataloader-hybrid-mode "
                    f"{shlex.quote(opendataloader_hybrid_mode)}"
                )
            if opendataloader_hybrid_url:
                cmd_parts[-1] += (
                    f" --opendataloader-hybrid-url "
                    f"{shlex.quote(opendataloader_hybrid_url)}"
                )
            if opendataloader_hybrid_timeout:
                cmd_parts[-1] += (
                    f" --opendataloader-hybrid-timeout "
                    f"{shlex.quote(opendataloader_hybrid_timeout)}"
                )
            if opendataloader_hybrid_fallback:
                cmd_parts[-1] += " --opendataloader-hybrid-fallback"

        print("Starting remote single-file preprocessing...", flush=True)
        self._run_ssh(" && ".join(cmd_parts), stream_output=True)

        print("Downloading markdown artifact...", flush=True)
        self._download_single_file(remote_output, local_output)
        print(f"Markdown downloaded to {local_output}", flush=True)

    def run_classify_single_remote(
        self,
        local_pdf_path: str,
        local_output_path: str,
        checkpoint_path: str = "model/checkpoints/best",
        device: str | None = None,
        backend: str = "docling",
        ollama_model: str = "qwen2.5vl:7b",
        ollama_host: str = "127.0.0.1",
        ollama_port: int = 11434,
        opendataloader_hybrid: bool = False,
        opendataloader_hybrid_mode: str | None = None,
        opendataloader_hybrid_url: str | None = None,
        opendataloader_hybrid_timeout: str | None = None,
        opendataloader_hybrid_fallback: bool = False,
        allow_cpu: bool = False,
    ) -> None:
        """Run single-PDF classification remotely and download result JSON."""
        venv_python = self._setup_remote_environment(allow_cpu=allow_cpu)
        backend = backend.strip().lower()

        marker_env_assignments = ""
        if backend == "marker":
            ollama_base_url = self._setup_remote_ollama(
                model=ollama_model,
                host=ollama_host,
                port=ollama_port,
            )

            marker_env = self._build_marker_llm_env()
            marker_env_assignments = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in marker_env.items()
            )
        elif backend == "opendataloader":
            self._ensure_remote_java()
            if opendataloader_hybrid:
                if opendataloader_hybrid_url:
                    from urllib.parse import urlparse

                    parsed = urlparse(opendataloader_hybrid_url)
                    host = (parsed.hostname or "").strip().lower()
                    if host in {"", "127.0.0.1", "localhost", "0.0.0.0"}:
                        port = parsed.port or 5002
                        opendataloader_hybrid_url = self._setup_remote_opendataloader_hybrid(
                            host="127.0.0.1",
                            port=port,
                        )
                else:
                    opendataloader_hybrid_url = self._setup_remote_opendataloader_hybrid(
                        host="127.0.0.1",
                        port=5002,
                    )
        elif backend == "docling":
            pass
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported backends: marker, opendataloader, docling"
            )

        local_pdf = Path(local_pdf_path).expanduser().resolve()
        local_output = Path(local_output_path).expanduser().resolve()
        remote_input = f"{REMOTE_WORKSPACE}/tmp/input_classify.pdf"
        remote_output = f"{REMOTE_WORKSPACE}/tmp/output_classify.json"

        print("Uploading source PDF...", flush=True)
        self._upload_single_file(local_pdf, remote_input)

        device_arg = f"--device {shlex.quote(device)}" if device else ""

        classify_cmd = (
            "TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
            + (
                f"OLLAMA_BASE_URL={shlex.quote(ollama_base_url)} "
                f"OLLAMA_MODEL={shlex.quote(ollama_model)} "
                f"{marker_env_assignments} "
                if backend == "marker"
                else ""
            )
            +
            f"{venv_python} -m scripts.classify_single "
            f"--pdf {shlex.quote(remote_input)} "
            f"--checkpoint {shlex.quote(checkpoint_path)} "
            f"--output {shlex.quote(remote_output)} "
            f"--backend {shlex.quote(backend)}"
            + (f" {device_arg}" if device_arg else "")
        )
        if backend == "opendataloader" and opendataloader_hybrid:
            classify_cmd += " --opendataloader-hybrid"
            if opendataloader_hybrid_mode:
                classify_cmd += (
                    f" --opendataloader-hybrid-mode "
                    f"{shlex.quote(opendataloader_hybrid_mode)}"
                )
            if opendataloader_hybrid_url:
                classify_cmd += (
                    f" --opendataloader-hybrid-url "
                    f"{shlex.quote(opendataloader_hybrid_url)}"
                )
            if opendataloader_hybrid_timeout:
                classify_cmd += (
                    f" --opendataloader-hybrid-timeout "
                    f"{shlex.quote(opendataloader_hybrid_timeout)}"
                )
            if opendataloader_hybrid_fallback:
                classify_cmd += " --opendataloader-hybrid-fallback"

        print("Starting remote single-file classification...", flush=True)
        self._run_ssh(
            " && ".join(
                [
                    f"cd {REMOTE_WORKSPACE}",
                    "mkdir -p tmp",
                    "nvidia-smi > /dev/null 2>&1",
                    classify_cmd,
                ]
            ),
            stream_output=True,
        )

        print("Downloading classification artifact...", flush=True)
        self._download_single_file(remote_output, local_output)
        print(f"Classification JSON downloaded to {local_output}", flush=True)

    # ------------------------------------------------------------------
    # Artifact download
    # ------------------------------------------------------------------

    def download_model(self, local_dir: str = "model/checkpoints/best") -> None:
        print("Downloading model artifacts...", flush=True)
        target = Path(local_dir).resolve()
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._scp_download(f"{REMOTE_WORKSPACE}/model/checkpoints/best", target.parent, recursive=True)
        print(f"Model downloaded to {local_dir}", flush=True)

    def download_logs(self, local_dir: str = "model/logs") -> None:
        print("Downloading training logs...", flush=True)
        target = Path(local_dir).resolve()
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._scp_download(f"{REMOTE_WORKSPACE}/model/logs", target.parent, recursive=True)
        print(f"Logs downloaded to {local_dir}", flush=True)

    def download_evaluation_report(
        self,
        local_path: str = "model/logs/evaluation_remote.txt",
        remote_path: str = f"{REMOTE_WORKSPACE}/model/logs/evaluation_remote.txt",
    ) -> None:
        print("Downloading evaluation report...", flush=True)
        target = Path(local_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._scp_download(remote_path, target, recursive=False)
        print(f"Evaluation report downloaded to {local_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote training/evaluation on RunPod GPU")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Remote task to run",
    )
    parser.add_argument("--data", default="data/output", help="Local data directory to upload")
    parser.add_argument("--gpu", default="NVIDIA GeForce RTX 5090", help="GPU type ID (e.g. 'NVIDIA GeForce RTX 5090')")
    parser.add_argument("--disk", type=int, default=50, help="Container disk size (GB)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--checkpoint", default="model/checkpoints/best", help="Checkpoint path for evaluation")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Split for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument(
        "--report-path",
        default="model/logs/evaluation_remote.txt",
        help="Local destination path for downloaded evaluation report",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow remote task to proceed even when CUDA preflight fails",
    )
    parser.add_argument("--no-terminate", action="store_true", help="Keep pod running after remote task")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    ssh_key_path = os.environ.get("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")

    if not api_key:
        print("Error: RUNPOD_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    trainer = RunPodTrainer(
        api_key=api_key,
        ssh_key_path=ssh_key_path,
        gpu_type=args.gpu,
    )

    try:
        trainer.create_pod_with_cuda_check(disk_size=args.disk)
        trainer.sync_project(".")
        trainer.sync_data(args.data)

        if args.mode == "train":
            train_args_parts = ["--data data/output"]
            if args.epochs is not None:
                train_args_parts.append(f"--epochs {args.epochs}")
            if args.batch_size is not None:
                train_args_parts.append(f"--batch-size {args.batch_size}")
            train_args = " ".join(train_args_parts)

            trainer.run_training(train_args, allow_cpu=args.allow_cpu)
            trainer.download_model()
            trainer.download_logs()
            print("\nTraining complete! Artifacts downloaded.", flush=True)
        else:
            eval_args_parts = [
                "--data data/output",
                f"--checkpoint {shlex.quote(args.checkpoint)}",
                f"--split {args.split}",
                f"--seed {args.seed}",
            ]
            eval_args = " ".join(eval_args_parts)

            trainer.run_evaluation(eval_args, allow_cpu=args.allow_cpu)
            trainer.download_evaluation_report(local_path=args.report_path)
            print("\nEvaluation complete! Report downloaded.", flush=True)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if not args.no_terminate:
            trainer.terminate_pod()
        else:
            print(f"\nPod {trainer.pod_id} left running. Terminate manually when done.", flush=True)


if __name__ == "__main__":
    main()
