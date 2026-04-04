#!/usr/bin/env python3
"""
Simplify — terminal menu orchestrator.

Navigate with ↑/↓ arrow keys and press Enter to select.
Run:  python menu.py
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from pathlib import Path

try:
    import questionary
    from questionary import Choice, Separator
except ImportError:
    sys.exit("questionary is not installed.  Run:  pip install 'questionary>=2.0'")

# ── Logging ────────────────────────────────────────────────────────────────

VERBOSE = False
_handler: logging.Handler | None = None

_COLORS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColorFmt(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, "")
        ts    = self.formatTime(record, "%H:%M:%S")
        lvl   = f"{color}{record.levelname:<8}{_RESET}"
        msg   = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{ts} [{lvl}] {msg}"


def _apply_logging() -> None:
    global _handler
    root = logging.getLogger()
    if _handler:
        root.removeHandler(_handler)
    root.setLevel(logging.DEBUG if VERBOSE else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_ColorFmt())
    root.addHandler(h)
    _handler = h


log = logging.getLogger("simplify")


# ── Banner ─────────────────────────────────────────────────────────────────

_BANNER = (
    "\033[36m"
    "\nTranscript Classifier"
    "\033[0m"
)


# ── Small helpers ──────────────────────────────────────────────────────────

def _int(s: str, default: int) -> int:
    try:
        return int(s.strip()) if s.strip() else default
    except ValueError:
        return default


def _float(s: str, default: float) -> float:
    try:
        return float(s.strip()) if s.strip() else default
    except ValueError:
        return default


def _pause() -> None:
    input("\n  Press Enter to return to menu...")


def _ask_preprocess_backend(prompt: str = "Choose PDF-to-Markdown backend") -> str | None:
    return questionary.select(
        prompt,
        choices=[
            Choice("Docling (current default)", value="docling"),
            Choice("Marker", value="marker"),
            Choice("OpenDataLoader", value="opendataloader"),
        ],
        default="docling",
    ).ask()


def _ask_opendataloader_hybrid_options() -> dict | None:
    hybrid_enabled = questionary.confirm(
        "Enable OpenDataLoader hybrid mode?",
        default=True,
    ).ask()
    if hybrid_enabled is None:
        return None

    result = {
        "opendataloader_hybrid": bool(hybrid_enabled),
        "opendataloader_hybrid_mode": None,
        "opendataloader_hybrid_url": None,
        "opendataloader_hybrid_timeout": None,
        "opendataloader_hybrid_fallback": False,
    }
    if not hybrid_enabled:
        return result

    hybrid_mode = questionary.select(
        "Hybrid mode",
        choices=["auto", "full"],
        default="auto",
    ).ask()
    if hybrid_mode is None:
        return None

    default_url = os.environ.get("OPENDATALOADER_HYBRID_URL", "http://127.0.0.1:5002")
    hybrid_url = questionary.text("Hybrid server URL", default=default_url).ask()
    if hybrid_url is None:
        return None

    timeout_default = os.environ.get("OPENDATALOADER_HYBRID_TIMEOUT", "")
    hybrid_timeout = questionary.text(
        "Hybrid timeout in ms (blank = library default)",
        default=timeout_default,
    ).ask()
    if hybrid_timeout is None:
        return None

    hybrid_fallback = questionary.confirm(
        "Enable Java fallback if hybrid backend fails?",
        default=False,
    ).ask()
    if hybrid_fallback is None:
        return None

    result["opendataloader_hybrid_mode"] = hybrid_mode
    result["opendataloader_hybrid_url"] = hybrid_url.strip() or None
    result["opendataloader_hybrid_timeout"] = hybrid_timeout.strip() or None
    result["opendataloader_hybrid_fallback"] = bool(hybrid_fallback)
    return result


# ── Actions ────────────────────────────────────────────────────────────────

def run_generate() -> None:
    print()
    log.info("Configure transcript generation")

    output = questionary.text("Output directory", default="data/output").ask()
    if output is None:
        return

    count = questionary.text("Transcripts per template", default="200").ask()
    if count is None:
        return

    render = questionary.confirm("Render PDFs?", default=True).ask()
    if render is None:
        return

    md_method = questionary.select(
        "Markdown conversion method:",
        choices=[
            "HTML → Markdown (fast, recommended)",
            "PDF → Markdown (backend selectable)",
            "None",
        ],
        default="HTML → Markdown (fast, recommended)",
    ).ask()
    if md_method is None:
        return
    html_to_md = md_method.startswith("HTML")
    prep_md = md_method.startswith("PDF")

    preprocess_backend = "docling"
    preprocess_use_llm = True
    preprocess_batch_multiplier = 2
    preprocess_disable_table_recognition = False
    preprocess_opendataloader_hybrid = False
    preprocess_opendataloader_hybrid_mode = None
    preprocess_opendataloader_hybrid_url = None
    preprocess_opendataloader_hybrid_timeout = None
    preprocess_opendataloader_hybrid_fallback = False

    if prep_md:
        backend = _ask_preprocess_backend("PDF-to-Markdown backend for generation")
        if backend is None:
            return
        preprocess_backend = backend

        if backend == "marker":
            preprocess_use_llm = questionary.confirm(
                "Enable LLM-enhanced processing? (uses Ollama locally)",
                default=True,
            ).ask()
            if preprocess_use_llm is None:
                return

            bm = questionary.text(
                "Batch multiplier (higher = faster but more RAM)",
                default="2",
            ).ask()
            if bm is None:
                return
            preprocess_batch_multiplier = _int(bm, 2)

            preprocess_disable_table_recognition = questionary.confirm(
                "Disable table recognition?",
                default=False,
            ).ask()
            if preprocess_disable_table_recognition is None:
                return
        elif backend == "opendataloader":
            hybrid_opts = _ask_opendataloader_hybrid_options()
            if hybrid_opts is None:
                return
            preprocess_opendataloader_hybrid = hybrid_opts["opendataloader_hybrid"]
            preprocess_opendataloader_hybrid_mode = hybrid_opts["opendataloader_hybrid_mode"]
            preprocess_opendataloader_hybrid_url = hybrid_opts["opendataloader_hybrid_url"]
            preprocess_opendataloader_hybrid_timeout = hybrid_opts["opendataloader_hybrid_timeout"]
            preprocess_opendataloader_hybrid_fallback = hybrid_opts["opendataloader_hybrid_fallback"]

    aug = questionary.confirm("Apply noise augmentation?", default=True).ask()
    if aug is None:
        return

    noise_intensity = 1.0
    if aug:
        ni = questionary.text("Noise intensity (0–1)", default="1.0").ask()
        if ni is None:
            return
        noise_intensity = _float(ni, 1.0)

    workers = questionary.text("Parallel workers (0 = auto)", default="1").ask()
    if workers is None:
        return

    split_c = questionary.confirm("Disjoint train/test course pools?", default=False).ask()
    if split_c is None:
        return

    tmpl_str = questionary.text(
        "Templates to use (comma-sep, blank = all)", default=""
    ).ask()
    if tmpl_str is None:
        return
    templates = [t.strip() for t in tmpl_str.split(",") if t.strip()] or None

    seed = questionary.text("Random seed", default="42").ask()
    if seed is None:
        return

    print()
    log.info("Starting transcript generation (this may take a while)…")
    try:
        from transcript_generator.config import GenerationConfig
        from transcript_generator.generator import TranscriptGenerator

        config = GenerationConfig(
            count_per_template=_int(count, 200),
            render_pdf=render or prep_md,
            pdf_extract=False,
            preprocess_md=prep_md,
            html_to_md=html_to_md,
            preprocess_backend=preprocess_backend,
            preprocess_use_llm=preprocess_use_llm,
            preprocess_batch_multiplier=preprocess_batch_multiplier,
            preprocess_disable_table_recognition=preprocess_disable_table_recognition,
            preprocess_opendataloader_hybrid=preprocess_opendataloader_hybrid,
            preprocess_opendataloader_hybrid_mode=preprocess_opendataloader_hybrid_mode,
            preprocess_opendataloader_hybrid_url=preprocess_opendataloader_hybrid_url,
            preprocess_opendataloader_hybrid_timeout=preprocess_opendataloader_hybrid_timeout,
            preprocess_opendataloader_hybrid_fallback=preprocess_opendataloader_hybrid_fallback,
            augment_noise=aug,
            noise_intensity=noise_intensity,
            workers=_int(workers, 1),
            split_courses=split_c,
            template_names=templates,
            seed=_int(seed, 42),
        )
        gen = TranscriptGenerator(output)
        manifest = gen.generate(config)
        log.info(
            "Done!  Generated %d transcripts → %s/",
            manifest["total_transcripts"],
            output,
        )
    except Exception as exc:
        log.error("Generation failed: %s", exc, exc_info=VERBOSE)

    _pause()


def run_train() -> None:
    print()
    log.info("Configure local training")

    # Import constants only (no heavy deps)
    from model.config import BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, LEARNING_RATE, NUM_EPOCHS, SEED

    data = questionary.text("Data directory", default=DATA_DIR).ask()
    if data is None:
        return

    out = questionary.text("Checkpoint output directory", default=CHECKPOINT_DIR).ask()
    if out is None:
        return

    epochs = questionary.text("Max epochs", default=str(NUM_EPOCHS)).ask()
    if epochs is None:
        return

    bs = questionary.text("Batch size", default=str(BATCH_SIZE)).ask()
    if bs is None:
        return

    lr = questionary.text("Learning rate", default=str(LEARNING_RATE)).ask()
    if lr is None:
        return

    seed = questionary.text("Random seed", default=str(SEED)).ask()
    if seed is None:
        return

    log_every = questionary.text("Log every N steps", default="5").ask()
    if log_every is None:
        return

    print()
    log.info("Starting training…  (device will be auto-detected)")
    try:
        from model.trainer import train

        train(
            data_dir=data,
            output_dir=out,
            num_epochs=_int(epochs, NUM_EPOCHS),
            batch_size=_int(bs, BATCH_SIZE),
            learning_rate=_float(lr, LEARNING_RATE),
            seed=_int(seed, SEED),
            log_every=_int(log_every, 5),
        )
        log.info("Training complete!  Best checkpoint saved to %s/best/", out)
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=VERBOSE)

    _pause()


def run_preprocess() -> None:
    print()
    log.info("Configure PDF → Markdown preprocessing")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    input_dir = questionary.text(
        "Directory containing generated PDFs", default="data/output"
    ).ask()
    if input_dir is None:
        return

    out_dir = questionary.text(
        "Output directory for .md files (blank = same as input)", default=""
    ).ask()
    if out_dir is None:
        return

    pattern = questionary.text("File glob pattern", default="**/*.pdf").ask()
    if pattern is None:
        return

    skip = questionary.confirm(
        "Skip PDFs that already have a .md file? (safe to resume interrupted runs)",
        default=True,
    ).ask()
    if skip is None:
        return

    backend = _ask_preprocess_backend()
    if backend is None:
        return

    use_llm = True
    batch_mult = "2"
    disable_table_recognition = False
    opendataloader_hybrid = False
    opendataloader_hybrid_mode = None
    opendataloader_hybrid_url = None
    opendataloader_hybrid_timeout = None
    opendataloader_hybrid_fallback = False

    if backend == "marker":
        use_llm = questionary.confirm(
            "Enable LLM-enhanced processing? (uses Ollama locally)",
            default=True,
        ).ask()
        if use_llm is None:
            return

        batch_mult = questionary.text(
            "Batch multiplier (higher = faster but more RAM; 2 = default, 4–8 for Apple Silicon)",
            default="2",
        ).ask()
        if batch_mult is None:
            return

        disabled_parts = questionary.checkbox(
            "Disable Marker components (optional)",
            choices=[
                Choice(
                    "Table recognition (skip TableProcessor; often better for transcript-style tables)",
                    value="table_recognition",
                ),
            ],
        ).ask()
        if disabled_parts is None:
            return
        disable_table_recognition = "table_recognition" in disabled_parts
    elif backend == "opendataloader":
        hybrid_opts = _ask_opendataloader_hybrid_options()
        if hybrid_opts is None:
            return
        opendataloader_hybrid = hybrid_opts["opendataloader_hybrid"]
        opendataloader_hybrid_mode = hybrid_opts["opendataloader_hybrid_mode"]
        opendataloader_hybrid_url = hybrid_opts["opendataloader_hybrid_url"]
        opendataloader_hybrid_timeout = hybrid_opts["opendataloader_hybrid_timeout"]
        opendataloader_hybrid_fallback = hybrid_opts["opendataloader_hybrid_fallback"]

    print()
    log.info("Initializing preprocessor backend (first run may be slow)…")
    try:
        from preprocessor import TranscriptPreprocessor

        pre = TranscriptPreprocessor(
            backend=backend,
            batch_multiplier=_int(batch_mult, 2),
            disable_table_recognition=disable_table_recognition,
            use_llm=use_llm,
            opendataloader_hybrid=opendataloader_hybrid,
            opendataloader_hybrid_mode=opendataloader_hybrid_mode,
            opendataloader_hybrid_url=opendataloader_hybrid_url,
            opendataloader_hybrid_timeout=opendataloader_hybrid_timeout,
            opendataloader_hybrid_fallback=opendataloader_hybrid_fallback,
        )
        results = pre.convert_bulk(
            input_dir=input_dir,
            output_dir=out_dir.strip() or None,
            pattern=pattern,
            skip_existing=skip,
        )
        skipped = sum(1 for r in results if r.get("skipped"))
        ok      = sum(1 for r in results if r.get("markdown") and not r.get("skipped"))
        failed  = sum(1 for r in results if r.get("error"))
        log.info("Done!  Converted %d  |  Skipped %d  |  Failed %d", ok, skipped, failed)
        if failed:
            for r in results:
                if r.get("error"):
                    log.warning("  %s — %s", r["pdf"].name, r["error"])
    except Exception as exc:
        log.error("Preprocessing failed: %s", exc, exc_info=VERBOSE)

    _pause()


_GPU_TYPES_FALLBACK = [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA L40",
    "NVIDIA H100 80GB HBM3",
]


def run_preprocess_remote_single() -> None:
    print()
    log.info("Configure RunPod single-PDF preprocessing")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        log.error("RUNPOD_API_KEY is not set.  Add it to .env or export it.")
        _pause()
        return

    ssh_key = os.environ.get("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")

    pdf_path = questionary.text("PDF path").ask()
    if pdf_path is None:
        return
    pdf_file = Path(pdf_path.strip().strip("'\"")).expanduser().resolve()
    if not pdf_file.exists():
        log.error("File not found: %s", pdf_file)
        _pause()
        return

    out_default = str(pdf_file.with_suffix(".md"))
    out_file = questionary.text("Local markdown output path", default=out_default).ask()
    if out_file is None:
        return
    out_target = Path(out_file.strip() or out_default).expanduser().resolve()

    batch_mult = questionary.text(
        "Batch multiplier (higher = faster but more RAM)",
        default="2",
    ).ask()
    if batch_mult is None:
        return

    backend = _ask_preprocess_backend()
    if backend is None:
        return

    disable_table = False
    default_ollama_model = os.environ.get("RUNPOD_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
    ollama_model = default_ollama_model
    opendataloader_hybrid = False
    opendataloader_hybrid_mode = None
    opendataloader_hybrid_url = None
    opendataloader_hybrid_timeout = None
    opendataloader_hybrid_fallback = False

    if backend == "marker":
        disable_table = questionary.confirm(
            "Disable table recognition?",
            default=False,
        ).ask()
        if disable_table is None:
            return

        ollama_model = questionary.text(
            "Ollama model to pull on RunPod",
            default=default_ollama_model,
        ).ask()
        if ollama_model is None:
            return
    elif backend == "opendataloader":
        hybrid_opts = _ask_opendataloader_hybrid_options()
        if hybrid_opts is None:
            return
        opendataloader_hybrid = hybrid_opts["opendataloader_hybrid"]
        opendataloader_hybrid_mode = hybrid_opts["opendataloader_hybrid_mode"]
        opendataloader_hybrid_url = hybrid_opts["opendataloader_hybrid_url"]
        opendataloader_hybrid_timeout = hybrid_opts["opendataloader_hybrid_timeout"]
        opendataloader_hybrid_fallback = hybrid_opts["opendataloader_hybrid_fallback"]

    from scripts.runpod_train import RunPodTrainer
    print("Fetching available GPUs...", flush=True)
    try:
        _tmp = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key)
        gpu_choices = _tmp.list_available_gpus()
    except Exception:
        gpu_choices = []
    if not gpu_choices:
        log.warning("Could not fetch GPU availability — showing all types.")
        gpu_choices = _GPU_TYPES_FALLBACK

    default_gpu = next((g for g in gpu_choices if "5090" in g), gpu_choices[0])
    gpu = questionary.select(
        "GPU type (secure cloud, available now)",
        choices=gpu_choices,
        default=default_gpu,
    ).ask()
    if gpu is None:
        return

    disk = questionary.text("Container disk size (GB)", default="50").ask()
    if disk is None:
        return

    no_term = questionary.confirm("Keep pod running after preprocessing?", default=False).ask()
    if no_term is None:
        return

    ollama_host = os.environ.get("RUNPOD_OLLAMA_HOST", "127.0.0.1")
    ollama_port = _int(os.environ.get("RUNPOD_OLLAMA_PORT", "11434"), 11434)

    print()
    log.info("Launching RunPod pod (GPU: %s)…", gpu)
    try:
        trainer = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key, gpu_type=gpu)
        try:
            trainer.create_pod_with_cuda_check(disk_size=_int(disk, 50))
            trainer.sync_code(".")
            trainer.run_preprocess_single_remote(
                local_pdf_path=str(pdf_file),
                local_output_path=str(out_target),
                backend=backend,
                batch_multiplier=_int(batch_mult, 2),
                disable_table_recognition=disable_table,
                ollama_model=ollama_model.strip() or default_ollama_model,
                ollama_host=ollama_host,
                ollama_port=ollama_port,
                opendataloader_hybrid=opendataloader_hybrid,
                opendataloader_hybrid_mode=opendataloader_hybrid_mode,
                opendataloader_hybrid_url=opendataloader_hybrid_url,
                opendataloader_hybrid_timeout=opendataloader_hybrid_timeout,
                opendataloader_hybrid_fallback=opendataloader_hybrid_fallback,
            )
            log.info("Remote preprocessing complete!  Markdown saved to %s", out_target)
        finally:
            if not no_term:
                trainer.terminate_pod()
            elif trainer.pod_id:
                log.warning(
                    "Pod %s left running — remember to terminate it!", trainer.pod_id
                )
    except Exception as exc:
        log.error("RunPod preprocessing failed: %s", exc, exc_info=VERBOSE)

    _pause()


def run_train_remote() -> None:
    print()
    log.info("Configure RunPod remote training")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        log.error("RUNPOD_API_KEY is not set.  Add it to .env or export it.")
        _pause()
        return

    ssh_key = os.environ.get("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")

    data = questionary.text("Local data directory", default="data/output").ask()
    if data is None:
        return

    from scripts.runpod_train import RunPodTrainer
    print("Fetching available GPUs...", flush=True)
    try:
        _tmp = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key)
        gpu_choices = _tmp.list_available_gpus()
    except Exception:
        gpu_choices = []
    if not gpu_choices:
        log.warning("Could not fetch GPU availability — showing all types.")
        gpu_choices = _GPU_TYPES_FALLBACK

    default_gpu = next(
        (g for g in gpu_choices if "5090" in g),
        gpu_choices[0],
    )
    gpu = questionary.select("GPU type (secure cloud, available now)", choices=gpu_choices, default=default_gpu).ask()
    if gpu is None:
        return

    disk = questionary.text("Container disk size (GB)", default="50").ask()
    if disk is None:
        return

    ep_s = questionary.text(
        "Override epochs? (blank = inherit defaults)", default=""
    ).ask()
    if ep_s is None:
        return

    bs_s = questionary.text(
        "Override batch size? (blank = inherit defaults)", default=""
    ).ask()
    if bs_s is None:
        return

    no_term = questionary.confirm("Keep pod running after training?", default=False).ask()
    if no_term is None:
        return

    train_parts = ["--data data/output"]
    if ep_s.strip():
        train_parts.append(f"--epochs {ep_s.strip()}")
    if bs_s.strip():
        train_parts.append(f"--batch-size {bs_s.strip()}")
    train_args = " ".join(train_parts)

    print()
    log.info("Launching RunPod pod (GPU: %s)…", gpu)
    try:
        trainer = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key, gpu_type=gpu)
        try:
            trainer.create_pod_with_cuda_check(disk_size=_int(disk, 50))
            trainer.sync_code(".")
            trainer.sync_data(data)
            trainer.run_training(train_args)
            trainer.download_model()
            trainer.download_logs()
            log.info("Remote training complete!  Model saved to model/checkpoints/best/")
        finally:
            if not no_term:
                trainer.terminate_pod()
            elif trainer.pod_id:
                log.warning(
                    "Pod %s left running — remember to terminate it!", trainer.pod_id
                )
    except Exception as exc:
        log.error("RunPod training failed: %s", exc, exc_info=VERBOSE)

    _pause()


def run_evaluate_remote() -> None:
    print()
    log.info("Configure RunPod remote evaluation")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        log.error("RUNPOD_API_KEY is not set.  Add it to .env or export it.")
        _pause()
        return

    ssh_key = os.environ.get("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")

    from model.config import BEST_CHECKPOINT_DIR, SEED

    data = questionary.text("Local data directory", default="data/output").ask()
    if data is None:
        return

    ckpt = questionary.text("Checkpoint path", default=BEST_CHECKPOINT_DIR).ask()
    if ckpt is None:
        return

    split = questionary.select(
        "Split to evaluate", choices=["test", "val"], default="test"
    ).ask()
    if split is None:
        return

    seed = questionary.text("Random seed", default=str(SEED)).ask()
    if seed is None:
        return

    report_path = questionary.text(
        "Local evaluation report path",
        default="model/logs/evaluation_remote.txt",
    ).ask()
    if report_path is None:
        return

    from scripts.runpod_train import RunPodTrainer
    print("Fetching available GPUs...", flush=True)
    try:
        _tmp = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key)
        gpu_choices = _tmp.list_available_gpus()
    except Exception:
        gpu_choices = []
    if not gpu_choices:
        log.warning("Could not fetch GPU availability — showing all types.")
        gpu_choices = _GPU_TYPES_FALLBACK

    default_gpu = next(
        (g for g in gpu_choices if "5090" in g),
        gpu_choices[0],
    )
    gpu = questionary.select("GPU type (secure cloud, available now)", choices=gpu_choices, default=default_gpu).ask()
    if gpu is None:
        return

    disk = questionary.text("Container disk size (GB)", default="50").ask()
    if disk is None:
        return

    no_term = questionary.confirm("Keep pod running after evaluation?", default=False).ask()
    if no_term is None:
        return

    eval_parts = [
        "--data data/output",
        f"--checkpoint {shlex.quote(ckpt.strip())}",
        f"--split {split}",
        f"--seed {_int(seed, SEED)}",
    ]
    eval_args = " ".join(eval_parts)

    report_local = report_path.strip() or "model/logs/evaluation_remote.txt"

    print()
    log.info("Launching RunPod pod (GPU: %s)…", gpu)
    try:
        trainer = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key, gpu_type=gpu)
        try:
            trainer.create_pod_with_cuda_check(disk_size=_int(disk, 50))
            trainer.sync_code(".")
            trainer.sync_data(data)
            trainer.run_evaluation(eval_args)
            trainer.download_evaluation_report(local_path=report_local)
            log.info("Remote evaluation complete!  Report saved to %s", report_local)
        finally:
            if not no_term:
                trainer.terminate_pod()
            elif trainer.pod_id:
                log.warning(
                    "Pod %s left running — remember to terminate it!", trainer.pod_id
                )
    except Exception as exc:
        log.error("RunPod evaluation failed: %s", exc, exc_info=VERBOSE)

    _pause()


def run_evaluate() -> None:
    print()
    log.info("Configure model evaluation")

    from model.config import BEST_CHECKPOINT_DIR, DATA_DIR, SEED

    data = questionary.text("Data directory", default=DATA_DIR).ask()
    if data is None:
        return

    ckpt = questionary.text("Checkpoint path", default=BEST_CHECKPOINT_DIR).ask()
    if ckpt is None:
        return

    split = questionary.select(
        "Split to evaluate", choices=["test", "val"], default="test"
    ).ask()
    if split is None:
        return

    seed = questionary.text("Random seed", default=str(SEED)).ask()
    if seed is None:
        return

    print()
    log.info("Loading data from %s…", data)
    try:
        from seqeval.metrics import classification_report
        from seqeval.metrics import f1_score as seqeval_f1

        from model.dataset import group_by_template, split_by_template
        from model.predictor import NERPredictor
        from pipeline.reconstructor import reconstruct_semesters

        _, val_s, test_s, _ = split_by_template(data_dir=data, seed=_int(seed, SEED))
        eval_s = test_s if split == "test" else val_s
        log.info("Evaluating on %s split: %d samples", split, len(eval_s))

        log.info("Loading model from %s…", ckpt)
        predictor = NERPredictor(ckpt)

        log.info("Running predictions…")
        all_true, all_pred = [], []
        for sample in eval_s:
            all_true.append(sample["ner_tags"])
            all_pred.append(predictor.predict(sample["tokens"]))

        print("\n" + "=" * 60)
        print("Entity-level Classification Report:")
        print("=" * 60)
        print(classification_report(all_true, all_pred))

        f1 = seqeval_f1(all_true, all_pred)
        log.info("Overall Entity F1:  %.4f", f1)

        correct = sum(
            t == p
            for ts, ps in zip(all_true, all_pred)
            for t, p in zip(ts, ps)
        )
        total = sum(len(ts) for ts in all_true)
        log.info("Token Accuracy:     %.4f", correct / total if total else 0.0)

        log.info("Computing course extraction rate…")
        total_c = perfect_c = 0
        for sample in eval_s:
            true_tags = sample["ner_tags"]
            pred_tags = predictor.predict(sample["tokens"])
            tc = {
                (c.get("code"), c.get("name"), c.get("grade"))
                for s in reconstruct_semesters(sample["tokens"], true_tags)
                for c in s["courses"]
            }
            pc = {
                (c.get("code"), c.get("name"), c.get("grade"))
                for s in reconstruct_semesters(sample["tokens"], pred_tags)
                for c in s["courses"]
            }
            total_c   += len(tc)
            perfect_c += len(tc & pc)
        rate = perfect_c / total_c if total_c else 0.0
        log.info(
            "Course Extraction Rate: %.4f  (%d / %d)", rate, perfect_c, total_c
        )

        print("\nPer-template F1:")
        groups  = group_by_template(eval_s)
        results = {}
        for tid, samps in sorted(groups.items()):
            tt = [s["ner_tags"] for s in samps]
            tp = [predictor.predict(s["tokens"]) for s in samps]
            results[tid] = seqeval_f1(tt, tp)
            print(f"  {tid}: {results[tid]:.4f}  ({len(samps)} samples)")

        worst = sorted(results.items(), key=lambda x: x[1])[:3]
        print("\nWorst 3 templates:")
        for tid, v in worst:
            print(f"  {tid}: {v:.4f}")

    except Exception as exc:
        log.error("Evaluation failed: %s", exc, exc_info=VERBOSE)

    _pause()


def _checkpoint_path_for_remote(ckpt_input: str) -> str:
    """Map a local checkpoint path to the equivalent path inside REMOTE_WORKSPACE."""
    raw = ckpt_input.strip()
    if not raw:
        raise ValueError("Checkpoint path is empty")

    ckpt_path = Path(raw).expanduser()
    if not ckpt_path.is_absolute():
        return str(ckpt_path)

    repo_root = Path.cwd().resolve()
    try:
        rel = ckpt_path.resolve().relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Checkpoint path must be inside the project directory") from exc
    return str(rel)


def run_classify_remote() -> None:
    print()
    log.info("Configure RunPod single-PDF classification")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        log.error("RUNPOD_API_KEY is not set.  Add it to .env or export it.")
        _pause()
        return

    ssh_key = os.environ.get("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")

    from model.config import BEST_CHECKPOINT_DIR

    pdf_path = questionary.text("PDF path").ask()
    if pdf_path is None:
        return
    pdf_file = Path(pdf_path.strip().strip("'\"")).expanduser().resolve()
    if not pdf_file.exists():
        log.error("File not found: %s", pdf_file)
        _pause()
        return

    ckpt = questionary.text("Checkpoint path", default=BEST_CHECKPOINT_DIR).ask()
    if ckpt is None:
        return
    local_ckpt = Path(ckpt.strip()).expanduser()
    if not local_ckpt.is_absolute():
        local_ckpt = Path.cwd() / local_ckpt
    if not local_ckpt.exists():
        log.error("Checkpoint path not found: %s", local_ckpt)
        _pause()
        return
    try:
        remote_ckpt = _checkpoint_path_for_remote(ckpt)
    except ValueError as exc:
        log.error("%s", exc)
        _pause()
        return

    device = questionary.text("Device (blank = auto-detect)", default="").ask()
    if device is None:
        return

    out_default = str(pdf_file.with_suffix(".json"))
    out_file = questionary.text("Local JSON output path", default=out_default).ask()
    if out_file is None:
        return
    out_target = Path(out_file.strip() or out_default).expanduser().resolve()

    backend = _ask_preprocess_backend()
    if backend is None:
        return

    default_ollama_model = os.environ.get("RUNPOD_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
    ollama_model = default_ollama_model
    opendataloader_hybrid = False
    opendataloader_hybrid_mode = None
    opendataloader_hybrid_url = None
    opendataloader_hybrid_timeout = None
    opendataloader_hybrid_fallback = False

    if backend == "marker":
        ollama_model = questionary.text(
            "Ollama model to pull on RunPod",
            default=default_ollama_model,
        ).ask()
        if ollama_model is None:
            return
    elif backend == "opendataloader":
        hybrid_opts = _ask_opendataloader_hybrid_options()
        if hybrid_opts is None:
            return
        opendataloader_hybrid = hybrid_opts["opendataloader_hybrid"]
        opendataloader_hybrid_mode = hybrid_opts["opendataloader_hybrid_mode"]
        opendataloader_hybrid_url = hybrid_opts["opendataloader_hybrid_url"]
        opendataloader_hybrid_timeout = hybrid_opts["opendataloader_hybrid_timeout"]
        opendataloader_hybrid_fallback = hybrid_opts["opendataloader_hybrid_fallback"]

    from scripts.runpod_train import RunPodTrainer
    print("Fetching available GPUs...", flush=True)
    try:
        _tmp = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key)
        gpu_choices = _tmp.list_available_gpus()
    except Exception:
        gpu_choices = []
    if not gpu_choices:
        log.warning("Could not fetch GPU availability — showing all types.")
        gpu_choices = _GPU_TYPES_FALLBACK

    default_gpu = next((g for g in gpu_choices if "5090" in g), gpu_choices[0])
    gpu = questionary.select(
        "GPU type (secure cloud, available now)",
        choices=gpu_choices,
        default=default_gpu,
    ).ask()
    if gpu is None:
        return

    disk = questionary.text("Container disk size (GB)", default="50").ask()
    if disk is None:
        return

    no_term = questionary.confirm("Keep pod running after classification?", default=False).ask()
    if no_term is None:
        return

    ollama_host = os.environ.get("RUNPOD_OLLAMA_HOST", "127.0.0.1")
    ollama_port = _int(os.environ.get("RUNPOD_OLLAMA_PORT", "11434"), 11434)

    print()
    log.info("Launching RunPod pod (GPU: %s)…", gpu)
    try:
        trainer = RunPodTrainer(api_key=api_key, ssh_key_path=ssh_key, gpu_type=gpu)
        try:
            trainer.create_pod_with_cuda_check(disk_size=_int(disk, 50))
            trainer.sync_code(".")
            trainer.run_classify_single_remote(
                local_pdf_path=str(pdf_file),
                local_output_path=str(out_target),
                checkpoint_path=remote_ckpt,
                device=device.strip() or None,
                backend=backend,
                ollama_model=ollama_model.strip() or default_ollama_model,
                ollama_host=ollama_host,
                ollama_port=ollama_port,
                opendataloader_hybrid=opendataloader_hybrid,
                opendataloader_hybrid_mode=opendataloader_hybrid_mode,
                opendataloader_hybrid_url=opendataloader_hybrid_url,
                opendataloader_hybrid_timeout=opendataloader_hybrid_timeout,
                opendataloader_hybrid_fallback=opendataloader_hybrid_fallback,
            )
            log.info("Remote classification complete!  JSON saved to %s", out_target)
        finally:
            if not no_term:
                trainer.terminate_pod()
            elif trainer.pod_id:
                log.warning(
                    "Pod %s left running — remember to terminate it!", trainer.pod_id
                )
    except Exception as exc:
        log.error("RunPod classification failed: %s", exc, exc_info=VERBOSE)

    _pause()


def run_classify() -> None:
    print()
    log.info("Configure PDF classification")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from model.config import BEST_CHECKPOINT_DIR

    pdf_path = questionary.text("PDF path").ask()
    if pdf_path is None:
        return
    pdf_path = Path(pdf_path.strip().strip("'\"")).expanduser().resolve()
    if not pdf_path.exists():
        log.error("File not found: %s", pdf_path)
        _pause()
        return
    pdf_path = str(pdf_path)

    ckpt = questionary.text("Checkpoint path", default=BEST_CHECKPOINT_DIR).ask()
    if ckpt is None:
        return

    device = questionary.text("Device (blank = auto-detect)", default="").ask()
    if device is None:
        return

    backend = _ask_preprocess_backend()
    if backend is None:
        return

    use_llm = True
    opendataloader_hybrid = False
    opendataloader_hybrid_mode = None
    opendataloader_hybrid_url = None
    opendataloader_hybrid_timeout = None
    opendataloader_hybrid_fallback = False

    if backend == "marker":
        use_llm = questionary.confirm(
            "Enable LLM-enhanced processing? (uses Ollama locally)",
            default=True,
        ).ask()
        if use_llm is None:
            return
    elif backend == "opendataloader":
        hybrid_opts = _ask_opendataloader_hybrid_options()
        if hybrid_opts is None:
            return
        opendataloader_hybrid = hybrid_opts["opendataloader_hybrid"]
        opendataloader_hybrid_mode = hybrid_opts["opendataloader_hybrid_mode"]
        opendataloader_hybrid_url = hybrid_opts["opendataloader_hybrid_url"]
        opendataloader_hybrid_timeout = hybrid_opts["opendataloader_hybrid_timeout"]
        opendataloader_hybrid_fallback = hybrid_opts["opendataloader_hybrid_fallback"]

    out_file = questionary.text(
        "Save JSON output to file (blank = print to terminal)", default=""
    ).ask()
    if out_file is None:
        return

    print()
    log.info("Loading model from %s…", ckpt)
    try:
        from pipeline.classifier import TranscriptClassifier

        clf    = TranscriptClassifier(
            ckpt,
            device=device.strip() or None,
            backend=backend,
            use_llm=use_llm,
            opendataloader_hybrid=opendataloader_hybrid,
            opendataloader_hybrid_mode=opendataloader_hybrid_mode,
            opendataloader_hybrid_url=opendataloader_hybrid_url,
            opendataloader_hybrid_timeout=opendataloader_hybrid_timeout,
            opendataloader_hybrid_fallback=opendataloader_hybrid_fallback,
        )
        log.info("Converting PDF → Markdown and running NER…")
        result = clf.classify_pdf(pdf_path)

        output_json = json.dumps(result, indent=2)

        if out_file.strip():
            Path(out_file.strip()).write_text(output_json)
            log.info("Result saved to %s", out_file.strip())
        else:
            print("\n" + "=" * 60)
            print("Classification Result:")
            print("=" * 60)
            print(output_json)

        log.info(
            "Found %d semester(s), %d total course(s)",
            len(result),
            sum(len(s.get("courses", [])) for s in result),
        )
    except Exception as exc:
        log.error("Classification failed: %s", exc, exc_info=VERBOSE)

    _pause()


# ── Main menu loop ─────────────────────────────────────────────────────────

def _choices() -> list:
    verb_label = f"Toggle verbose logging  [{'ON ' if VERBOSE else 'OFF'}]"
    return [
        Choice("Generate Transcripts",       value="generate"),
        Choice("Preprocess PDFs → Markdown", value="preprocess"),
        Choice("Preprocess Single PDF  (RunPod)", value="preprocess_remote_single"),
        Choice("Train Model  (Local)",        value="train"),
        Choice("Train Model  (RunPod)",       value="train_remote"),
        Choice("Evaluate Model",         value="evaluate"),
        Choice("Evaluate Model  (RunPod)", value="evaluate_remote"),
        Choice("Classify a PDF",         value="classify"),
        Choice("Classify a PDF  (RunPod)", value="classify_remote"),
        Separator(),
        Choice(verb_label,              value="verbose"),
        Choice("Exit",                  value="exit"),
    ]


def main() -> None:
    global VERBOSE
    _apply_logging()
    print(_BANNER)

    _DISPATCH = {
        "generate":     run_generate,
        "preprocess":   run_preprocess,
        "preprocess_remote_single": run_preprocess_remote_single,
        "train":        run_train,
        "train_remote": run_train_remote,
        "evaluate":     run_evaluate,
        "evaluate_remote": run_evaluate_remote,
        "classify":     run_classify,
        "classify_remote": run_classify_remote,
    }

    while True:
        print()
        action = questionary.select(
            "What would you like to do?",
            choices=_choices(),
        ).ask()

        if action is None or action == "exit":
            log.info("Bye!")
            break
        elif action == "verbose":
            VERBOSE = not VERBOSE
            _apply_logging()
            log.info("Verbose logging %s", "enabled" if VERBOSE else "disabled")
        elif action in _DISPATCH:
            _DISPATCH[action]()


if __name__ == "__main__":
    main()
