"""Backend implementations for PDF-to-Markdown preprocessing."""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_backend_name(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "marker": "marker",
        "opendataloader": "opendataloader",
        "open-data-loader": "opendataloader",
        "opendataloader-pdf": "opendataloader",
        "docling": "docling",
        "docling-pdf": "docling",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported preprocessor backend '{name}'. "
            "Supported backends: marker, opendataloader, docling"
        )
    return aliases[normalized]


class BasePreprocessorBackend(ABC):
    """Contract for PDF-to-Markdown backend implementations."""

    @abstractmethod
    def convert(
        self,
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> str:
        raise NotImplementedError

    def convert_bulk(
        self,
        input_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        pattern: str = "**/*.pdf",
        skip_existing: bool = False,
    ) -> list[dict]:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

        pdf_paths = sorted(input_dir.glob(pattern))
        if not pdf_paths:
            return []

        results: list[dict] = []
        for pdf_path in pdf_paths:
            if output_dir is not None:
                relative = pdf_path.relative_to(input_dir)
                md_path = Path(output_dir) / relative.with_suffix(".md")
            else:
                md_path = pdf_path.with_suffix(".md")

            if skip_existing and md_path.exists():
                results.append({"pdf": pdf_path, "markdown": md_path, "skipped": True})
                continue

            try:
                self.convert(pdf_path, md_path)
                results.append({"pdf": pdf_path, "markdown": md_path})
            except Exception as exc:
                results.append({"pdf": pdf_path, "markdown": None, "error": str(exc)})

        return results


class MarkerPreprocessorBackend(BasePreprocessorBackend):
    """PDF-to-Markdown backend using marker."""

    def __init__(
        self,
        batch_multiplier: int = 2,
        disable_table_recognition: bool = False,
        use_llm: bool = True,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
    ) -> None:
        self._converter = None
        self._batch_multiplier = batch_multiplier
        self._disable_table_recognition = disable_table_recognition
        self._use_llm = use_llm
        self._ollama_base_url = ollama_base_url
        self._ollama_model = ollama_model

    def _get_converter(self):
        if self._converter is None:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.util import classes_to_strings

            config = {"batch_multiplier": self._batch_multiplier}
            processor_list = None
            llm_service = None

            if self._use_llm:
                config["use_llm"] = True
                ollama_base_url = self._ollama_base_url or os.getenv("OLLAMA_BASE_URL")
                ollama_model = self._ollama_model or os.getenv("OLLAMA_MODEL")
                llm_timeout = _env_int("MARKER_LLM_TIMEOUT", 900)
                llm_max_retries = _env_int("MARKER_LLM_MAX_RETRIES", 3)
                llm_max_concurrency = _env_int("MARKER_LLM_MAX_CONCURRENCY", 1)
                llm_image_min_side = _env_int("MARKER_LLM_IMAGE_MIN_SIDE", 56)
                llm_image_max_side = _env_int("MARKER_LLM_IMAGE_MAX_SIDE", 1536)
                llm_error_body_chars = _env_int("MARKER_LLM_ERROR_BODY_CHARS", 600)
                llm_fail_fast = _env_bool("MARKER_LLM_FAIL_FAST_KNOWN_ERRORS", True)
                llm_text_only_fallback = _env_bool(
                    "MARKER_LLM_TEXT_ONLY_FALLBACK_ON_INVALID_IMAGES", True
                )
                if ollama_base_url:
                    config["ollama_base_url"] = ollama_base_url
                if ollama_model:
                    config["ollama_model"] = ollama_model
                config["timeout"] = llm_timeout
                config["max_retries"] = llm_max_retries
                config["max_concurrency"] = llm_max_concurrency
                config["min_image_side"] = llm_image_min_side
                config["max_image_side"] = llm_image_max_side
                config["max_error_body_chars"] = llm_error_body_chars
                config["fail_fast_on_known_errors"] = llm_fail_fast
                config["text_only_fallback_on_invalid_images"] = llm_text_only_fallback
                llm_service = "preprocessor.ollama_service.SimplifyOllamaService"

            if self._disable_table_recognition:
                config["enable_table_ocr"] = True
                disabled_processors = {
                    "marker.processors.table.TableProcessor",
                    "marker.processors.llm.llm_table.LLMTableProcessor",
                    "marker.processors.llm.llm_table_merge.LLMTableMergeProcessor",
                }
                processor_list = [
                    p
                    for p in classes_to_strings(list(PdfConverter.default_processors))
                    if p not in disabled_processors
                ]

            try:
                self._converter = PdfConverter(
                    artifact_dict=create_model_dict(),
                    processor_list=processor_list,
                    config=config,
                    llm_service=llm_service,
                )
            except TypeError:
                try:
                    self._converter = PdfConverter(
                        artifact_dict=create_model_dict(),
                        processor_list=processor_list,
                        llm_service=llm_service,
                    )
                except TypeError:
                    try:
                        self._converter = PdfConverter(
                            artifact_dict=create_model_dict(),
                            llm_service=llm_service,
                        )
                    except TypeError:
                        self._converter = PdfConverter(
                            artifact_dict=create_model_dict(),
                        )

        return self._converter

    def convert(
        self,
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> str:
        from marker.output import text_from_rendered

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        converter = self._get_converter()
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")

        return text


class OpenDataLoaderPreprocessorBackend(BasePreprocessorBackend):
    """PDF-to-Markdown backend using opendataloader-pdf."""

    def __init__(
        self,
        hybrid: bool = False,
        hybrid_mode: str | None = None,
        hybrid_url: str | None = None,
        hybrid_timeout: str | None = None,
        hybrid_fallback: bool = False,
    ) -> None:
        self._hybrid = hybrid
        self._hybrid_mode = hybrid_mode
        self._hybrid_url = hybrid_url
        self._hybrid_timeout = hybrid_timeout
        self._hybrid_fallback = hybrid_fallback

    def _hybrid_kwargs(self) -> dict:
        if not self._hybrid:
            return {}

        kwargs = {"hybrid": "docling-fast"}
        if self._hybrid_mode:
            kwargs["hybrid_mode"] = self._hybrid_mode
        if self._hybrid_url:
            kwargs["hybrid_url"] = self._hybrid_url
        if self._hybrid_timeout:
            kwargs["hybrid_timeout"] = self._hybrid_timeout
        if self._hybrid_fallback:
            kwargs["hybrid_fallback"] = True
        return kwargs

    def convert(
        self,
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> str:
        try:
            import opendataloader_pdf
        except ImportError as exc:
            raise ImportError(
                "OpenDataLoader backend requires opendataloader-pdf. "
                "Install it with: pip install 'opendataloader-pdf[hybrid]'"
            ) from exc

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if shutil.which("java") is None:
            raise RuntimeError(
                "OpenDataLoader requires Java 11+ on PATH. "
                "Install a JRE/JDK and ensure 'java -version' works."
            )

        target_path = Path(output_path) if output_path is not None else None
        output_dir = (target_path.parent if target_path is not None else pdf_path.parent)
        output_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {
            "input_path": str(pdf_path),
            "output_dir": str(output_dir),
            "format": "markdown",
            "quiet": True,
        }
        kwargs.update(self._hybrid_kwargs())

        try:
            opendataloader_pdf.convert(**kwargs)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "OpenDataLoader failed to launch. Ensure Java 11+ is installed and accessible."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = str(exc)
            raise RuntimeError(f"OpenDataLoader conversion failed: {detail}") from exc

        generated_md = output_dir / f"{pdf_path.stem}.md"
        if not generated_md.exists():
            raise RuntimeError(
                f"OpenDataLoader did not produce expected markdown file: {generated_md}"
            )

        text = generated_md.read_text(encoding="utf-8")

        if target_path is not None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if generated_md.resolve() != target_path.resolve():
                target_path.write_text(text, encoding="utf-8")
                try:
                    generated_md.unlink()
                except OSError:
                    pass

        return text


class DoclingPreprocessorBackend(BasePreprocessorBackend):
    """PDF-to-Markdown backend using Docling."""

    def __init__(self) -> None:
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ModuleNotFoundError as exc:
                if exc.name == "docling":
                    raise ImportError(
                        "Docling backend requires docling. "
                        "Install it with: pip install 'docling[easyocr,rapidocr]'"
                    ) from exc
                raise
            except ImportError as exc:
                detail = str(exc)
                if (
                    "AutoModelForImageTextToText" in detail
                    and "transformers" in detail
                ):
                    raise ImportError(
                        "Docling import failed because transformers is too old. "
                        "Upgrade with: uv pip install --python .venv/bin/python "
                        "'transformers>=4.56,<5'"
                    ) from exc
                raise ImportError(
                    f"Docling import failed: {detail}"
                ) from exc

            self._converter = DocumentConverter()
        return self._converter

    def convert(
        self,
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        converter = self._get_converter()

        result = converter.convert(str(pdf_path))
        text = result.document.export_to_markdown()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")

        return text


def create_preprocessor_backend(
    backend: str,
    batch_multiplier: int = 2,
    disable_table_recognition: bool = False,
    use_llm: bool = True,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
    opendataloader_hybrid: bool = False,
    opendataloader_hybrid_mode: str | None = None,
    opendataloader_hybrid_url: str | None = None,
    opendataloader_hybrid_timeout: str | None = None,
    opendataloader_hybrid_fallback: bool = False,
) -> BasePreprocessorBackend:
    resolved = normalize_backend_name(backend)
    if resolved == "marker":
        return MarkerPreprocessorBackend(
            batch_multiplier=batch_multiplier,
            disable_table_recognition=disable_table_recognition,
            use_llm=use_llm,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )

    if resolved == "opendataloader":
        return OpenDataLoaderPreprocessorBackend(
            hybrid=opendataloader_hybrid,
            hybrid_mode=opendataloader_hybrid_mode,
            hybrid_url=opendataloader_hybrid_url,
            hybrid_timeout=opendataloader_hybrid_timeout,
            hybrid_fallback=opendataloader_hybrid_fallback,
        )

    if resolved == "docling":
        return DoclingPreprocessorBackend()

    raise ValueError(f"Unsupported preprocessor backend: {backend}")
