"""TranscriptPreprocessor — convert transcript PDFs to Markdown via pluggable backends."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .backends import create_preprocessor_backend, normalize_backend_name


class TranscriptPreprocessor:
    """Convert transcript PDFs to clean Markdown using pluggable backends.

    Backends are selected by name via ``backend`` and instantiated once for this
    preprocessor instance. Docling is the default backend.

    Usage::

        pp = TranscriptPreprocessor()

        # Single file
        markdown = pp.convert("transcript.pdf", "transcript.md")

        # Bulk
        results = pp.convert_bulk("output_dir/")
    """

    def __init__(
        self,
        backend: str = "docling",
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
    ) -> None:
        """
        Args:
            backend: Backend package for PDF-to-Markdown conversion.
                Supported: ``docling`` (default), ``marker``,
                ``opendataloader``.
            batch_multiplier: How many pages to process in parallel inside
                marker's model pipeline.  Higher values are faster but use
                more RAM/VRAM.  2 is the marker default; 4–8 works well on
                Apple Silicon with ≥16 GB RAM.
            disable_table_recognition: If ``True``, removes marker's table
                recognition processors.  This can improve results on
                transcript-like tabular data where reconstructed Markdown
                tables are noisy.
            use_llm: If ``True``, enables marker's LLM processors and uses
                Ollama for enhanced table/region parsing.
            ollama_base_url: Optional Ollama endpoint URL. If omitted,
                falls back to ``OLLAMA_BASE_URL``.
            ollama_model: Optional Ollama model name. If omitted,
                falls back to ``OLLAMA_MODEL``.
            opendataloader_hybrid: If ``True``, enable OpenDataLoader hybrid
                mode using backend ``docling-fast``.
            opendataloader_hybrid_mode: OpenDataLoader hybrid mode
                (typically ``auto`` or ``full``).
            opendataloader_hybrid_url: OpenDataLoader hybrid server URL.
            opendataloader_hybrid_timeout: OpenDataLoader hybrid timeout in
                milliseconds as a string.
            opendataloader_hybrid_fallback: Enable OpenDataLoader Java
                fallback when hybrid backend fails.
        """
        self._backend_name = normalize_backend_name(backend)
        self._backend = create_preprocessor_backend(
            backend=self._backend_name,
            batch_multiplier=batch_multiplier,
            disable_table_recognition=disable_table_recognition,
            use_llm=use_llm,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            opendataloader_hybrid=opendataloader_hybrid,
            opendataloader_hybrid_mode=opendataloader_hybrid_mode,
            opendataloader_hybrid_url=opendataloader_hybrid_url,
            opendataloader_hybrid_timeout=opendataloader_hybrid_timeout,
            opendataloader_hybrid_fallback=opendataloader_hybrid_fallback,
        )

    @property
    def backend(self) -> str:
        """Return normalized backend name for this preprocessor instance."""
        return self._backend_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """Convert a single PDF to Markdown.

        Args:
            pdf_path: Path to the input PDF file.
            output_path: If provided, write the Markdown text to this file.

        Returns:
            The Markdown text.
        """
        return self._backend.convert(pdf_path=pdf_path, output_path=output_path)

    def convert_bulk(
        self,
        input_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        pattern: str = "**/*.pdf",
        skip_existing: bool = False,
    ) -> list[dict]:
        """Convert all matching PDFs under *input_dir* to Markdown.

        Args:
            input_dir: Directory to search for PDF files.
            output_dir: Where to write ``.md`` files. If ``None``, each
                        ``.md`` is written alongside its source PDF.
            pattern: Glob pattern for finding PDFs (default ``**/*.pdf``).
            skip_existing: If ``True``, skip any PDF whose ``.md`` output
                           already exists (useful for resuming interrupted runs).

        Returns:
            List of result dicts, one per PDF::

                {"pdf": Path, "markdown": Path}                        # converted
                {"pdf": Path, "markdown": Path, "skipped": True}       # already existed
                {"pdf": Path, "markdown": None, "error": str}          # failed
        """
        return self._backend.convert_bulk(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            skip_existing=skip_existing,
        )
