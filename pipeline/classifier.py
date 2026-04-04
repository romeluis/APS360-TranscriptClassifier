"""
TranscriptClassifier -- public API for transcript classification.

Usage by an orchestrator:

    from pipeline import TranscriptClassifier

    clf = TranscriptClassifier("model/checkpoints/best")
    semesters = clf.classify_pdf("transcript.pdf")
    # or
    semesters = clf.classify_markdown(md_text)

Returns:
    [{"semester_name": "Fall 2021",
      "courses": [{"code": "CSC301", "name": "Intro to SE", "grade": "A-"}, ...]}]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from model.predictor import NERPredictor
from .reconstructor import reconstruct_semesters


class TranscriptClassifier:
    """Classify a transcript (PDF, markdown, or raw text) into structured semester/course data."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        use_llm: bool = True,
        backend: str = "docling",
        opendataloader_hybrid: bool = False,
        opendataloader_hybrid_mode: str | None = None,
        opendataloader_hybrid_url: str | None = None,
        opendataloader_hybrid_timeout: str | None = None,
        opendataloader_hybrid_fallback: bool = False,
    ):
        """
        Args:
            model_path: Path to a trained model checkpoint directory.
            device: Device string ("cuda", "cpu", "mps"). Auto-detected if None.
            use_llm: If ``True``, use marker's LLM-enhanced PDF preprocessing.
            backend: PDF-to-Markdown backend package (``marker`` or
                ``opendataloader`` or ``docling``).
        """
        self._predictor = NERPredictor(model_path, device=device)
        self._preprocessor = None  # lazy-loaded
        self._use_llm = use_llm
        self._backend = backend
        self._opendataloader_hybrid = opendataloader_hybrid
        self._opendataloader_hybrid_mode = opendataloader_hybrid_mode
        self._opendataloader_hybrid_url = opendataloader_hybrid_url
        self._opendataloader_hybrid_timeout = opendataloader_hybrid_timeout
        self._opendataloader_hybrid_fallback = opendataloader_hybrid_fallback

    def _get_preprocessor(self):
        """Lazy-load the TranscriptPreprocessor (backend can be heavy)."""
        if self._preprocessor is None:
            from preprocessor import TranscriptPreprocessor
            self._preprocessor = TranscriptPreprocessor(
                backend=self._backend,
                use_llm=self._use_llm,
                opendataloader_hybrid=self._opendataloader_hybrid,
                opendataloader_hybrid_mode=self._opendataloader_hybrid_mode,
                opendataloader_hybrid_url=self._opendataloader_hybrid_url,
                opendataloader_hybrid_timeout=self._opendataloader_hybrid_timeout,
                opendataloader_hybrid_fallback=self._opendataloader_hybrid_fallback,
            )
        return self._preprocessor

    def classify_pdf(self, pdf_path: str | Path) -> list[dict]:
        """Classify a PDF transcript.

        Pipeline: PDF -> markdown (via configured backend) -> NER -> structured output.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        preprocessor = self._get_preprocessor()
        md_text = preprocessor.convert(pdf_path)
        return self.classify_markdown(md_text)

    def classify_markdown(self, md_text: str) -> list[dict]:
        """Classify markdown text into structured semester/course data."""
        tokens = md_text.split()
        if not tokens:
            return []
        tags = self._predictor.predict(tokens)
        return reconstruct_semesters(tokens, tags)

    def classify_text(self, text: str) -> list[dict]:
        """Classify raw text (non-markdown) into structured semester/course data."""
        tokens = text.split()
        if not tokens:
            return []
        tags = self._predictor.predict(tokens)
        return reconstruct_semesters(tokens, tags)
