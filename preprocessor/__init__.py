"""preprocessor — PDF-to-Markdown conversion using pluggable backends."""

from .html_to_markdown import html_to_markdown
from .backends import normalize_backend_name
from .preprocessor import TranscriptPreprocessor

SUPPORTED_PREPROCESS_BACKENDS = ("docling", "marker", "opendataloader")

__all__ = [
	"TranscriptPreprocessor",
	"html_to_markdown",
	"normalize_backend_name",
	"SUPPORTED_PREPROCESS_BACKENDS",
]
