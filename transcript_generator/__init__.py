"""transcript_generator — synthetic transcript PDF + BIO label generation."""

from .config import GenerationConfig
from .generator import TranscriptGenerator

__all__ = ["TranscriptGenerator", "GenerationConfig"]
