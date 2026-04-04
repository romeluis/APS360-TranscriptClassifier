"""Generation configuration for TranscriptGenerator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig:
    """All parameters controlling transcript generation.

    Attributes:
        count_per_template: Number of transcripts to generate per template.
        render_pdf: Whether to write PDF files to disk.
        pdf_extract: If True, extract BIO labels from rendered PDF text
                     (realistic noisy input) instead of clean HTML text.
        augment_noise: If True, apply token-level noise augmentation.
        noise_intensity: Scales augmentation probabilities (0=off, 1=default).
        workers: Number of parallel worker processes. 0 = os.cpu_count().
        split_courses: If True, assign disjoint course-name pools to
                       train/val vs. test templates.
        seed: RNG seed for course pool splitting and template assignment.
        train_ratio: Fraction of templates for train split.
        val_ratio: Fraction of templates for val split.
        template_names: Optional list of specific template stems to use.
                        If None, all templates in the templates/ directory are used.
        preprocess_md: If True, run the selected backend on each rendered PDF to produce
                       a companion ``.md`` file.
        preprocess_backend: PDF-to-Markdown backend package
                    (``docling`` default, or ``marker``/``opendataloader``).
    """

    count_per_template: int = 50
    render_pdf: bool = True
    pdf_extract: bool = False
    augment_noise: bool = False
    noise_intensity: float = 1.0
    workers: int = 1
    split_courses: bool = False
    seed: int = 42
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    template_names: Optional[list[str]] = None
    preprocess_md: bool = False
    html_to_md: bool = False
    preprocess_backend: str = "docling"
    preprocess_use_llm: bool = True
    preprocess_batch_multiplier: int = 2
    preprocess_disable_table_recognition: bool = False
    preprocess_opendataloader_hybrid: bool = False
    preprocess_opendataloader_hybrid_mode: str | None = None
    preprocess_opendataloader_hybrid_url: str | None = None
    preprocess_opendataloader_hybrid_timeout: str | None = None
    preprocess_opendataloader_hybrid_fallback: bool = False
