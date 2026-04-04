"""TranscriptGenerator — public API for programmatic transcript generation."""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .config import GenerationConfig
from .template_parser import parse_template
from .generators import COURSE_NAMES, generate_transcript_data
from .assembler import assemble
from .label_extractor import extract_labels, validate_bio_labels


class TranscriptGenerator:
    """Generate synthetic transcript PDFs with BIO-labeled JSON files.

    Usage::

        gen = TranscriptGenerator(output_dir="./output")
        manifest = gen.generate(GenerationConfig(count_per_template=50))

    The generator discovers templates from its package's ``templates/``
    directory, renders PDFs via WeasyPrint, extracts BIO labels, and writes
    per-transcript JSON files plus a top-level ``manifest.json``.
    """

    def __init__(self, output_dir: str | Path):
        self._output_dir = Path(output_dir)
        self._pkg_dir = Path(__file__).parent
        self._templates_dir = self._pkg_dir / "templates"

        # Lazy-loaded optional dependency callables
        self._pdf_render_fn = None
        self._pdf_align_fn = None
        self._noise_aug_fn = None
        self._preprocessor = None
        self._preprocessor_key = None

    # ------------------------------------------------------------------
    # Optional dependency loading
    # ------------------------------------------------------------------

    def _get_pdf_renderer(self):
        if self._pdf_render_fn is None:
            try:
                from .pdf_renderer import render_pdf
                self._pdf_render_fn = render_pdf
            except (ImportError, OSError):
                self._pdf_render_fn = False
        return self._pdf_render_fn if self._pdf_render_fn is not False else None

    def _get_pdf_aligner(self):
        if self._pdf_align_fn is None:
            try:
                from .pdf_aligner import extract_pdf_labels
                self._pdf_align_fn = extract_pdf_labels
            except ImportError:
                self._pdf_align_fn = False
        return self._pdf_align_fn if self._pdf_align_fn is not False else None

    def _get_noise_augmenter(self):
        if self._noise_aug_fn is None:
            try:
                from .noise_augmenter import augment_tokens
                self._noise_aug_fn = augment_tokens
            except ImportError:
                self._noise_aug_fn = False
        return self._noise_aug_fn if self._noise_aug_fn is not False else None

    def _get_preprocessor(
        self,
        backend: str = "docling",
        use_llm: bool = True,
        batch_multiplier: int = 2,
        disable_table_recognition: bool = False,
        opendataloader_hybrid: bool = False,
        opendataloader_hybrid_mode: str | None = None,
        opendataloader_hybrid_url: str | None = None,
        opendataloader_hybrid_timeout: str | None = None,
        opendataloader_hybrid_fallback: bool = False,
    ):
        key = (
            backend,
            use_llm,
            batch_multiplier,
            disable_table_recognition,
            opendataloader_hybrid,
            opendataloader_hybrid_mode,
            opendataloader_hybrid_url,
            opendataloader_hybrid_timeout,
            opendataloader_hybrid_fallback,
        )

        if self._preprocessor is None or self._preprocessor_key != key:
            try:
                from preprocessor import TranscriptPreprocessor

                self._preprocessor = TranscriptPreprocessor(
                    backend=backend,
                    use_llm=use_llm,
                    batch_multiplier=batch_multiplier,
                    disable_table_recognition=disable_table_recognition,
                    opendataloader_hybrid=opendataloader_hybrid,
                    opendataloader_hybrid_mode=opendataloader_hybrid_mode,
                    opendataloader_hybrid_url=opendataloader_hybrid_url,
                    opendataloader_hybrid_timeout=opendataloader_hybrid_timeout,
                    opendataloader_hybrid_fallback=opendataloader_hybrid_fallback,
                )
                self._preprocessor_key = key
            except ImportError:
                self._preprocessor = False

        return self._preprocessor if self._preprocessor is not False else None

    # ------------------------------------------------------------------
    # Template discovery
    # ------------------------------------------------------------------

    def list_templates(self) -> list[Path]:
        """Return sorted list of all available template paths."""
        return sorted(self._templates_dir.glob("*.html"))

    # ------------------------------------------------------------------
    # Course pool splitting
    # ------------------------------------------------------------------

    def _split_course_pool(self, seed: int, test_ratio: float = 0.30):
        """Split COURSE_NAMES into disjoint train and test pools.

        Each department's names are split independently, then a global
        dedup pass removes any names that ended up in both pools.
        """
        rng = random.Random(seed)
        train_pool: dict[str, list[str]] = {}
        test_pool: dict[str, list[str]] = {}

        for dept, names in COURSE_NAMES.items():
            shuffled = list(names)
            rng.shuffle(shuffled)
            n_test = max(1, round(len(shuffled) * test_ratio))
            test_pool[dept] = shuffled[:n_test]
            train_pool[dept] = shuffled[n_test:]

        # Any name in test_pool wins over train_pool (cross-dept dedup)
        test_names = {n for v in test_pool.values() for n in v}
        train_pool = {
            dept: [n for n in names if n not in test_names]
            for dept, names in train_pool.items()
        }
        return train_pool, test_pool

    def _get_test_template_names(
        self, train_ratio: float, val_ratio: float, seed: int,
    ) -> set[str]:
        """Determine which template stems belong to the test split."""
        stems = sorted(p.stem for p in self._templates_dir.glob("*.html"))
        rng = random.Random(seed)
        rng.shuffle(stems)
        n = len(stems)
        n_train_val = round(n * train_ratio) + round(n * val_ratio)
        return set(stems[n_train_val:])

    # ------------------------------------------------------------------
    # Single-template generation
    # ------------------------------------------------------------------

    def _generate_for_template(
        self,
        template_path: str | Path,
        count: int,
        course_pool: Optional[dict] = None,
        render_pdf: bool = True,
        pdf_extract: bool = False,
        augment_noise: bool = False,
        noise_intensity: float = 1.0,
        preprocess_md: bool = False,
        html_to_md: bool = False,
        preprocess_backend: str = "docling",
        preprocess_use_llm: bool = True,
        preprocess_batch_multiplier: int = 2,
        preprocess_disable_table_recognition: bool = False,
        preprocess_opendataloader_hybrid: bool = False,
        preprocess_opendataloader_hybrid_mode: str | None = None,
        preprocess_opendataloader_hybrid_url: str | None = None,
        preprocess_opendataloader_hybrid_timeout: str | None = None,
        preprocess_opendataloader_hybrid_fallback: bool = False,
    ) -> tuple[list[dict], Counter]:
        """Generate *count* transcripts from one template.

        Returns ``(file_list, label_counts)`` where each entry in
        *file_list* is ``{"pdf": relative_path, "json": relative_path}``.
        """
        template_path = Path(template_path)
        template_name = template_path.stem
        template_output_dir = self._output_dir / template_name
        template_output_dir.mkdir(parents=True, exist_ok=True)

        raw_html, config, blocks = parse_template(str(template_path))

        render_pdf_fn = self._get_pdf_renderer()
        extract_pdf_labels_fn = self._get_pdf_aligner()
        augment_tokens_fn = self._get_noise_augmenter()
        preprocessor = (
            self._get_preprocessor(
                backend=preprocess_backend,
                use_llm=preprocess_use_llm,
                batch_multiplier=preprocess_batch_multiplier,
                disable_table_recognition=preprocess_disable_table_recognition,
                opendataloader_hybrid=preprocess_opendataloader_hybrid,
                opendataloader_hybrid_mode=preprocess_opendataloader_hybrid_mode,
                opendataloader_hybrid_url=preprocess_opendataloader_hybrid_url,
                opendataloader_hybrid_timeout=preprocess_opendataloader_hybrid_timeout,
                opendataloader_hybrid_fallback=preprocess_opendataloader_hybrid_fallback,
            )
            if preprocess_md
            else None
        )

        # Import html_to_markdown converter if needed
        html_to_md_fn = None
        md_labeler_fn = None
        if html_to_md:
            from preprocessor.html_to_markdown import html_to_markdown as _h2m
            from pipeline.markdown_labeler import label_markdown as _lmd
            html_to_md_fn = _h2m
            md_labeler_fn = _lmd

        files: list[dict] = []
        label_counts: Counter = Counter()
        total_errors = 0
        alignment_stats: list[dict] = []
        md_coverages: list[float] = []

        for i in range(1, count + 1):
            transcript_id = f"{template_name}_{i:03d}"

            # Generate random data
            data = generate_transcript_data(config, course_pool=course_pool)

            # Assemble HTML
            html_string = assemble(raw_html, config, blocks, data)

            # Render PDF (skip if only html_to_md is requested)
            pdf_path = template_output_dir / f"transcript_{i:03d}.pdf"
            need_pdf = pdf_extract or preprocess_md or (render_pdf and render_pdf_fn is not None)
            if need_pdf and render_pdf_fn is not None:
                render_pdf_fn(html_string, str(pdf_path))

            # HTML-to-Markdown fast path: convert HTML directly and label
            # the markdown tokens (bypasses PDF rendering and backend preprocessing)
            if html_to_md_fn is not None and md_labeler_fn is not None:
                md_text = html_to_md_fn(html_string)
                md_path = template_output_dir / f"transcript_{i:03d}.md"
                md_path.write_text(md_text, encoding="utf-8")

                ground_truth = {"semesters": data["semesters"]}
                tokens, labels, coverage = md_labeler_fn(md_text, ground_truth)
                md_coverages.append(coverage)
                extraction_mode = "html_to_md"
                alignment_coverage = coverage
            # Extract BIO labels (existing paths)
            elif (
                pdf_extract
                and extract_pdf_labels_fn is not None
                and pdf_path.exists()
            ):
                tokens, labels, stats = extract_pdf_labels_fn(
                    html_string, str(pdf_path),
                )
                extraction_mode = stats["extraction_mode"]
                alignment_coverage = stats["alignment_coverage"]
                alignment_stats.append(stats)
            else:
                tokens, labels = extract_labels(html_string)
                extraction_mode = "html"
                alignment_coverage = None

            # Optional noise augmentation
            if augment_noise and augment_tokens_fn is not None:
                rng = random.Random(hash((template_name, i)))
                tokens, labels = augment_tokens_fn(
                    tokens, labels, rng, intensity=noise_intensity,
                )

            # Validate labels
            errors = validate_bio_labels(tokens, labels)
            if errors:
                total_errors += len(errors)

            # Clean up intermediate PDF if not explicitly requested
            if pdf_extract and not render_pdf and pdf_path.exists():
                pdf_path.unlink()

            # Write JSON
            metadata: dict = {
                "template_id": template_name,
                "transcript_id": transcript_id,
                "num_semesters": len(data["semesters"]),
                "num_courses": sum(
                    len(s["courses"]) for s in data["semesters"]
                ),
            }
            if pdf_extract or html_to_md:
                metadata["extraction_mode"] = extraction_mode
                if alignment_coverage is not None:
                    metadata["alignment_coverage"] = alignment_coverage

            json_data = {
                "tokens": tokens,
                "ner_tags": labels,
                "metadata": metadata,
            }
            json_path = template_output_dir / f"transcript_{i:03d}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # Track stats
            label_counts.update(labels)
            entry: dict = {
                "json": str(json_path.relative_to(self._output_dir)),
            }
            if need_pdf:
                entry["pdf"] = str(pdf_path.relative_to(self._output_dir))
            if html_to_md:
                md_rel = template_output_dir / f"transcript_{i:03d}.md"
                entry["md"] = str(md_rel.relative_to(self._output_dir))
            files.append(entry)

        # Bulk preprocess PDFs to Markdown via marker (loads model once)
        if preprocessor is not None:
            results = preprocessor.convert_bulk(
                template_output_dir, pattern="*.pdf",
            )
            md_lookup = {
                r["pdf"].stem: r["markdown"]
                for r in results if r.get("markdown")
            }
            for entry in files:
                pdf_stem = Path(entry.get("pdf", "")).stem
                if pdf_stem in md_lookup:
                    entry["md"] = str(
                        md_lookup[pdf_stem].relative_to(self._output_dir)
                    )

        # Per-template summary
        status_parts = [f"  [{template_name}] {count} transcripts"]
        if html_to_md and md_coverages:
            mean_cov = sum(md_coverages) / len(md_coverages)
            status_parts.append(f"md_coverage={mean_cov:.1%}")
        elif pdf_extract and alignment_stats:
            coverages = [s["alignment_coverage"] for s in alignment_stats]
            mean_cov = sum(coverages) / len(coverages)
            fallbacks = sum(
                1 for s in alignment_stats
                if s["extraction_mode"] == "html_fallback"
            )
            status_parts.append(f"alignment={mean_cov:.1%}")
            if fallbacks:
                status_parts.append(f"fallbacks={fallbacks}")
        if total_errors > 0:
            status_parts.append(f"BIO_errors={total_errors}")
        print(" | ".join(status_parts))

        return files, label_counts

    # ------------------------------------------------------------------
    # Parallel dispatch
    # ------------------------------------------------------------------

    def _run_tasks(
        self,
        task_kwargs_list: list[dict],
        workers: int,
    ) -> tuple[list[dict], Counter]:
        num_workers = workers
        if num_workers == 0:
            num_workers = os.cpu_count() or 4
        num_workers = min(num_workers, len(task_kwargs_list))

        all_files: list[dict] = []
        total_label_counts: Counter = Counter()

        if num_workers > 1:
            print(f"Running with {num_workers} parallel workers...\n")
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(
                        _worker_generate,
                        str(self._output_dir),
                        kw,
                    ): kw["template_path"]
                    for kw in task_kwargs_list
                }
                for future in as_completed(futures):
                    template_path = futures[future]
                    try:
                        files, counts = future.result()
                        all_files.extend(files)
                        total_label_counts.update(counts)
                    except Exception as exc:
                        print(
                            f"  ERROR: {Path(template_path).name} failed: {exc}"
                        )
        else:
            for kw in task_kwargs_list:
                files, counts = self._generate_for_template(**kw)
                all_files.extend(files)
                total_label_counts.update(counts)

        return all_files, total_label_counts

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def generate(self, config: Optional[GenerationConfig] = None) -> dict:
        """Generate transcripts for all (or selected) templates.

        Args:
            config: Controls all generation parameters.
                    If ``None``, uses defaults.

        Returns:
            The manifest dict (also written to ``output_dir/manifest.json``).
            Keys: ``total_transcripts``, ``templates_used``,
            ``label_distribution``, ``files``.
        """
        if config is None:
            config = GenerationConfig()

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Validate optional dependencies early
        if config.pdf_extract:
            if not self._get_pdf_renderer():
                raise ImportError(
                    "pdf_extract requires WeasyPrint. "
                    "Install with: pip install weasyprint"
                )
            if not self._get_pdf_aligner():
                raise ImportError(
                    "pdf_extract requires pypdf. "
                    "Install with: pip install pypdf"
                )
        if config.augment_noise and not self._get_noise_augmenter():
            raise ImportError("augment_noise requires noise_augmenter deps.")
        if config.preprocess_md:
            if not self._get_pdf_renderer():
                raise ImportError(
                    "preprocess_md requires WeasyPrint. "
                    "Install with: uv pip install weasyprint"
                )
            from preprocessor import normalize_backend_name

            preprocess_backend = normalize_backend_name(config.preprocess_backend)
            if preprocess_backend == "marker":
                try:
                    import marker  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "preprocess_md with marker backend requires marker-pdf. "
                        "Install with: uv pip install marker-pdf"
                    ) from exc
            elif preprocess_backend == "opendataloader":
                try:
                    import opendataloader_pdf  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "preprocess_md with OpenDataLoader backend requires opendataloader-pdf. "
                        "Install with: uv pip install 'opendataloader-pdf[hybrid]'"
                    ) from exc
            elif preprocess_backend == "docling":
                try:
                    import docling.document_converter  # noqa: F401
                except ImportError as exc:
                    detail = str(exc)
                    if (
                        "AutoModelForImageTextToText" in detail
                        and "transformers" in detail
                    ):
                        raise ImportError(
                            "preprocess_md with Docling backend requires a newer "
                            "transformers version. Install with: "
                            "uv pip install --python .venv/bin/python "
                            "'transformers>=4.56,<5'"
                        ) from exc
                    raise ImportError(
                        "preprocess_md with Docling backend requires docling. "
                        "Install with: uv pip install 'docling[easyocr,rapidocr]'"
                    ) from exc

        # Discover templates
        if config.template_names:
            template_paths = [
                self._templates_dir / f"{name}.html"
                for name in config.template_names
            ]
            for tp in template_paths:
                if not tp.exists():
                    raise FileNotFoundError(f"Template not found: {tp}")
        else:
            template_paths = self.list_templates()

        if not template_paths:
            raise FileNotFoundError(
                f"No .html templates found in {self._templates_dir}"
            )

        print(f"Found {len(template_paths)} template(s)")

        # Course pool splitting
        train_pool = test_pool = None
        test_template_names: set[str] = set()
        if config.split_courses:
            test_template_names = self._get_test_template_names(
                config.train_ratio, config.val_ratio, config.seed,
            )
            train_pool, test_pool = self._split_course_pool(config.seed)
            train_count = sum(len(v) for v in train_pool.values())
            test_count = sum(len(v) for v in test_pool.values())
            print(
                f"\nCourse pool split enabled:\n"
                f"  Train/val pool: {train_count} names\n"
                f"  Test pool:      {test_count} names\n"
                f"  Test templates: {sorted(test_template_names)}\n"
            )

        # Build per-template task kwargs
        task_kwargs_list: list[dict] = []
        for tp in template_paths:
            course_pool = None
            if config.split_courses:
                is_test = tp.stem in test_template_names
                course_pool = test_pool if is_test else train_pool

            task_kwargs_list.append({
                "template_path": str(tp),
                "count": config.count_per_template,
                "course_pool": course_pool,
                "render_pdf": config.render_pdf or config.pdf_extract or config.preprocess_md,
                "pdf_extract": config.pdf_extract,
                "augment_noise": config.augment_noise,
                "noise_intensity": config.noise_intensity,
                "preprocess_md": config.preprocess_md,
                "html_to_md": config.html_to_md,
                "preprocess_backend": config.preprocess_backend,
                "preprocess_use_llm": config.preprocess_use_llm,
                "preprocess_batch_multiplier": config.preprocess_batch_multiplier,
                "preprocess_disable_table_recognition": config.preprocess_disable_table_recognition,
                "preprocess_opendataloader_hybrid": config.preprocess_opendataloader_hybrid,
                "preprocess_opendataloader_hybrid_mode": config.preprocess_opendataloader_hybrid_mode,
                "preprocess_opendataloader_hybrid_url": config.preprocess_opendataloader_hybrid_url,
                "preprocess_opendataloader_hybrid_timeout": config.preprocess_opendataloader_hybrid_timeout,
                "preprocess_opendataloader_hybrid_fallback": config.preprocess_opendataloader_hybrid_fallback,
            })

        # Execute
        start_time = time.time()
        all_files, total_label_counts = self._run_tasks(
            task_kwargs_list, config.workers,
        )
        elapsed = time.time() - start_time

        # Write manifest
        manifest = {
            "total_transcripts": len(all_files),
            "templates_used": [tp.stem for tp in template_paths],
            "label_distribution": dict(total_label_counts.most_common()),
            "files": all_files,
        }
        manifest_path = self._output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Summary
        print(f"\n{'=' * 50}")
        print(f"Generated {len(all_files)} transcripts in {elapsed:.1f}s")
        print(f"Output directory: {self._output_dir}")
        print(f"Manifest: {manifest_path}")
        print(f"\nLabel distribution:")
        for label, count in total_label_counts.most_common():
            print(f"  {label}: {count}")

        return manifest


# ------------------------------------------------------------------
# Module-level function for ProcessPoolExecutor (must be picklable)
# ------------------------------------------------------------------

def _worker_generate(output_dir: str, kwargs: dict):
    """Top-level picklable wrapper — creates a fresh generator per worker."""
    gen = TranscriptGenerator(output_dir)
    return gen._generate_for_template(**kwargs)
