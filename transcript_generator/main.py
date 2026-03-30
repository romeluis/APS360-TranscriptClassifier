"""CLI entry point — generate synthetic transcript PDFs with BIO labels."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from template_parser import parse_template
from generators import COURSE_NAMES, generate_transcript_data
from assembler import assemble
from label_extractor import extract_labels, validate_bio_labels

try:
    from pdf_renderer import render_pdf
    _PDF_AVAILABLE = True
except Exception:
    _PDF_AVAILABLE = False

try:
    from pdf_aligner import extract_pdf_labels
    _PDF_ALIGNER_AVAILABLE = True
except Exception:
    _PDF_ALIGNER_AVAILABLE = False

try:
    from noise_augmenter import augment_tokens
    _NOISE_AUGMENTER_AVAILABLE = True
except Exception:
    _NOISE_AUGMENTER_AVAILABLE = False

# Match model/config.py constants so pool assignment is consistent with data_split.py
_TRAIN_RATIO = 0.60
_VAL_RATIO = 0.20
_SEED = 42


def split_course_pool(seed=_SEED, test_ratio=0.30):
    """Split COURSE_NAMES into train/val and test sub-pools.

    Each department's names are split independently, then a global dedup pass
    removes any names that ended up in both pools (can happen when the same
    name appears under multiple departments).

    Args:
        seed: RNG seed for reproducibility
        test_ratio: fraction of names per department reserved for test

    Returns:
        (train_pool, test_pool) — each is a dict mapping dept -> list[str],
        with strictly disjoint name sets across both pools.
    """
    rng = random.Random(seed)
    train_pool, test_pool = {}, {}
    for dept, names in COURSE_NAMES.items():
        shuffled = list(names)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio))
        test_pool[dept] = shuffled[:n_test]
        train_pool[dept] = shuffled[n_test:]

    # Remove cross-department overlap: any name in test_pool wins over train_pool
    test_names = {n for v in test_pool.values() for n in v}
    train_pool = {
        dept: [n for n in names if n not in test_names]
        for dept, names in train_pool.items()
    }
    return train_pool, test_pool


def get_test_template_names(templates_dir, train_ratio=_TRAIN_RATIO, val_ratio=_VAL_RATIO, seed=_SEED):
    """Pre-compute which template stems are assigned to the test split.

    Mirrors the shuffle + slice logic in model/data_split.py so the course
    pool assignment at generation time matches the downstream split.

    Returns:
        set of template stem strings that belong to the test split
    """
    stems = sorted(p.stem for p in Path(templates_dir).glob("*.html"))
    rng = random.Random(seed)
    rng.shuffle(stems)
    n = len(stems)
    n_train_val = round(n * train_ratio) + round(n * val_ratio)
    return set(stems[n_train_val:])


def generate_for_template(
    template_path,
    count,
    output_dir,
    course_pool=None,
    render_pdf_files=True,
    pdf_extract=False,
    augment_noise=False,
    noise_intensity=1.0,
):
    """
    Generate `count` transcripts from a single template.

    Args:
        template_path: path to the HTML template file
        count: number of transcripts to generate
        output_dir: root output directory
        course_pool: optional course name pool (for split-courses mode)
        render_pdf_files: whether to keep rendered PDFs on disk
        pdf_extract: if True, extract BIO labels from rendered PDF text
                     instead of clean HTML text (requires WeasyPrint + pypdf)
        augment_noise: if True, apply token-level noise augmentation
        noise_intensity: scales augmentation probabilities (0=off, 1=default)

    Returns:
        list of {"pdf": relative_path, "json": relative_path} dicts
        aggregated label counts
    """
    template_name = Path(template_path).stem
    template_output_dir = Path(output_dir) / template_name
    template_output_dir.mkdir(parents=True, exist_ok=True)

    raw_html, config, blocks = parse_template(template_path)

    files = []
    label_counts = Counter()
    total_errors = 0
    alignment_stats = []

    for i in range(1, count + 1):
        transcript_id = f"{template_name}_{i:03d}"

        # Generate random data
        data = generate_transcript_data(config, course_pool=course_pool)

        # Assemble HTML
        html_string = assemble(raw_html, config, blocks, data)

        # Render PDF — required for pdf_extract, optional otherwise
        pdf_path = template_output_dir / f"transcript_{i:03d}.pdf"
        need_pdf = pdf_extract or (render_pdf_files and _PDF_AVAILABLE)
        if need_pdf and _PDF_AVAILABLE:
            render_pdf(html_string, str(pdf_path))

        # Extract BIO labels
        extraction_mode = "html"
        alignment_coverage = None

        if pdf_extract and _PDF_ALIGNER_AVAILABLE and pdf_path.exists():
            tokens, labels, stats = extract_pdf_labels(html_string, str(pdf_path))
            extraction_mode = stats["extraction_mode"]
            alignment_coverage = stats["alignment_coverage"]
            alignment_stats.append(stats)
        else:
            tokens, labels = extract_labels(html_string)

        # Optional noise augmentation
        if augment_noise and _NOISE_AUGMENTER_AVAILABLE:
            rng = random.Random(hash((template_name, i)))
            tokens, labels = augment_tokens(tokens, labels, rng, intensity=noise_intensity)

        # Validate labels
        errors = validate_bio_labels(tokens, labels)
        if errors:
            total_errors += len(errors)

        # Clean up intermediate PDF if not explicitly requested
        if pdf_extract and not render_pdf_files and pdf_path.exists():
            pdf_path.unlink()

        # Write JSON
        metadata = {
            "template_id": template_name,
            "transcript_id": transcript_id,
            "num_semesters": len(data["semesters"]),
            "num_courses": sum(len(s["courses"]) for s in data["semesters"]),
        }
        if pdf_extract:
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
        files.append({
            "pdf": str(pdf_path.relative_to(output_dir)),
            "json": str(json_path.relative_to(output_dir)),
        })

    # Template-level summary (safe for parallel workers — one print per template)
    status_parts = [f"  [{template_name}] {count} transcripts"]
    if pdf_extract and alignment_stats:
        coverages = [s["alignment_coverage"] for s in alignment_stats]
        mean_cov = sum(coverages) / len(coverages)
        fallbacks = sum(1 for s in alignment_stats if s["extraction_mode"] == "html_fallback")
        status_parts.append(f"alignment={mean_cov:.1%}")
        if fallbacks:
            status_parts.append(f"fallbacks={fallbacks}")
    if total_errors > 0:
        status_parts.append(f"BIO_errors={total_errors}")
    print(" | ".join(status_parts))

    return files, label_counts


# ---------------------------------------------------------------------------
# Parallel wrapper
# ---------------------------------------------------------------------------

def _worker_generate(kwargs):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    return generate_for_template(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic transcript PDFs with BIO labels for NER training."
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to a single template HTML file.",
    )
    parser.add_argument(
        "--all-templates",
        action="store_true",
        help="Process all .html templates in the templates/ directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of transcripts to generate per template (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output/).",
    )
    parser.add_argument(
        "--split-courses",
        action="store_true",
        help=(
            "Assign disjoint course name pools to train/val vs. test templates. "
            "Test templates receive course names never seen in training, "
            "preventing the model from memorizing specific names."
        ),
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF rendering (WeasyPrint). JSON files are sufficient for ML training.",
    )
    parser.add_argument(
        "--pdf-extract",
        action="store_true",
        help=(
            "Extract BIO labels from rendered PDF text (via pypdf) instead of "
            "clean HTML text. Trains the model on realistic noisy input. "
            "Requires WeasyPrint and pypdf."
        ),
    )
    parser.add_argument(
        "--augment-noise",
        action="store_true",
        help="Apply token-level noise augmentation to training data.",
    )
    parser.add_argument(
        "--noise-intensity",
        type=float,
        default=1.0,
        help="Scale noise augmentation probabilities (default: 1.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for template generation. "
            "Default: 1 (sequential). Use 0 for os.cpu_count()."
        ),
    )
    args = parser.parse_args()

    if not args.template and not args.all_templates:
        parser.error("Must specify --template or --all-templates")

    if args.pdf_extract and not _PDF_AVAILABLE:
        parser.error("--pdf-extract requires WeasyPrint. Install with: pip install weasyprint")

    if args.pdf_extract and not _PDF_ALIGNER_AVAILABLE:
        parser.error("--pdf-extract requires pdf_aligner module (and pypdf).")

    if args.augment_noise and not _NOISE_AUGMENTER_AVAILABLE:
        parser.error("--augment-noise requires noise_augmenter module.")

    # Resolve paths relative to this script's directory
    script_dir = Path(__file__).parent
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect templates
    if args.all_templates:
        templates_dir = script_dir / "templates"
        template_paths = sorted(templates_dir.glob("*.html"))
        if not template_paths:
            print(f"No .html templates found in {templates_dir}")
            sys.exit(1)
        print(f"Found {len(template_paths)} templates")
    else:
        tp = Path(args.template)
        if not tp.is_absolute():
            tp = script_dir / tp
        template_paths = [tp]

    # Optionally assign disjoint course name pools to train/val vs. test templates
    train_pool = test_pool = None
    test_template_names = set()
    if args.split_courses:
        templates_dir = script_dir / "templates"
        test_template_names = get_test_template_names(templates_dir)
        train_pool, test_pool = split_course_pool()
        train_name_count = sum(len(v) for v in train_pool.values())
        test_name_count = sum(len(v) for v in test_pool.values())
        print(
            f"\nCourse pool split enabled:\n"
            f"  Train/val pool: {train_name_count} names across {len(train_pool)} depts\n"
            f"  Test pool:      {test_name_count} names across {len(test_pool)} depts\n"
            f"  Test templates: {sorted(test_template_names)}\n"
        )

    # Build per-template kwargs
    task_kwargs_list = []
    for template_path in template_paths:
        course_pool = None
        if args.split_courses:
            is_test = template_path.stem in test_template_names
            course_pool = test_pool if is_test else train_pool

        task_kwargs_list.append({
            "template_path": str(template_path),
            "count": args.count,
            "output_dir": str(output_dir),
            "course_pool": course_pool,
            "render_pdf_files": not args.no_pdf or args.pdf_extract,
            "pdf_extract": args.pdf_extract,
            "augment_noise": args.augment_noise,
            "noise_intensity": args.noise_intensity,
        })

    # Generate transcripts (parallel or sequential)
    num_workers = args.workers
    if num_workers == 0:
        num_workers = os.cpu_count() or 4
    num_workers = min(num_workers, len(task_kwargs_list))

    all_files = []
    total_label_counts = Counter()
    start_time = time.time()

    if num_workers > 1:
        print(f"Running with {num_workers} parallel workers...\n")
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_worker_generate, kw): kw["template_path"]
                for kw in task_kwargs_list
            }
            for future in as_completed(futures):
                template_path = futures[future]
                try:
                    files, label_counts = future.result()
                    all_files.extend(files)
                    total_label_counts.update(label_counts)
                except Exception as exc:
                    print(f"  ERROR: {Path(template_path).name} failed: {exc}")
    else:
        for kw in task_kwargs_list:
            files, label_counts = generate_for_template(**kw)
            all_files.extend(files)
            total_label_counts.update(label_counts)

    elapsed = time.time() - start_time

    # Write manifest
    manifest = {
        "total_transcripts": len(all_files),
        "templates_used": [p.stem for p in template_paths],
        "label_distribution": dict(total_label_counts.most_common()),
        "files": all_files,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Generated {len(all_files)} transcripts in {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nLabel distribution:")
    for label, count in total_label_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
