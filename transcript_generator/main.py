"""CLI entry point — generate synthetic transcript PDFs with BIO labels."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from template_parser import parse_template
from generators import generate_transcript_data
from assembler import assemble
from label_extractor import extract_labels, validate_bio_labels
from pdf_renderer import render_pdf


def generate_for_template(template_path, count, output_dir):
    """
    Generate `count` transcripts from a single template.

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

    for i in range(1, count + 1):
        transcript_id = f"{template_name}_{i:03d}"

        # Generate random data
        data = generate_transcript_data(config)

        # Assemble HTML
        html_string = assemble(raw_html, config, blocks, data)

        # Render PDF
        pdf_path = template_output_dir / f"transcript_{i:03d}.pdf"
        render_pdf(html_string, str(pdf_path))

        # Extract BIO labels
        tokens, labels = extract_labels(html_string)

        # Validate labels
        errors = validate_bio_labels(tokens, labels)
        if errors:
            total_errors += len(errors)
            print(f"  WARNING: {transcript_id} has {len(errors)} label errors:")
            for err in errors[:3]:
                print(f"    - {err}")

        # Write JSON
        json_data = {
            "tokens": tokens,
            "ner_tags": labels,
            "metadata": {
                "template_id": template_name,
                "transcript_id": transcript_id,
                "num_semesters": len(data["semesters"]),
                "num_courses": sum(len(s["courses"]) for s in data["semesters"]),
            },
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

        # Progress
        if i % 10 == 0 or i == count:
            print(f"  [{template_name}] {i}/{count} transcripts generated")

    if total_errors > 0:
        print(f"  WARNING: {total_errors} total BIO label errors in {template_name}")

    return files, label_counts


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
    args = parser.parse_args()

    if not args.template and not args.all_templates:
        parser.error("Must specify --template or --all-templates")

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

    # Generate transcripts
    all_files = []
    total_label_counts = Counter()
    start_time = time.time()

    for template_path in template_paths:
        print(f"\nProcessing template: {template_path.name}")
        files, label_counts = generate_for_template(
            str(template_path), args.count, str(output_dir)
        )
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
