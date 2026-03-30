"""Validate PDF alignment quality across templates before full data regeneration.

For each template, generates a small number of transcripts via both the HTML
and PDF label-extraction paths, then compares token counts, entity counts,
alignment coverage, and BIO errors.

Usage:
    python test_alignment.py --all-templates --count 3
    python test_alignment.py --template templates/31_twocol_flex_teal.html
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from template_parser import parse_template
from generators import generate_transcript_data
from assembler import assemble
from label_extractor import extract_labels, validate_bio_labels
from pdf_renderer import render_pdf
from pdf_aligner import extract_pdf_labels


def _count_entities(labels):
    """Count B- tags per entity type."""
    counts = Counter()
    for lab in labels:
        if lab.startswith("B-"):
            counts[lab[2:]] += 1
    return dict(counts)


def test_template(template_path, count=3, verbose=True):
    """Test alignment for a single template.

    Returns a dict of aggregate stats.
    """
    template_name = Path(template_path).stem
    raw_html, config, blocks = parse_template(str(template_path))

    coverages = []
    fallbacks = 0
    results = []

    for i in range(1, count + 1):
        data = generate_transcript_data(config)
        html_string = assemble(raw_html, config, blocks, data)

        # HTML path
        html_tokens, html_labels = extract_labels(html_string)
        html_entities = _count_entities(html_labels)

        # PDF path: render to a temp file, then extract
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        render_pdf(html_string, tmp_path)
        pdf_tokens, pdf_labels, stats = extract_pdf_labels(html_string, tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        pdf_entities = _count_entities(pdf_labels)
        pdf_bio_errors = validate_bio_labels(pdf_tokens, pdf_labels)

        coverage = stats["alignment_coverage"]
        coverages.append(coverage)
        if stats["extraction_mode"] == "html_fallback":
            fallbacks += 1

        result = {
            "transcript": i,
            "html_tokens": len(html_tokens),
            "pdf_tokens": len(pdf_tokens),
            "coverage": coverage,
            "mode": stats["extraction_mode"],
            "html_entities": html_entities,
            "pdf_entities": pdf_entities,
            "bio_errors": len(pdf_bio_errors),
        }
        results.append(result)

        if verbose:
            print(f"  #{i}: html={len(html_tokens)} tok, pdf={len(pdf_tokens)} tok, "
                  f"coverage={coverage:.1%}, mode={stats['extraction_mode']}, "
                  f"bio_errors={len(pdf_bio_errors)}")

            # Show entity comparison
            all_entity_types = sorted(set(list(html_entities.keys()) + list(pdf_entities.keys())))
            for et in all_entity_types:
                h = html_entities.get(et, 0)
                p = pdf_entities.get(et, 0)
                match = "OK" if h == p else "MISMATCH"
                print(f"    {et}: html={h} pdf={p} {match}")

            # Show first few non-O token/tag pairs side by side
            if i == 1:
                print(f"\n  First transcript token-tag sample (non-O):")
                print(f"    {'HTML':<40} {'PDF'}")
                print(f"    {'-'*40} {'-'*40}")
                html_entities_list = [(t, l) for t, l in zip(html_tokens, html_labels) if l != "O"]
                pdf_entities_list = [(t, l) for t, l in zip(pdf_tokens, pdf_labels) if l != "O"]
                for j in range(max(len(html_entities_list), len(pdf_entities_list))):
                    h = f"{html_entities_list[j][0]} ({html_entities_list[j][1]})" if j < len(html_entities_list) else ""
                    p = f"{pdf_entities_list[j][0]} ({pdf_entities_list[j][1]})" if j < len(pdf_entities_list) else ""
                    print(f"    {h:<40} {p}")
                print()

    mean_coverage = sum(coverages) / len(coverages) if coverages else 0
    return {
        "template": template_name,
        "count": count,
        "mean_coverage": mean_coverage,
        "min_coverage": min(coverages) if coverages else 0,
        "fallbacks": fallbacks,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate PDF alignment quality.")
    parser.add_argument("--template", type=str, help="Path to a single template.")
    parser.add_argument("--all-templates", action="store_true", help="Test all templates.")
    parser.add_argument("--count", type=int, default=3, help="Transcripts per template (default: 3).")
    args = parser.parse_args()

    if not args.template and not args.all_templates:
        parser.error("Must specify --template or --all-templates")

    script_dir = Path(__file__).parent

    if args.all_templates:
        templates_dir = script_dir / "templates"
        template_paths = sorted(templates_dir.glob("*.html"))
    else:
        tp = Path(args.template)
        if not tp.is_absolute():
            tp = script_dir / tp
        template_paths = [tp]

    print(f"Testing alignment on {len(template_paths)} template(s), {args.count} transcript(s) each\n")

    summaries = []
    for tp in template_paths:
        print(f"=== {tp.stem} ===")
        summary = test_template(tp, count=args.count, verbose=True)
        summaries.append(summary)
        print()

    # Summary table
    print("=" * 80)
    print(f"{'Template':<40} {'Mean Cov':>8} {'Min Cov':>8} {'Fallbacks':>9}")
    print("-" * 80)

    problem_templates = []
    for s in sorted(summaries, key=lambda x: x["mean_coverage"]):
        flag = " ***" if s["mean_coverage"] < 0.80 else ""
        print(f"{s['template']:<40} {s['mean_coverage']:>7.1%} {s['min_coverage']:>7.1%} {s['fallbacks']:>9}{flag}")
        if s["mean_coverage"] < 0.80:
            problem_templates.append(s["template"])

    print("-" * 80)
    all_coverages = [s["mean_coverage"] for s in summaries]
    overall_mean = sum(all_coverages) / len(all_coverages) if all_coverages else 0
    total_fallbacks = sum(s["fallbacks"] for s in summaries)
    print(f"{'OVERALL':<40} {overall_mean:>7.1%} {'':>8} {total_fallbacks:>9}")

    if problem_templates:
        print(f"\n*** Templates with <80% coverage ({len(problem_templates)}):")
        for t in problem_templates:
            print(f"  - {t}")
        print("These templates will fall back to HTML extraction if coverage drops below 50%.")


if __name__ == "__main__":
    main()
