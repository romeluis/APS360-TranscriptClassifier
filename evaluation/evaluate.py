"""
Unified NER evaluation script.

Runs a model (baseline or future ML) against generated transcript data and
produces textual metrics, worst-case conversion analysis, and charts.

Usage:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --data transcript_generator/output/ --verbose
    python -m evaluation.evaluate --no-charts --worst 10
"""

import argparse
import sys
from pathlib import Path

from seqeval.metrics import classification_report, f1_score
from tabulate import tabulate

from .charts import generate_all_charts
from .data_loader import load_dataset
from .metrics import (
    compute_error_patterns,
    compute_per_template_metrics,
    compute_per_transcript_metrics,
    compute_tag_distribution,
    compute_token_accuracy,
    compute_worst_case_stats,
    find_worst_transcripts,
)
from .model_registry import get_model

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Qualitative display
# ---------------------------------------------------------------------------

def _show_example(tokens, true_tags, pred_tags, title=""):
    """Print aligned token/true/pred columns, highlighting mismatches."""
    if title:
        print(f"\n{BOLD}{title}{RESET}")

    rows = []
    for tok, true, pred in zip(tokens, true_tags, pred_tags):
        match = true == pred
        marker = "" if match else f"{RED}X{RESET}"
        pred_display = pred if match else f"{RED}{pred}{RESET}"
        rows.append([tok, true, pred_display, marker])

    print(tabulate(rows, headers=["Token", "True", "Predicted", ""],
                   tablefmt="simple"))

    total = len(tokens)
    correct = sum(1 for t, p in zip(true_tags, pred_tags) if t == p)
    print(f"  Tokens: {total}  |  Correct: {correct}  |  Errors: {total - correct}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    data_dir: str,
    model_name: str = "baseline",
    output_dir: str = "evaluation/output",
    n_examples: int = 3,
    n_worst: int = 5,
    verbose: bool = False,
    no_charts: bool = False,
):
    """Run the full evaluation pipeline."""

    # Load model
    model = get_model(model_name)

    # Load data
    samples = load_dataset(data_dir)
    if not samples:
        print(f"No transcript JSON files found in {data_dir}")
        sys.exit(1)

    print(f"{BOLD}NER Evaluation Report{RESET}")
    print(f"{'=' * 60}")
    print(f"Model:        {model.name}")
    print(f"Transcripts:  {len(samples)}")
    print(f"Data:         {data_dir}\n")

    # Run predictions
    all_true = []
    all_pred = []

    for sample in samples:
        tokens = sample["tokens"]
        true_tags = sample["ner_tags"]
        pred_tags = model.predict(tokens)

        assert len(pred_tags) == len(true_tags), (
            f"Length mismatch: {len(pred_tags)} predicted vs {len(true_tags)} true "
            f"for {sample.get('metadata', {}).get('transcript_id', '?')}"
        )

        all_true.append(true_tags)
        all_pred.append(pred_tags)

    # ---- Overall metrics ----

    token_acc = compute_token_accuracy(all_true, all_pred)
    entity_f1 = f1_score(all_true, all_pred)

    print(f"{BOLD}TOKEN-LEVEL ACCURACY:{RESET}  {token_acc:.4f} ({token_acc * 100:.1f}%)")
    print(f"{BOLD}ENTITY-LEVEL F1 (micro):{RESET} {entity_f1:.4f}\n")

    print(f"{BOLD}Per-Entity Classification Report:{RESET}")
    print(classification_report(all_true, all_pred, digits=4))

    # ---- Per-template breakdown ----

    template_metrics = compute_per_template_metrics(samples, all_true, all_pred)
    if len(template_metrics) > 1:
        print(f"{BOLD}Per-Template Breakdown:{RESET}")
        rows = [
            [m["template_id"], m["count"],
             f"{m['token_accuracy']:.4f}", f"{m['entity_f1']:.4f}",
             f"{m['worst_f1']:.4f}"]
            for m in sorted(template_metrics, key=lambda x: x["entity_f1"])
        ]
        print(tabulate(
            rows,
            headers=["Template", "Samples", "Token Acc", "Entity F1", "Worst F1"],
            tablefmt="simple",
        ))
        print()

    # ---- Tag distribution (verbose) ----

    if verbose:
        print(f"{BOLD}Tag Distribution:{RESET}")
        true_dist = compute_tag_distribution(all_true)
        pred_dist = compute_tag_distribution(all_pred)
        all_tags = sorted(set(list(true_dist.keys()) + list(pred_dist.keys())))
        dist_rows = [[tag, true_dist.get(tag, 0), pred_dist.get(tag, 0)]
                     for tag in all_tags]
        print(tabulate(dist_rows, headers=["Tag", "True Count", "Pred Count"],
                       tablefmt="simple"))
        print()

    # ---- Worst-case conversion analysis ----

    per_transcript = compute_per_transcript_metrics(samples, all_true, all_pred)
    worst_stats = compute_worst_case_stats(per_transcript)

    print(f"{BOLD}WORST-CASE CONVERSION ANALYSIS{RESET}")
    print(f"{'=' * 60}")

    if worst_stats:
        print(f"Worst transcript:       {worst_stats['worst_transcript_id']}")
        print(f"  Template:             {worst_stats['worst_template_id']}")
        print(f"  Entity F1:            {worst_stats['worst_entity_f1']:.4f}")
        wc = worst_stats['worst_courses_correct']
        wt = worst_stats['worst_courses_total']
        wr = worst_stats['worst_course_rate']
        print(f"  Course extraction:    {wc}/{wt} courses fully correct ({wr * 100:.1f}%)")
        print()
        print(f"5th percentile F1:      {worst_stats['percentile_5_f1']:.4f}")
        print(f"Mean F1:                {worst_stats['mean_f1']:.4f}")
        print(f"Mean course extraction: {worst_stats['mean_course_rate'] * 100:.1f}%")
        pc = worst_stats['perfect_count']
        tt = worst_stats['total_transcripts']
        pr = worst_stats['perfect_rate']
        print(f"Perfect extractions:    {pc}/{tt} ({pr * 100:.1f}%)")
        print()

    # Top N worst
    worst_list = find_worst_transcripts(per_transcript, n=n_worst)
    if worst_list:
        print(f"{BOLD}Top {len(worst_list)} Worst Transcripts:{RESET}")
        for i, m in enumerate(worst_list, 1):
            cc = m['courses_correct']
            ct = m['courses_total']
            cr = m['course_extraction_rate']
            print(f"  {i}. {m['transcript_id']:40s} "
                  f"F1: {m['entity_f1']:.3f}  "
                  f"Extraction: {cc}/{ct} ({cr * 100:.1f}%)")
        print()

    # ---- Error summary ----

    errors = compute_error_patterns(all_true, all_pred)
    if not errors:
        print(f"\n{GREEN}No errors found!{RESET}")
    else:
        print(f"{BOLD}Error Summary (true -> predicted):{RESET}")
        err_rows = [[t, "->", p, c] for t, p, c in errors[:15]]
        print(tabulate(err_rows, headers=["True", "", "Predicted", "Count"],
                       tablefmt="simple"))
    print()

    # ---- Qualitative examples ----

    if n_examples > 0:
        # Show examples from the worst transcripts
        print(f"{BOLD}Qualitative Examples "
              f"({min(n_examples, len(samples))} worst samples):{RESET}")
        for idx in range(min(n_examples, len(worst_list))):
            m = worst_list[idx]
            # Find the sample by transcript_id
            sample = next(
                s for s in samples
                if s.get("metadata", {}).get("transcript_id") == m["transcript_id"]
            )
            sample_idx = samples.index(sample)
            _show_example(
                sample["tokens"],
                all_true[sample_idx],
                all_pred[sample_idx],
                title=f"[{m['template_id']}] {m['transcript_id']}",
            )

    # ---- Charts ----

    if not no_charts:
        print(f"\n{BOLD}Generating charts...{RESET}")
        chart_paths = generate_all_charts(
            all_true=all_true,
            all_pred=all_pred,
            per_template_metrics=template_metrics,
            per_transcript_metrics=per_transcript,
            worst_case_stats=worst_stats,
            output_dir=output_dir,
        )
        print(f"Charts saved to: {Path(output_dir) / 'charts'}/")
        for p in chart_paths:
            print(f"  - {Path(p).name}")

    # ---- Final summary ----

    print(f"\n{'=' * 60}")
    print(f"{BOLD}Summary:{RESET}")
    print(f"  Model:                 {model.name}")
    print(f"  Transcripts evaluated: {len(samples)}")
    print(f"  Token accuracy:        {token_acc:.4f}")
    print(f"  Entity F1 (micro):     {entity_f1:.4f}")
    if worst_stats:
        print(f"  Worst-case F1:         {worst_stats['worst_entity_f1']:.4f}")
        print(f"  Perfect extractions:   {worst_stats['perfect_rate'] * 100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an NER model on generated transcript data.",
    )
    parser.add_argument(
        "--data", type=str, default="transcript_generator/output/",
        help="Path to the transcript output directory",
    )
    parser.add_argument(
        "--model", type=str, default="baseline",
        help="Model to evaluate (default: baseline)",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/output",
        help="Directory for charts and reports",
    )
    parser.add_argument(
        "--examples", type=int, default=3,
        help="Number of qualitative examples to show",
    )
    parser.add_argument(
        "--worst", type=int, default=5,
        help="Number of worst transcripts to detail",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show additional details (tag distribution)",
    )
    parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation",
    )
    args = parser.parse_args()

    evaluate(
        data_dir=args.data,
        model_name=args.model,
        output_dir=args.output,
        n_examples=args.examples,
        n_worst=args.worst,
        verbose=args.verbose,
        no_charts=args.no_charts,
    )


if __name__ == "__main__":
    main()
