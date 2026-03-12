"""
Transcript visualizer — renders BIO-tagged tokens as semester/course tables.

Simulates what the iOS app would display after importing a transcript,
providing a visual check that extraction worked correctly.

Usage:
    python -m evaluation.visualize --file transcript_generator/output/tabular_classic/transcript_001.json
    python -m evaluation.visualize --file path/to/transcript.json --compare
    python -m evaluation.visualize --data transcript_generator/output/ --worst 3
"""

import argparse
import sys

from tabulate import tabulate

from .data_loader import load_dataset, load_single
from .metrics import (
    compute_per_transcript_metrics,
    find_worst_transcripts,
)
from .model_registry import get_model
from .reconstructor import reconstruct_semesters

# ANSI
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
DIM = "\033[2m"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _render_semesters(semesters: list[dict], label: str = ""):
    """Print semester tables to the terminal."""
    if label:
        print(f"\n{BOLD}{label}{RESET}")

    if not semesters:
        print(f"  {DIM}(no semesters found){RESET}")
        return

    for sem in semesters:
        print(f"\n  {CYAN}{BOLD}=== {sem['semester_name']} ==={RESET}")
        if not sem["courses"]:
            print(f"    {DIM}(no courses){RESET}")
            continue

        rows = []
        for c in sem["courses"]:
            rows.append([
                c["code"] or f"{DIM}—{RESET}",
                c["name"] or f"{DIM}—{RESET}",
                c["grade"] or f"{DIM}—{RESET}",
            ])

        print(tabulate(
            rows,
            headers=["Code", "Course Name", "Grade"],
            tablefmt="simple",
            stralign="left",
        ))


def _show_differences(true_semesters: list[dict], pred_semesters: list[dict]):
    """Show differences between ground truth and predicted extractions."""
    true_courses = []
    for sem in true_semesters:
        for c in sem["courses"]:
            true_courses.append(c)
    pred_courses = []
    for sem in pred_semesters:
        for c in sem["courses"]:
            pred_courses.append(c)

    n_true = len(true_courses)
    n_pred = len(pred_courses)
    n_match = min(n_true, n_pred)

    diffs = []
    for i in range(n_match):
        tc = true_courses[i]
        pc = pred_courses[i]
        course_diffs = []
        if tc["code"] != pc["code"]:
            course_diffs.append(f"code: '{tc['code']}' vs '{pc['code']}'")
        if tc["name"] != pc["name"]:
            course_diffs.append(f"name: '{tc['name']}' vs '{pc['name']}'")
        if tc["grade"] != pc["grade"]:
            course_diffs.append(f"grade: '{tc['grade']}' vs '{pc['grade']}'")
        if course_diffs:
            diffs.append((i + 1, course_diffs))

    print(f"\n  {BOLD}Differences:{RESET}")
    if not diffs and n_true == n_pred:
        print(f"    {GREEN}All courses match exactly.{RESET}")
    else:
        if n_pred < n_true:
            print(f"    {RED}Missing {n_true - n_pred} course(s) in prediction{RESET}")
        elif n_pred > n_true:
            print(f"    {YELLOW}{n_pred - n_true} extra course(s) in prediction{RESET}")
        for idx, field_diffs in diffs:
            print(f"    {RED}Course {idx}:{RESET}")
            for d in field_diffs:
                print(f"      - {d}")
        if not diffs and n_true != n_pred:
            print(f"    {GREEN}Overlapping courses all match.{RESET}")


# ---------------------------------------------------------------------------
# Visualization modes
# ---------------------------------------------------------------------------

def visualize_single(json_path: str, model_name: str = "baseline",
                     compare: bool = False):
    """Visualize a single transcript file."""
    sample = load_single(json_path)
    tokens = sample["tokens"]
    true_tags = sample["ner_tags"]
    meta = sample.get("metadata", {})

    tid = meta.get("transcript_id", "unknown")
    template = meta.get("template_id", "unknown")

    print(f"{BOLD}Transcript Visualizer{RESET}")
    print(f"{'=' * 60}")
    print(f"Transcript: {tid}")
    print(f"Template:   {template}")
    print(f"Tokens:     {len(tokens)}")

    model = get_model(model_name)
    pred_tags = model.predict(tokens)

    if compare:
        # Show both ground truth and predicted
        true_sems = reconstruct_semesters(tokens, true_tags)
        pred_sems = reconstruct_semesters(tokens, pred_tags)

        _render_semesters(true_sems, label=f"{GREEN}GROUND TRUTH{RESET}")
        _render_semesters(pred_sems, label=f"{YELLOW}PREDICTED ({model.name}){RESET}")
        _show_differences(true_sems, pred_sems)
    else:
        # Show predicted only (simulating app import)
        pred_sems = reconstruct_semesters(tokens, pred_tags)
        _render_semesters(pred_sems,
                         label=f"Extracted by {model.name}")


def visualize_worst(data_dir: str, model_name: str = "baseline",
                    n: int = 3):
    """Find and visualize the N worst-performing transcripts."""
    samples = load_dataset(data_dir)
    if not samples:
        print(f"No transcript JSON files found in {data_dir}")
        sys.exit(1)

    model = get_model(model_name)

    # Run predictions
    all_true = []
    all_pred = []
    for sample in samples:
        true_tags = sample["ner_tags"]
        pred_tags = model.predict(sample["tokens"])
        all_true.append(true_tags)
        all_pred.append(pred_tags)

    # Find worst
    per_transcript = compute_per_transcript_metrics(samples, all_true, all_pred)
    worst_list = find_worst_transcripts(per_transcript, n=n)

    print(f"{BOLD}Transcript Visualizer — {n} Worst Transcripts{RESET}")
    print(f"{'=' * 60}")
    print(f"Model: {model.name}")
    print(f"Total transcripts: {len(samples)}\n")

    for i, m in enumerate(worst_list, 1):
        # Find the sample
        sample = next(
            s for s in samples
            if s.get("metadata", {}).get("transcript_id") == m["transcript_id"]
        )
        sample_idx = samples.index(sample)

        print(f"\n{'─' * 60}")
        print(f"{BOLD}#{i} — {m['transcript_id']}{RESET}")
        print(f"  Template: {m['template_id']}")
        print(f"  Entity F1: {m['entity_f1']:.4f}")
        cc = m['courses_correct']
        ct = m['courses_total']
        cr = m['course_extraction_rate']
        print(f"  Course extraction: {cc}/{ct} ({cr * 100:.1f}%)")

        tokens = sample["tokens"]
        true_tags = all_true[sample_idx]
        pred_tags = all_pred[sample_idx]

        true_sems = reconstruct_semesters(tokens, true_tags)
        pred_sems = reconstruct_semesters(tokens, pred_tags)

        _render_semesters(true_sems, label=f"{GREEN}GROUND TRUTH{RESET}")
        _render_semesters(pred_sems, label=f"{YELLOW}PREDICTED{RESET}")
        _show_differences(true_sems, pred_sems)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize transcript extraction as semester/course tables.",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a single transcript JSON file",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to transcript output directory (for --worst mode)",
    )
    parser.add_argument(
        "--model", type=str, default="baseline",
        help="Model to use for predictions (default: baseline)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Show ground truth and predicted side by side",
    )
    parser.add_argument(
        "--worst", type=int, default=None,
        help="Visualize the N worst-performing transcripts (requires --data)",
    )

    args = parser.parse_args()

    if args.file:
        visualize_single(args.file, model_name=args.model, compare=args.compare)
    elif args.data and args.worst:
        visualize_worst(args.data, model_name=args.model, n=args.worst)
    elif args.data:
        # Default: visualize first transcript
        samples = load_dataset(args.data)
        if not samples:
            print(f"No transcripts found in {args.data}")
            sys.exit(1)
        # Find first json file to use
        from pathlib import Path
        first_json = sorted(Path(args.data).rglob("transcript_*.json"))[0]
        visualize_single(str(first_json), model_name=args.model,
                        compare=args.compare)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
