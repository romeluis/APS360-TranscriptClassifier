"""CLI entry point for evaluation.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --data data/output --checkpoint model/checkpoints/best
"""

import argparse
import json

from seqeval.metrics import classification_report, f1_score as seqeval_f1

from model.config import BEST_CHECKPOINT_DIR, DATA_DIR, SEED
from model.dataset import split_by_template
from model.predictor import NERPredictor
from pipeline.reconstructor import reconstruct_semesters


def _course_extraction_rate(samples: list[dict], predictor: NERPredictor) -> dict:
    """Compute how many courses are perfectly extracted (code + name + grade match)."""
    total_courses = 0
    perfect_courses = 0

    for sample in samples:
        tokens = sample["tokens"]
        true_tags = sample["ner_tags"]

        pred_tags = predictor.predict(tokens)

        true_sems = reconstruct_semesters(tokens, true_tags)
        pred_sems = reconstruct_semesters(tokens, pred_tags)

        # Flatten to course sets for comparison
        true_courses = set()
        for sem in true_sems:
            for c in sem["courses"]:
                true_courses.add((c.get("code"), c.get("name"), c.get("grade")))

        pred_courses = set()
        for sem in pred_sems:
            for c in sem["courses"]:
                pred_courses.add((c.get("code"), c.get("name"), c.get("grade")))

        total_courses += len(true_courses)
        perfect_courses += len(true_courses & pred_courses)

    rate = perfect_courses / total_courses if total_courses > 0 else 0.0
    return {
        "total_courses": total_courses,
        "perfect_courses": perfect_courses,
        "extraction_rate": rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Transcript NER model")
    parser.add_argument("--data", default=DATA_DIR, help="Path to data directory")
    parser.add_argument("--checkpoint", default=BEST_CHECKPOINT_DIR, help="Model checkpoint")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Which split to evaluate")
    args = parser.parse_args()

    print("Loading data...")
    train_samples, val_samples, test_samples, split_info = split_by_template(
        data_dir=args.data, seed=args.seed,
    )
    eval_samples = test_samples if args.split == "test" else val_samples
    print(f"Evaluating on {args.split} split: {len(eval_samples)} samples")

    print(f"Loading model from {args.checkpoint}...")
    predictor = NERPredictor(args.checkpoint)

    # Token-level evaluation
    print("\nRunning predictions...")
    all_true, all_pred = [], []
    for sample in eval_samples:
        tokens = sample["tokens"]
        true_tags = sample["ner_tags"]
        pred_tags = predictor.predict(tokens)
        all_true.append(true_tags)
        all_pred.append(pred_tags)

    # seqeval report
    print("\n" + "=" * 60)
    print("Entity-level Classification Report:")
    print("=" * 60)
    print(classification_report(all_true, all_pred))

    f1 = seqeval_f1(all_true, all_pred)
    print(f"Overall Entity F1: {f1:.4f}")

    # Token accuracy
    correct = sum(t == p for ts, ps in zip(all_true, all_pred) for t, p in zip(ts, ps))
    total = sum(len(ts) for ts in all_true)
    print(f"Token Accuracy: {correct / total:.4f}" if total > 0 else "Token Accuracy: N/A")

    # Course extraction rate
    print("\nComputing course extraction rate...")
    cer = _course_extraction_rate(eval_samples, predictor)
    print(f"Course Extraction Rate: {cer['extraction_rate']:.4f} ({cer['perfect_courses']}/{cer['total_courses']})")

    # Per-template breakdown
    print("\nPer-template F1:")
    from model.dataset import group_by_template
    groups = group_by_template(eval_samples)
    template_results = {}
    for tid, samples in sorted(groups.items()):
        t_true, t_pred = [], []
        for s in samples:
            t_true.append(s["ner_tags"])
            t_pred.append(predictor.predict(s["tokens"]))
        t_f1 = seqeval_f1(t_true, t_pred)
        template_results[tid] = t_f1
        print(f"  {tid}: {t_f1:.4f} ({len(samples)} samples)")

    # Worst templates
    sorted_templates = sorted(template_results.items(), key=lambda x: x[1])
    print(f"\nWorst 3 templates:")
    for tid, f1_val in sorted_templates[:3]:
        print(f"  {tid}: {f1_val:.4f}")


if __name__ == "__main__":
    main()
