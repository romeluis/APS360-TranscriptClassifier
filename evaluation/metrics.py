"""
Metric computation for NER evaluation.

Provides token-level accuracy, per-transcript metrics, course extraction
rate, worst-case analysis, and per-template aggregation.
"""

from collections import defaultdict

import numpy as np
from seqeval.metrics import f1_score

from .reconstructor import reconstruct_semesters


# ---------------------------------------------------------------------------
# Token-level
# ---------------------------------------------------------------------------

def compute_token_accuracy(all_true: list[list[str]],
                           all_pred: list[list[str]]) -> float:
    """Flat token-level accuracy across all samples."""
    correct = 0
    total = 0
    for true_tags, pred_tags in zip(all_true, all_pred):
        for t, p in zip(true_tags, pred_tags):
            total += 1
            if t == p:
                correct += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Course extraction rate (single transcript)
# ---------------------------------------------------------------------------

def _courses_match(true_course: dict, pred_course: dict) -> bool:
    """Check if code, name, AND grade all match exactly."""
    return (
        true_course["code"] == pred_course["code"]
        and true_course["name"] == pred_course["name"]
        and true_course["grade"] == pred_course["grade"]
    )


def compute_course_extraction_rate(
    tokens: list[str],
    true_tags: list[str],
    pred_tags: list[str],
) -> tuple[float, int, int]:
    """Compute what fraction of courses were fully correctly extracted.

    A course is "fully correct" only if its code, name, and grade all
    match exactly between ground truth and prediction (positional matching).

    Returns:
        (rate, correct_count, total_count)
    """
    true_sems = reconstruct_semesters(tokens, true_tags)
    pred_sems = reconstruct_semesters(tokens, pred_tags)

    # Flatten to ordered lists for positional matching
    true_courses = []
    for sem in true_sems:
        true_courses.extend(sem["courses"])
    pred_courses = []
    for sem in pred_sems:
        pred_courses.extend(sem["courses"])

    total = len(true_courses)
    if total == 0:
        return 1.0, 0, 0

    correct = 0
    for i in range(min(len(true_courses), len(pred_courses))):
        if _courses_match(true_courses[i], pred_courses[i]):
            correct += 1

    return correct / total, correct, total


# ---------------------------------------------------------------------------
# Per-transcript metrics
# ---------------------------------------------------------------------------

def compute_per_transcript_metrics(
    samples: list[dict],
    all_true: list[list[str]],
    all_pred: list[list[str]],
) -> list[dict]:
    """Compute metrics for each individual transcript.

    Returns:
        list of dicts with keys: transcript_id, template_id,
        token_accuracy, entity_f1, course_extraction_rate,
        courses_correct, courses_total
    """
    results = []
    for sample, true_tags, pred_tags in zip(samples, all_true, all_pred):
        meta = sample.get("metadata", {})

        # Token accuracy
        correct = sum(1 for t, p in zip(true_tags, pred_tags) if t == p)
        tok_acc = correct / len(true_tags) if true_tags else 0.0

        # Entity F1 (seqeval on single sample)
        ent_f1 = f1_score([true_tags], [pred_tags])

        # Course extraction rate
        rate, c_correct, c_total = compute_course_extraction_rate(
            sample["tokens"], true_tags, pred_tags,
        )

        results.append({
            "transcript_id": meta.get("transcript_id", "unknown"),
            "template_id": meta.get("template_id", "unknown"),
            "token_accuracy": tok_acc,
            "entity_f1": ent_f1,
            "course_extraction_rate": rate,
            "courses_correct": c_correct,
            "courses_total": c_total,
        })

    return results


# ---------------------------------------------------------------------------
# Worst-case analysis
# ---------------------------------------------------------------------------

def compute_worst_case_stats(per_transcript: list[dict]) -> dict:
    """Compute worst-case conversion statistics.

    Returns dict with:
        worst_entity_f1, worst_transcript_id, worst_template_id,
        worst_course_rate, percentile_5_f1, perfect_count, perfect_rate
    """
    if not per_transcript:
        return {}

    f1s = np.array([m["entity_f1"] for m in per_transcript])
    rates = np.array([m["course_extraction_rate"] for m in per_transcript])

    worst_idx = int(np.argmin(f1s))
    worst = per_transcript[worst_idx]

    perfect = sum(1 for r in rates if r >= 1.0)

    return {
        "worst_entity_f1": worst["entity_f1"],
        "worst_course_rate": worst["course_extraction_rate"],
        "worst_transcript_id": worst["transcript_id"],
        "worst_template_id": worst["template_id"],
        "worst_courses_correct": worst["courses_correct"],
        "worst_courses_total": worst["courses_total"],
        "percentile_5_f1": float(np.percentile(f1s, 5)),
        "mean_f1": float(np.mean(f1s)),
        "median_f1": float(np.median(f1s)),
        "mean_course_rate": float(np.mean(rates)),
        "perfect_count": perfect,
        "perfect_rate": perfect / len(per_transcript),
        "total_transcripts": len(per_transcript),
    }


def find_worst_transcripts(per_transcript: list[dict], n: int = 5) -> list[dict]:
    """Return the N worst-performing transcripts by entity F1."""
    return sorted(per_transcript, key=lambda m: m["entity_f1"])[:n]


# ---------------------------------------------------------------------------
# Per-template aggregation
# ---------------------------------------------------------------------------

def compute_per_template_metrics(
    samples: list[dict],
    all_true: list[list[str]],
    all_pred: list[list[str]],
) -> list[dict]:
    """Aggregate metrics per template.

    Returns list of dicts with: template_id, count, token_accuracy,
    entity_f1, worst_f1
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        tid = sample.get("metadata", {}).get("template_id", "unknown")
        groups[tid].append(idx)

    results = []
    for tid, indices in sorted(groups.items()):
        g_true = [all_true[i] for i in indices]
        g_pred = [all_pred[i] for i in indices]

        # Token accuracy
        correct = sum(
            1 for true_s, pred_s in zip(g_true, g_pred)
            for t, p in zip(true_s, pred_s) if t == p
        )
        total = sum(len(s) for s in g_true)
        tok_acc = correct / total if total > 0 else 0.0

        # Entity F1
        ent_f1 = f1_score(g_true, g_pred)

        # Worst individual F1 in this template group
        individual_f1s = [f1_score([t], [p]) for t, p in zip(g_true, g_pred)]
        worst_f1 = min(individual_f1s) if individual_f1s else 0.0

        results.append({
            "template_id": tid,
            "count": len(indices),
            "token_accuracy": tok_acc,
            "entity_f1": ent_f1,
            "worst_f1": worst_f1,
        })

    return results


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def compute_error_patterns(
    all_true: list[list[str]],
    all_pred: list[list[str]],
) -> list[tuple[str, str, int]]:
    """Find the most common error patterns (true_tag -> pred_tag).

    Returns list of (true_tag, pred_tag, count) sorted by count descending.
    """
    error_counts: dict[tuple[str, str], int] = defaultdict(int)
    for true_tags, pred_tags in zip(all_true, all_pred):
        for t, p in zip(true_tags, pred_tags):
            if t != p:
                error_counts[(t, p)] += 1

    return sorted(
        [(t, p, c) for (t, p), c in error_counts.items()],
        key=lambda x: -x[2],
    )


def compute_tag_distribution(
    tags_list: list[list[str]],
) -> dict[str, int]:
    """Count occurrences of each tag across all samples."""
    counts: dict[str, int] = defaultdict(int)
    for tags in tags_list:
        for t in tags:
            counts[t] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
