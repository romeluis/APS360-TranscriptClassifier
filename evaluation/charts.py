"""
Chart generation for NER evaluation reports.

Generates and saves matplotlib charts as PNG files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from seqeval.metrics import classification_report


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _setup_style():
    """Configure a clean style for academic-quality charts."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


# ---------------------------------------------------------------------------
# Chart 1: Per-entity F1
# ---------------------------------------------------------------------------

def chart_per_entity_f1(
    all_true: list[list[str]],
    all_pred: list[list[str]],
    save_path: str,
):
    """Grouped bar chart: precision, recall, F1 per entity type + micro avg."""
    _setup_style()

    report = classification_report(
        all_true, all_pred, output_dict=True, digits=4,
    )

    # Entity types to plot (skip 'micro avg', 'macro avg', 'weighted avg' for now)
    entity_types = [k for k in report if k not in ("micro avg", "macro avg", "weighted avg")]
    entity_types.sort()

    # Add micro avg at the end
    labels = entity_types + ["micro avg"]

    precisions = [report[k]["precision"] for k in labels]
    recalls = [report[k]["recall"] for k in labels]
    f1s = [report[k]["f1-score"] for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_p = ax.bar(x - width, precisions, width, label="Precision", color="#4C72B0")
    bars_r = ax.bar(x, recalls, width, label="Recall", color="#55A868")
    bars_f = ax.bar(x + width, f1s, width, label="F1", color="#C44E52")

    ax.set_ylabel("Score")
    ax.set_title("Per-Entity NER Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Value labels on bars
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: Per-template accuracy
# ---------------------------------------------------------------------------

def chart_per_template_accuracy(
    per_template_metrics: list[dict],
    save_path: str,
):
    """Horizontal bar chart: entity F1 per template, sorted worst to best."""
    _setup_style()

    # Sort by entity F1 ascending (worst at top)
    sorted_metrics = sorted(per_template_metrics, key=lambda m: m["entity_f1"])

    labels = [m["template_id"] for m in sorted_metrics]
    f1s = [m["entity_f1"] for m in sorted_metrics]

    # Color code
    colors = []
    for f1 in f1s:
        if f1 >= 0.95:
            colors.append("#55A868")  # green
        elif f1 >= 0.80:
            colors.append("#CCAA44")  # yellow
        else:
            colors.append("#C44E52")  # red

    fig_height = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y = np.arange(len(labels))
    ax.barh(y, f1s, color=colors, edgecolor="none")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Entity F1")
    ax.set_title("Entity F1 by Template (sorted worst → best)")
    ax.set_xlim(0, 1.05)

    # Add value labels
    for i, f1 in enumerate(f1s):
        ax.text(f1 + 0.005, i, f"{f1:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: Worst-case analysis
# ---------------------------------------------------------------------------

def chart_worst_case_analysis(
    per_transcript_metrics: list[dict],
    worst_case_stats: dict,
    save_path: str,
):
    """Two histograms: entity F1 distribution and course extraction rate distribution."""
    _setup_style()

    f1s = [m["entity_f1"] for m in per_transcript_metrics]
    rates = [m["course_extraction_rate"] for m in per_transcript_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Entity F1 distribution ---
    ax1.hist(f1s, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Entity F1")
    ax1.set_ylabel("Transcript Count")
    ax1.set_title("Per-Transcript Entity F1 Distribution")

    mean_f1 = worst_case_stats.get("mean_f1", np.mean(f1s))
    p5_f1 = worst_case_stats.get("percentile_5_f1", np.percentile(f1s, 5))
    min_f1 = worst_case_stats.get("worst_entity_f1", min(f1s))

    ax1.axvline(mean_f1, color="#55A868", linestyle="--", linewidth=1.5, label=f"Mean: {mean_f1:.3f}")
    ax1.axvline(p5_f1, color="#CCAA44", linestyle="--", linewidth=1.5, label=f"5th %ile: {p5_f1:.3f}")
    ax1.axvline(min_f1, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Min: {min_f1:.3f}")
    ax1.legend(fontsize=8)

    # --- Right: Course extraction rate distribution ---
    ax2.hist(rates, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Course Extraction Rate")
    ax2.set_ylabel("Transcript Count")
    ax2.set_title("Per-Transcript Course Extraction Rate")

    mean_rate = worst_case_stats.get("mean_course_rate", np.mean(rates))
    min_rate = min(rates) if rates else 0
    ax2.axvline(mean_rate, color="#4C72B0", linestyle="--", linewidth=1.5, label=f"Mean: {mean_rate:.3f}")
    ax2.axvline(min_rate, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Min: {min_rate:.3f}")
    ax2.legend(fontsize=8)

    fig.suptitle("Worst-Case Conversion Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Confusion matrix
# ---------------------------------------------------------------------------

def chart_confusion_matrix(
    all_true: list[list[str]],
    all_pred: list[list[str]],
    save_path: str,
):
    """Tag-level confusion matrix heatmap."""
    _setup_style()

    from collections import defaultdict

    # Count co-occurrences
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    all_tags_set: set[str] = set()
    for true_tags, pred_tags in zip(all_true, all_pred):
        for t, p in zip(true_tags, pred_tags):
            pair_counts[(t, p)] += 1
            all_tags_set.add(t)
            all_tags_set.add(p)

    # Sort tags: O first, then alphabetical
    tag_order = sorted(all_tags_set - {"O"})
    tag_order = ["O"] + tag_order

    n = len(tag_order)
    matrix = np.zeros((n, n), dtype=int)
    for i, true_tag in enumerate(tag_order):
        for j, pred_tag in enumerate(tag_order):
            matrix[i, j] = pair_counts.get((true_tag, pred_tag), 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tag_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tag_order, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Tag-Level Confusion Matrix")

    # Annotate cells with counts (skip zeros for readability)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generate all
# ---------------------------------------------------------------------------

def generate_all_charts(
    all_true: list[list[str]],
    all_pred: list[list[str]],
    per_template_metrics: list[dict],
    per_transcript_metrics: list[dict],
    worst_case_stats: dict,
    output_dir: str,
):
    """Generate all charts and save to output_dir/charts/.

    Returns list of saved file paths.
    """
    charts_dir = Path(output_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    path = str(charts_dir / "per_entity_f1.png")
    chart_per_entity_f1(all_true, all_pred, path)
    paths.append(path)

    path = str(charts_dir / "per_template_accuracy.png")
    chart_per_template_accuracy(per_template_metrics, path)
    paths.append(path)

    path = str(charts_dir / "worst_case_analysis.png")
    chart_worst_case_analysis(per_transcript_metrics, worst_case_stats, path)
    paths.append(path)

    path = str(charts_dir / "confusion_matrix.png")
    chart_confusion_matrix(all_true, all_pred, path)
    paths.append(path)

    return paths
