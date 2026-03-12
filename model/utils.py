"""
Utility functions: checkpoint management and learning-curve plotting.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


# ── Learning-curve plots ──────────────────────────────────────────────────


def plot_learning_curves(
    history: dict | str,
    save_path: str = "model/logs/learning_curves.png",
):
    """Plot train/val loss and validation metrics over epochs.

    Args:
        history: dict with keys train_loss, val_loss, val_token_acc,
                 val_entity_f1  (each a list of floats); or a path to
                 history.json.
        save_path: where to save the figure.
    """
    if isinstance(history, (str, Path)):
        with open(history) as f:
            history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metric curves
    ax2.plot(epochs, history["val_entity_f1"], "o-", label="Val Entity F1", color="green")
    ax2.plot(epochs, history["val_token_acc"], "s-", label="Val Token Accuracy", color="royalblue")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves saved to {save_path}")


def load_history(path: str = "model/logs/history.json") -> dict:
    """Load a saved training history JSON."""
    with open(path) as f:
        return json.load(f)
