"""
Template-level train / val / test splitting.

Templates (not individual transcripts) are the unit of splitting so the
test set contains transcript formats never seen during training, validating
true generalisation to unseen layouts.
"""

import json
import random
import sys
from pathlib import Path

# Allow imports from sibling packages
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.data_loader import group_by_template, load_dataset

from .config import DATA_DIR, SEED, TRAIN_RATIO, VAL_RATIO


def split_by_template(
    data_dir: str = DATA_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Split samples into train / val / test by template ID.

    Args:
        data_dir: path to transcript output directory
        train_ratio: fraction of templates for training
        val_ratio: fraction of templates for validation
        seed: random seed for reproducibility

    Returns:
        (train_samples, val_samples, test_samples, split_info)
    """
    samples = load_dataset(data_dir)
    groups = group_by_template(samples)
    template_ids = sorted(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(template_ids)

    n = len(template_ids)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    train_ids = template_ids[:n_train]
    val_ids = template_ids[n_train : n_train + n_val]
    test_ids = template_ids[n_train + n_val :]

    train_samples = [s for tid in train_ids for s in groups[tid]]
    val_samples = [s for tid in val_ids for s in groups[tid]]
    test_samples = [s for tid in test_ids for s in groups[tid]]

    split_info = {
        "seed": seed,
        "total_templates": n,
        "train_templates": train_ids,
        "val_templates": val_ids,
        "test_templates": test_ids,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }

    return train_samples, val_samples, test_samples, split_info


def save_split_info(split_info: dict, path: str = "model/split_info.json"):
    """Persist the split mapping for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split_info, f, indent=2)


# ── CLI quick-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train, val, test, info = split_by_template()
    save_split_info(info)

    print(f"Templates  : {info['total_templates']}")
    print(f"Train      : {len(info['train_templates'])} templates, {info['train_samples']} samples")
    print(f"Val        : {len(info['val_templates'])} templates, {info['val_samples']} samples")
    print(f"Test       : {len(info['test_templates'])} templates, {info['test_samples']} samples")
