"""
PyTorch Dataset for token-classification NER.

Handles:
  - Subword tokenisation alignment (first subword gets BIO label, rest -> -100)
  - Sliding-window chunking for transcripts exceeding MAX_SEQ_LEN
  - Optional online augmentation: raw samples augmented at __getitem__ time
  - Data loading from JSON files
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import LABEL_TO_ID, MAX_SEQ_LEN, STRIDE


# ── Data loading ──────��───────────────────────────────────────────────────


def load_dataset(data_dir: str) -> list[dict]:
    """Load all transcript JSON files from a directory tree.

    Each JSON file must contain: tokens (list[str]), ner_tags (list[str]),
    metadata (dict with at least template_id).
    """
    data_path = Path(data_dir)
    samples = []
    for json_path in sorted(data_path.rglob("transcript_*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if "tokens" in data and "ner_tags" in data:
            samples.append(data)
    return samples


def group_by_template(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by their template_id."""
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        tid = sample.get("metadata", {}).get("template_id", "unknown")
        groups.setdefault(tid, []).append(sample)
    return groups


def split_by_template(
    data_dir: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Split samples into train/val/test by template ID.

    Returns (train_samples, val_samples, test_samples, split_info).
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
    val_ids = template_ids[n_train:n_train + n_val]
    test_ids = template_ids[n_train + n_val:]

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


# ── Dataset ─────────────���────────────────────────────���────────────────────


class TranscriptNERDataset(Dataset):
    """Dataset supporting both pre-tokenised sliding windows and online augmentation.

    augment=False (val/test): Pre-tokenises all windows at construction.
    augment=True (train): Stores raw samples; augments + tokenises on-the-fly.
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_length: int = MAX_SEQ_LEN,
        stride: int = STRIDE,
        augment: bool = False,
        augment_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.augment = augment

        if augment:
            self._raw_samples = [
                (sample["tokens"], sample["ner_tags"]) for sample in samples
            ]
            self._rng = random.Random(augment_seed)
        else:
            self.windows: list[dict] = []
            self._prepare(samples, tokenizer, max_length, stride)

    def _prepare(self, samples, tokenizer, max_length, stride):
        for sample in samples:
            tokens = sample["tokens"]
            ner_tags = sample["ner_tags"]

            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_overflowing_tokens=True,
                stride=stride,
                return_tensors="pt",
            )

            for win_idx in range(encoding["input_ids"].size(0)):
                word_ids = encoding.word_ids(batch_index=win_idx)
                label_ids = self._align_labels(word_ids, ner_tags)

                self.windows.append({
                    "input_ids": encoding["input_ids"][win_idx],
                    "attention_mask": encoding["attention_mask"][win_idx],
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                })

    def _tokenize_single(self, tokens, ner_tags):
        """Tokenise one sample with truncation only (no overflow windows)."""
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=False,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = self._align_labels(word_ids, ner_tags)

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

    @staticmethod
    def _align_labels(word_ids, ner_tags):
        """Map BIO labels to subword positions.

        Special tokens -> -100, first subword -> BIO label, continuations -> -100.
        """
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                if word_id < len(ner_tags):
                    label_ids.append(LABEL_TO_ID[ner_tags[word_id]])
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        return label_ids

    def __len__(self):
        if self.augment:
            return len(self._raw_samples)
        return len(self.windows)

    def __getitem__(self, idx):
        if not self.augment:
            return self.windows[idx]

        from transcript_generator.noise_augmenter import augment_tokens

        tokens, ner_tags = self._raw_samples[idx]
        aug_tokens, aug_tags = augment_tokens(tokens, ner_tags, rng=self._rng)
        return self._tokenize_single(aug_tokens, aug_tags)
