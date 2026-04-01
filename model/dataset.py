"""
PyTorch Dataset for BERT token-classification NER.

Handles:
  - Subword tokenisation alignment (first subword gets BIO label, rest → -100)
  - Sliding-window chunking for transcripts that exceed MAX_SEQ_LEN
  - Optional online augmentation: raw samples are augmented at __getitem__ time
    so the model sees a different noisy version of each sample every epoch.
"""

import random
import torch
from torch.utils.data import Dataset

from .config import LABEL_TO_ID, MAX_SEQ_LEN, STRIDE


class TranscriptNERDataset(Dataset):
    """Dataset supporting both pre-tokenised sliding windows and online augmentation.

    augment=False (default — used for val/test):
        Pre-tokenises all samples into sliding windows at construction time.
        __len__ == number of windows. Fast DataLoader with zero per-item overhead.

    augment=True (used for train):
        Stores raw (tokens, ner_tags) pairs. At __getitem__ time, applies
        augment_tokens() with a fresh random seed then tokenises on-the-fly
        (truncation only, no overflow windows). __len__ == number of raw samples.
        The model sees a different noisy version of each sample every epoch,
        which significantly reduces overfitting.
    """

    def __init__(
        self,
        samples,
        tokenizer,
        max_length=MAX_SEQ_LEN,
        stride=STRIDE,
        augment=False,
        augment_seed=42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.augment = augment

        if augment:
            # Store raw samples; tokenise and augment on the fly in __getitem__
            self._raw_samples = [
                (sample["tokens"], sample["ner_tags"]) for sample in samples
            ]
            # Shared RNG advances on every __getitem__ call, giving different
            # augmentation each time (and across epochs) without fixing noise
            # by sample index.
            self._rng = random.Random(augment_seed)
        else:
            # Original behaviour: pre-tokenise all windows at construction time
            self.windows = []
            self._prepare(samples, tokenizer, max_length, stride)

    # ------------------------------------------------------------------ #
    #  Pre-tokenisation path (val / test)                                  #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Online-augmentation path (train)                                    #
    # ------------------------------------------------------------------ #

    def _tokenize_single(self, tokens, ner_tags):
        """Tokenise one (tokens, ner_tags) pair with truncation only (no overflow).

        Long transcripts are silently truncated to max_length. This is acceptable
        for training because augmentation provides variety; the sliding-window path
        is retained for val/test evaluation.
        """
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

    # ------------------------------------------------------------------ #
    #  Label alignment (shared by both paths)                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _align_labels(word_ids, ner_tags):
        """Map BIO labels to subword positions.

        - Special tokens ([CLS], [SEP], [PAD]) → -100
        - First subword of a whitespace token → its BIO label id
        - Continuation subwords → -100

        The word_id guard (word_id < len(ner_tags)) handles edge cases where
        augmentation grows the token list beyond what was originally labelled
        (e.g. _split_tokens can append a new token beyond the last label).
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

    # ------------------------------------------------------------------ #
    #  Dataset protocol                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self):
        if self.augment:
            return len(self._raw_samples)
        return len(self.windows)

    def __getitem__(self, idx):
        if not self.augment:
            return self.windows[idx]

        # Deferred import to avoid circular dependency at module load time
        from transcript_generator.noise_augmenter import augment_tokens

        tokens, ner_tags = self._raw_samples[idx]
        aug_tokens, aug_tags = augment_tokens(tokens, ner_tags, rng=self._rng)
        return self._tokenize_single(aug_tokens, aug_tags)
