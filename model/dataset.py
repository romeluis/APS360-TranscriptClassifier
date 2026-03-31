"""
PyTorch Dataset for BERT token-classification NER.

Handles:
  - Subword tokenisation alignment (first subword gets BIO label, rest → -100)
  - Sliding-window chunking for transcripts that exceed MAX_SEQ_LEN
"""

import torch
from torch.utils.data import Dataset

from .config import LABEL_TO_ID, MAX_SEQ_LEN, STRIDE


class TranscriptNERDataset(Dataset):
    """Pre-tokenises all samples and stores flat windows ready for training.

    Each item is a dict with:
        input_ids      : (max_length,) long tensor
        attention_mask : (max_length,) long tensor
        labels         : (max_length,) long tensor  (-100 for ignored positions)
    """

    def __init__(self, samples, tokenizer, max_length=MAX_SEQ_LEN, stride=STRIDE):
        self.windows = []
        self._prepare(samples, tokenizer, max_length, stride)

    # ------------------------------------------------------------------ #
    #  Preparation: tokenise + align labels for every sample / window     #
    # ------------------------------------------------------------------ #

    def _prepare(self, samples, tokenizer, max_length, stride):
        for sample_idx, sample in enumerate(samples):
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

            # encoding may contain multiple windows for long sequences
            for win_idx in range(encoding["input_ids"].size(0)):
                word_ids = encoding.word_ids(batch_index=win_idx)
                label_ids = self._align_labels(word_ids, ner_tags)

                self.windows.append({
                    "input_ids": encoding["input_ids"][win_idx],
                    "attention_mask": encoding["attention_mask"][win_idx],
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                })

    @staticmethod
    def _align_labels(word_ids, ner_tags):
        """Map BIO labels to subword positions.

        - Special tokens ([CLS], [SEP], [PAD]) → -100
        - First subword of a whitespace token → its BIO label id
        - Continuation subwords → -100
        """
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(LABEL_TO_ID[ner_tags[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        return label_ids

    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
