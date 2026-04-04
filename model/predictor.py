"""
Inference for the TranscriptNER model (DeBERTa + CRF).

Provides NERPredictor: predict(tokens) -> BIO tags.

For multi-window transcripts, uses centrality-weighted softmax averaging
across overlapping windows before argmax.
"""

import numpy as np
import torch
from transformers import AutoTokenizer

from .config import BEST_CHECKPOINT_DIR, ID_TO_LABEL, MAX_SEQ_LEN, NUM_LABELS, STRIDE
from .architecture import TranscriptNERModel


class NERPredictor:
    """Loads a trained checkpoint and predicts BIO tags for whitespace tokens."""

    def __init__(
        self,
        checkpoint_dir: str = BEST_CHECKPOINT_DIR,
        device: torch.device | str | None = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = TranscriptNERModel.from_pretrained(checkpoint_dir, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, tokens: list[str]) -> list[str]:
        """Predict BIO NER tags for whitespace tokens.

        Returns list of BIO tag strings, same length as tokens.
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=STRIDE,
            return_tensors="pt",
        )

        num_windows = encoding["input_ids"].size(0)

        if num_windows == 1:
            return self._predict_single(tokens, encoding)
        return self._predict_multiwindow(tokens, encoding, num_windows)

    def _predict_single(self, tokens, encoding):
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        crf_paths, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pred_ids = crf_paths[0]
        word_ids = encoding.word_ids(batch_index=0)

        return self._realign(tokens, word_ids, pred_ids)

    def _predict_multiwindow(self, tokens, encoding, num_windows):
        """Centrality-weighted softmax averaging across overlapping windows."""
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        _, emissions = self.model(input_ids=input_ids, attention_mask=attention_mask)
        all_probs = torch.softmax(emissions, dim=-1).cpu().numpy()

        n_tokens = len(tokens)
        token_prob_sum = np.zeros((n_tokens, NUM_LABELS), dtype=np.float64)
        token_weight_sum = np.zeros(n_tokens, dtype=np.float64)

        centre = MAX_SEQ_LEN / 2.0

        for win_idx in range(num_windows):
            word_ids = encoding.word_ids(batch_index=win_idx)
            probs = all_probs[win_idx]
            seen = set()

            for pos, word_id in enumerate(word_ids):
                if word_id is None or word_id >= n_tokens or word_id in seen:
                    continue
                seen.add(word_id)

                centrality = 1.0 - abs(pos - centre) / centre
                weight = centrality + 0.5

                token_prob_sum[word_id] += weight * probs[pos]
                token_weight_sum[word_id] += weight

        predictions = []
        for i in range(n_tokens):
            if token_weight_sum[i] > 0:
                avg_probs = token_prob_sum[i] / token_weight_sum[i]
                predictions.append(ID_TO_LABEL[int(np.argmax(avg_probs))])
            else:
                predictions.append("O")

        return self._fix_bio_consistency(predictions)

    @staticmethod
    def _realign(tokens, word_ids, pred_ids):
        """Map subword predictions back to one tag per whitespace token."""
        n = len(tokens)
        tags = ["O"] * n
        seen = set()
        for pos, word_id in enumerate(word_ids):
            if word_id is None or word_id >= n or word_id in seen:
                continue
            seen.add(word_id)
            if pos < len(pred_ids):
                tags[word_id] = ID_TO_LABEL[pred_ids[pos]]

        return NERPredictor._fix_bio_consistency(tags)

    @staticmethod
    def _fix_bio_consistency(tags: list[str]) -> list[str]:
        """Fix I-tags not preceded by a matching B- or I- tag."""
        fixed = list(tags)
        for i, tag in enumerate(fixed):
            if tag.startswith("I-"):
                entity = tag[2:]
                if i == 0 or fixed[i - 1] not in (f"B-{entity}", f"I-{entity}"):
                    fixed[i] = f"B-{entity}"
        return fixed
