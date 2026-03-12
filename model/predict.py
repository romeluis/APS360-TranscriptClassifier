"""
Inference for the BERT NER model.

Provides `BertNERPredictor` which implements the same interface expected by
the evaluation framework: `predict(tokens: list[str]) -> list[str]`.

Handles subword-to-whitespace-token realignment and sliding-window merging
for long transcripts.
"""

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

from .config import BEST_CHECKPOINT_DIR, ID_TO_LABEL, MAX_SEQ_LEN, STRIDE


class BertNERPredictor:
    """Loads a trained checkpoint and predicts BIO tags for whitespace tokens."""

    def __init__(
        self,
        checkpoint_dir: str = BEST_CHECKPOINT_DIR,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.tokenizer = BertTokenizerFast.from_pretrained(checkpoint_dir)
        self.model = BertForTokenClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, tokens: list[str]) -> list[str]:
        """Predict BIO NER tags for a list of whitespace tokens.

        Returns:
            list of BIO tag strings, same length as *tokens*.
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

    # ------------------------------------------------------------------ #
    #  Single-window (most transcripts)                                    #
    # ------------------------------------------------------------------ #

    def _predict_single(self, tokens, encoding):
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits[0]  # (seq_len, num_labels)

        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        return self._realign(tokens, word_ids, pred_ids)

    # ------------------------------------------------------------------ #
    #  Multi-window (long transcripts > 512 subwords)                      #
    # ------------------------------------------------------------------ #

    def _predict_multiwindow(self, tokens, encoding, num_windows):
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        all_logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits  # (num_windows, seq_len, num_labels)

        # For each whitespace token, collect (confidence, label_id) from all
        # windows; keep the one from the most-centred subword position.
        n_tokens = len(tokens)
        best = [(-1.0, 0)] * n_tokens  # (score, label_id)

        for win_idx in range(num_windows):
            word_ids = encoding.word_ids(batch_index=win_idx)
            logits = all_logits[win_idx]  # (seq_len, num_labels)
            confs = logits.softmax(dim=-1).cpu()

            seen = set()
            for pos, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)

                pred_id = int(torch.argmax(logits[pos]).item())
                conf = confs[pos, pred_id].item()

                # Prefer predictions from positions closer to centre
                centre = MAX_SEQ_LEN // 2
                centrality = 1.0 - abs(pos - centre) / centre
                score = conf + 0.1 * centrality

                if score > best[word_id][0]:
                    best[word_id] = (score, pred_id)

        predictions = [ID_TO_LABEL[label_id] for _, label_id in best]
        return self._fix_bio_consistency(predictions)

    # ------------------------------------------------------------------ #
    #  Subword → whitespace-token realignment                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _realign(tokens, word_ids, pred_ids):
        """Map subword predictions back to one tag per whitespace token."""
        n = len(tokens)
        tags = ["O"] * n
        seen = set()
        for pos, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            tags[word_id] = ID_TO_LABEL[pred_ids[pos]]

        return BertNERPredictor._fix_bio_consistency(tags)

    # ------------------------------------------------------------------ #
    #  BIO consistency post-processing                                     #
    # ------------------------------------------------------------------ #

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
