"""
TranscriptNERModel — DeBERTa encoder + linear emission layer + CRF decoder.

Replaces BertForTokenClassification with a proper sequence-labelling model
that enforces valid BIO transitions via Viterbi decoding.

Why CRF?
  The plain BERT + argmax approach predicts each token independently, allowing
  impossible sequences like B-NAME → B-GRADE → B-CODE within a single course row.
  The CRF transition matrix learns which label→label transitions are valid and
  penalises implausible sequences during training. At inference, Viterbi decoding
  finds the globally optimal label sequence for each window instead of greedy
  per-token argmax.

Why DeBERTa-v3-base?
  Disentangled attention separates content and position representations, which
  helps the model learn column-positional patterns (code is always left of name,
  grade is always right of name). It achieves ~93.4 F1 on CoNLL-2003 vs BERT's
  ~91.1, and loads identically via AutoModel/AutoTokenizer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

from model.config import MODEL_NAME, NUM_LABELS


class TranscriptNERModel(nn.Module):
    """DeBERTa encoder + dropout + linear emission + CRF decoder."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels:         (batch, seq_len) with -100 for ignored positions,
                            or None at inference time.

        Returns:
            Training:   (loss scalar, emissions tensor)
            Inference:  (list[list[int]] Viterbi paths, emissions tensor)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        # (batch, seq_len, num_labels)

        # Cast to fp32 before CRF: the Viterbi/forward-backward algorithm uses
        # log-sum-exp which is numerically sensitive and can produce NaN in bf16.
        # The encoder runs in bf16 (fast), only the CRF head needs fp32 (tiny cost).
        emissions_fp32 = emissions.float()

        # CRF requires a boolean mask over valid (non-padding) positions.
        crf_mask = attention_mask.bool()  # (batch, seq_len)

        if labels is not None:
            # ── Compact to first-subword positions only ───────────────
            # labels == -100 at CLS/SEP/padding/continuation-subword positions.
            # Training the CRF on ALL positions forces it to model transitions like
            # B-CODE → O(continuation) → I-CODE, which are invalid BIO sequences and
            # create conflicting gradients with the auxiliary CE loss.
            # Instead, gather only the positions where label != -100 (i.e. the first
            # subword of each whitespace token) into a contiguous compact tensor.
            # torchcrf requires mask[:, 0].all() — compact_mask satisfies this because
            # every sequence has at least one real token.
            first_subword_mask = (labels != -100)            # (B, L) bool
            real_lengths = first_subword_mask.sum(dim=1)     # (B,)
            max_real_len = int(real_lengths.max().item())

            batch_size = labels.size(0)
            compact_emissions = emissions_fp32.new_zeros(batch_size, max_real_len, self.num_labels)
            compact_labels    = labels.new_zeros(batch_size, max_real_len)
            compact_mask      = torch.zeros(batch_size, max_real_len,
                                            dtype=torch.bool, device=labels.device)

            for i in range(batch_size):
                idx = first_subword_mask[i].nonzero(as_tuple=True)[0]
                n = idx.size(0)
                if n > 0:
                    compact_emissions[i, :n] = emissions_fp32[i, idx]
                    compact_labels[i, :n]    = labels[i, idx]
                    compact_mask[i, :n]      = True

            num_real_tokens = compact_mask.sum().clamp(min=1)
            loss = -self.crf(compact_emissions, compact_labels,
                             mask=compact_mask, reduction="sum") / num_real_tokens
            # Viterbi decode on the same compact sequences (one prediction per
            # first-subword token); _decode_predictions_crf in train.py uses a
            # separate index to map these back to label positions.
            predictions = self.crf.decode(compact_emissions, mask=compact_mask)
            return loss, emissions_fp32, predictions

        else:
            # Inference only — Viterbi decode, no loss
            predictions = self.crf.decode(emissions_fp32, mask=crf_mask)
            return predictions, emissions_fp32

    def save_pretrained(self, save_dir: str):
        """Save encoder + full model state to directory."""
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        # Save the HuggingFace encoder (config + weights)
        self.encoder.save_pretrained(save_dir)
        # Save the full model state dict (includes classifier + CRF weights)
        torch.save(self.state_dict(), os.path.join(save_dir, "ner_model.pt"))
        # Save model hyperparams for reloading
        meta = {
            "model_name": MODEL_NAME,
            "num_labels": self.num_labels,
            "dropout": self.dropout.p,
        }
        with open(os.path.join(save_dir, "ner_model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_dir: str, device=None):
        """Reload a saved TranscriptNERModel from a checkpoint directory."""
        import os, json
        meta_path = os.path.join(load_dir, "ner_model_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        model = cls(
            model_name=load_dir,  # encoder weights are in load_dir
            num_labels=meta["num_labels"],
            dropout=meta.get("dropout", 0.1),
        )
        state = torch.load(
            os.path.join(load_dir, "ner_model.pt"),
            map_location=device or "cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        if device:
            model = model.to(device)
        return model


class FocalCRFLoss(nn.Module):
    """Focal-weighted CRF loss for extreme class imbalance.

    Combines two objectives:
    1. Standard CRF NLL loss (ensures valid BIO transitions via Viterbi)
    2. Focal weighting: scales the CRF loss by a factor derived from per-token
       emission confidence, so hard/uncertain tokens contribute more to the
       gradient than easy O-tokens the model already handles well.

    gamma: focal exponent (2.0 is standard; higher = more focus on hard tokens)
    alpha: optional class-weight tensor (same as weighted cross-entropy weights),
           applied on top of focal weighting for extra rare-class emphasis.
    """

    def __init__(self, crf: CRF, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.crf = crf
        self.gamma = gamma
        self.alpha = alpha  # (num_labels,) or None

    def forward(
        self,
        emissions: torch.Tensor,   # (batch, seq_len, num_labels)
        labels: torch.Tensor,      # (batch, seq_len), -100 for ignored
        mask: torch.Tensor,        # (batch, seq_len) bool, True = valid position
    ) -> torch.Tensor:
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0
        valid_mask = (labels != -100) & mask

        # ── Primary loss: CRF NLL ─────────────────────────────────────────
        crf_loss = -self.crf(emissions, crf_labels, mask=valid_mask, reduction="mean")

        # ── Focal scaling: computed from emission softmax probs ───────────
        with torch.no_grad():
            probs = torch.softmax(emissions, dim=-1)              # (B, L, C)
            p_t = probs.gather(-1, crf_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
            focal_weight = (1.0 - p_t) ** self.gamma             # (B, L)
            if self.alpha is not None:
                alpha_t = self.alpha[crf_labels]                  # (B, L)
                focal_weight = focal_weight * alpha_t
            # Average focal weight over valid positions only
            valid_focal = focal_weight[valid_mask]
            focal_scale = valid_focal.mean().clamp(min=0.1) if valid_focal.numel() > 0 else torch.tensor(1.0)

        return crf_loss * focal_scale
