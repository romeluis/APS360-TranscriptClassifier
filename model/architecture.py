"""
TranscriptNERModel -- DeBERTa encoder + linear emission layer + CRF decoder.

The CRF transition matrix learns valid BIO transitions and Viterbi decoding
finds the globally optimal label sequence instead of greedy per-token argmax.

DeBERTa-v3-base's disentangled attention separates content and position
representations, helping learn column-positional patterns in transcripts.
"""

import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from torchcrf import CRF
except ImportError:
    from TorchCRF import CRF

from .config import MODEL_NAME, NUM_LABELS


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
        Returns:
            Training:   (crf_loss, emissions_fp32, viterbi_paths)
            Inference:  (viterbi_paths, emissions_fp32)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))

        # CRF log-sum-exp is numerically sensitive -- always run in fp32.
        emissions_fp32 = emissions.float()
        crf_mask = attention_mask.bool()

        if labels is not None:
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0

            num_crf_tokens = crf_mask.sum().clamp(min=1)
            try:
                crf_loglik = self.crf(emissions_fp32, crf_labels, mask=crf_mask, reduction="sum")
            except TypeError:
                # TorchCRF uses (emissions, labels, mask) and returns per-sample scores.
                crf_loglik = self.crf(emissions_fp32, crf_labels, crf_mask)
                if isinstance(crf_loglik, torch.Tensor) and crf_loglik.ndim > 0:
                    crf_loglik = crf_loglik.sum()

            loss = -crf_loglik / num_crf_tokens

            if hasattr(self.crf, "decode"):
                predictions = self.crf.decode(emissions_fp32, mask=crf_mask)
            else:
                predictions = self.crf.viterbi_decode(emissions_fp32, crf_mask)
            return loss, emissions_fp32, predictions
        else:
            if hasattr(self.crf, "decode"):
                predictions = self.crf.decode(emissions_fp32, mask=crf_mask)
            else:
                predictions = self.crf.viterbi_decode(emissions_fp32, crf_mask)
            return predictions, emissions_fp32

    def save_pretrained(self, save_dir: str):
        """Save encoder + full model state to directory."""
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.save_pretrained(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, "ner_model.pt"))
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
        meta_path = os.path.join(load_dir, "ner_model_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        model = cls(
            model_name=load_dir,
            num_labels=meta["num_labels"],
            dropout=meta.get("dropout", 0.1),
        )
        state = torch.load(
            os.path.join(load_dir, "ner_model.pt"),
            map_location=device or "cpu",
            weights_only=True,
        )
        # Strip `_orig_mod.` prefix added by torch.compile()
        state = {k.replace("._orig_mod", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        if device:
            model = model.to(device)
        return model
