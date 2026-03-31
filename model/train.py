"""
Training loop for the TranscriptNER model (DeBERTa-v3-base + CRF).

Provides a `train()` function that handles the full pipeline:
data splitting → dataset creation → training with validation → checkpointing.

Key changes from the original BERT + cross-entropy baseline:
  - TranscriptNERModel (DeBERTa encoder + CRF) replaces BertForTokenClassification
  - FocalCRFLoss replaces weighted cross-entropy (better handling of O imbalance)
  - Cosine LR schedule replaces linear decay
  - AutoTokenizer replaces BertTokenizerFast (DeBERTa uses SentencePiece)

Can be run as a script or imported into a Colab notebook.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import f1_score as seqeval_f1
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from .config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_DIR,
    CHECKPOINT_DIR,
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    ID_TO_LABEL,
    LABEL_TO_ID,
    LEARNING_RATE,
    LOG_DIR,
    MAX_GRAD_NORM,
    MAX_SEQ_LEN,
    MODEL_NAME,
    NUM_EPOCHS,
    NUM_LABELS,
    SEED,
    STRIDE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from .data_split import save_split_info, split_by_template
from .dataset import TranscriptNERDataset
from .ner_model import FocalCRFLoss, TranscriptNERModel


# ── Helpers ───────────────────────────────────────────────────────────────


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_predictions_crf(crf_paths, labels):
    """Convert CRF Viterbi paths + labels into lists of BIO tag strings,
    ignoring positions where label == -100 (special tokens / padding).

    Args:
        crf_paths: list[list[int]] — Viterbi-decoded label IDs, one per batch item.
                   Length of each inner list equals seq_len (including padding positions).
        labels:    (B, L) tensor with -100 for ignored positions.

    Returns:
        (true_batch, pred_batch): lists of lists of BIO strings.
    """
    true_batch, pred_batch = [], []
    for i, path in enumerate(crf_paths):
        true_seq, pred_seq = [], []
        for j in range(labels.size(1)):
            if labels[i, j].item() != -100:
                true_seq.append(ID_TO_LABEL[labels[i, j].item()])
                # CRF path may be shorter than seq_len if mask cuts it;
                # guard with a bounds check.
                pred_label = ID_TO_LABEL[path[j]] if j < len(path) else "O"
                pred_seq.append(pred_label)
        true_batch.append(true_seq)
        pred_batch.append(pred_seq)
    return true_batch, pred_batch


# ── Main training function ────────────────────────────────────────────────


def train(
    data_dir: str = DATA_DIR,
    output_dir: str = CHECKPOINT_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    warmup_ratio: float = WARMUP_RATIO,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
    use_fp16: bool = True,
    log_every: int = 5,
) -> dict:
    """Full training pipeline. Returns training history dict."""

    _set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # DeBERTa-v3 has internal FP16 parameters that break GradScaler.
    # On A100/H100 use bfloat16 instead — same dynamic range as fp32, no scaling needed.
    # On older GPUs (T4, V100) fall back to fp32 (use_fp16=False).
    use_fp16 = False  # disabled; we use bf16 autocast below
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}  |  bf16: {use_bf16}")

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading and splitting data...")
    train_samples, val_samples, test_samples, split_info = split_by_template(
        data_dir=data_dir, seed=seed,
    )
    save_split_info(split_info, str(Path(output_dir) / "split_info.json"))
    print(
        f"  Train: {split_info['train_samples']} samples "
        f"({len(split_info['train_templates'])} templates)"
    )
    print(
        f"  Val:   {split_info['val_samples']} samples "
        f"({len(split_info['val_templates'])} templates)"
    )
    print(
        f"  Test:  {split_info['test_samples']} samples "
        f"({len(split_info['test_templates'])} templates)"
    )

    # AutoTokenizer handles both BERT (WordPiece) and DeBERTa (SentencePiece)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenising datasets...")
    train_ds = TranscriptNERDataset(train_samples, tokenizer, MAX_SEQ_LEN, STRIDE)
    val_ds = TranscriptNERDataset(val_samples, tokenizer, MAX_SEQ_LEN, STRIDE)
    print(f"  Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ── Class weights (inverse frequency) ────────────────────────────
    # BIO labels are heavily imbalanced: O is ~70% of tokens while
    # B-SEM is <1%. Upweighting rare entity classes improves recall.
    print("Computing class weights...")
    label_counts = np.zeros(NUM_LABELS, dtype=np.float64)
    for batch in train_loader:
        labels_flat = batch["labels"].numpy().flatten()
        for label_id in labels_flat:
            if label_id != -100:
                label_counts[label_id] += 1
    total = label_counts.sum()
    label_counts = np.maximum(label_counts, 1.0)
    class_weights = total / (NUM_LABELS * label_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("  Label counts:", {ID_TO_LABEL[i]: int(label_counts[i]) for i in range(NUM_LABELS)})
    print("  Class weights:", {ID_TO_LABEL[i]: f"{class_weights[i]:.2f}" for i in range(NUM_LABELS)})

    # ── Model ─────────────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME} + CRF...")
    model = TranscriptNERModel(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        dropout=0.1,
    )
    model.to(device)

    # FocalCRFLoss: CRF NLL scaled by focal weighting on emission probabilities.
    # This combines the sequence-constraint benefit of CRF with focal loss's
    # down-weighting of easy O tokens, focusing training on hard entity boundaries.
    loss_fn = FocalCRFLoss(
        crf=model.crf,
        gamma=2.0,
        alpha=class_weights_tensor,
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    # Cosine schedule: smoother decay than linear, better late-training convergence
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
    )

    # GradScaler is only needed for fp16. bf16 has fp32 dynamic range → no scaling.
    scaler = torch.amp.GradScaler(enabled=False)

    # ── Training loop ─────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_token_acc": [],
        "val_entity_f1": [],
    }

    best_f1 = -1.0
    patience_counter = 0
    best_dir = str(Path(output_dir) / "best")

    print(f"\nTraining for {num_epochs} epochs ({total_steps} steps)...\n", flush=True)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ── Train phase ───────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
                # TranscriptNERModel.forward() returns (loss, emissions) in train mode
                loss, emissions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # Apply focal CRF loss using the same emissions
                loss = loss_fn(emissions, labels, attention_mask.bool())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            running_loss += loss.item()
            step_count += 1

            if step % log_every == 0 or step == 1:
                avg = running_loss / step_count
                print(f"  Epoch {epoch} step {step}/{len(train_loader)}  loss={avg:.4f}", flush=True)

        train_loss = running_loss / step_count
        history["train_loss"].append(train_loss)

        # ── Validation phase ──────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        all_true, all_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
                    # Inference mode: returns (Viterbi paths, emissions)
                    crf_paths, emissions = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    val_loss = loss_fn(emissions, labels, attention_mask.bool())

                val_loss_sum += val_loss.item()
                val_steps += 1

                true_batch, pred_batch = _decode_predictions_crf(crf_paths, labels)
                all_true.extend(true_batch)
                all_pred.extend(pred_batch)

        val_loss = val_loss_sum / val_steps
        history["val_loss"].append(val_loss)

        # Token accuracy
        correct = sum(
            t == p for ts, ps in zip(all_true, all_pred) for t, p in zip(ts, ps)
        )
        total_toks = sum(len(ts) for ts in all_true)
        val_acc = correct / total_toks if total_toks else 0.0
        history["val_token_acc"].append(val_acc)

        # Entity F1 (seqeval)
        val_f1 = seqeval_f1(all_true, all_pred)
        history["val_entity_f1"].append(val_f1)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        # ── Checkpointing ────────────────────────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            Path(best_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  ↑ New best model saved (F1={best_f1:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"  No improvement (patience {patience_counter}/{EARLY_STOPPING_PATIENCE})", flush=True)
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered (patience={EARLY_STOPPING_PATIENCE})", flush=True)
                break

    # ── Save history ──────────────────────────────────────────────────
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Checkpoint: {best_dir}")
    print(f"History:    {log_dir / 'history.json'}")

    return history


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
