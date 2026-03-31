"""
Configuration constants for the BERT NER model.

Centralises label maps, hyperparameters, and paths so every other module
imports from here rather than hard-coding values.
"""

# ── Label scheme ──────────────────────────────────────────────────────────

LABEL_LIST = [
    "O",
    "B-CODE", "I-CODE",
    "B-NAME", "I-NAME",
    "B-GRADE", "I-GRADE",
    "B-SEM", "I-SEM",
]

NUM_LABELS = len(LABEL_LIST)
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# ── Pre-trained model ─────────────────────────────────────────────────────

MODEL_NAME = "microsoft/deberta-v3-base"  # was bert-base-cased; better NER performance

# ── Sequence handling ─────────────────────────────────────────────────────

MAX_SEQ_LEN = 512
STRIDE = 64           # smaller stride = more overlap, fewer semesters lost at window edges

# ── Training hyperparameters ──────────────────────────────────────────────

BATCH_SIZE = 48   # DeBERTa-v3-base on A100 (40GB): 48 fits well; drop to 16 for T4
LEARNING_RATE = 3e-5
NUM_EPOCHS = 15       # extra epochs for larger dataset; early stopping still triggers at patience=5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5

# ── Data split (by template) ─────────────────────────────────────────────

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
# TEST_RATIO is implicitly 1 - TRAIN - VAL = 0.20
SEED = 42

# ── Paths (relative to project root) ─────────────────────────────────────

DATA_DIR = "transcript_generator/output"
CHECKPOINT_DIR = "model/checkpoints"
BEST_CHECKPOINT_DIR = "model/checkpoints/best"
LOG_DIR = "model/logs"
