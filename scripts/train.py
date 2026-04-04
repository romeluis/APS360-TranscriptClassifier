"""CLI entry point for training.

Usage:
    python -m scripts.train
    python -m scripts.train --data data/output --epochs 15 --batch-size 48
"""

import argparse

from model.config import BATCH_SIZE, DATA_DIR, CHECKPOINT_DIR, LEARNING_RATE, NUM_EPOCHS, SEED
from model.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train the Transcript NER model")
    parser.add_argument("--data", default=DATA_DIR, help="Path to training data directory")
    parser.add_argument("--output", default=CHECKPOINT_DIR, help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--log-every", type=int, default=5, help="Log every N steps")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
