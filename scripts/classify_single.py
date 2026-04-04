"""CLI entry point for single-PDF classification.

Usage:
    python -m scripts.classify_single --pdf transcript.pdf --output transcript.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.config import BEST_CHECKPOINT_DIR
from pipeline.classifier import TranscriptClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single PDF transcript")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument(
        "--checkpoint",
        default=BEST_CHECKPOINT_DIR,
        help="Model checkpoint directory",
    )
    parser.add_argument("--device", default="", help="Torch device override")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--backend",
        choices=["marker", "opendataloader", "docling"],
        default="docling",
        help="PDF-to-Markdown backend package",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Enable marker LLM-enhanced processing",
    )
    parser.add_argument(
        "--no-use-llm",
        action="store_false",
        dest="use_llm",
        help="Disable marker LLM-enhanced processing",
    )
    parser.add_argument(
        "--opendataloader-hybrid",
        action="store_true",
        help="Enable OpenDataLoader hybrid mode (docling-fast backend)",
    )
    parser.add_argument(
        "--opendataloader-hybrid-mode",
        choices=["auto", "full"],
        default="auto",
        help="OpenDataLoader hybrid triage mode",
    )
    parser.add_argument(
        "--opendataloader-hybrid-url",
        default="",
        help="OpenDataLoader hybrid server URL",
    )
    parser.add_argument(
        "--opendataloader-hybrid-timeout",
        default="",
        help="OpenDataLoader hybrid timeout in milliseconds",
    )
    parser.add_argument(
        "--opendataloader-hybrid-fallback",
        action="store_true",
        help="Enable OpenDataLoader Java fallback on hybrid errors",
    )
    args = parser.parse_args()

    classifier = TranscriptClassifier(
        args.checkpoint,
        device=args.device.strip() or None,
        use_llm=args.use_llm,
        backend=args.backend,
        opendataloader_hybrid=args.opendataloader_hybrid,
        opendataloader_hybrid_mode=args.opendataloader_hybrid_mode,
        opendataloader_hybrid_url=args.opendataloader_hybrid_url.strip() or None,
        opendataloader_hybrid_timeout=args.opendataloader_hybrid_timeout.strip() or None,
        opendataloader_hybrid_fallback=args.opendataloader_hybrid_fallback,
    )
    result = classifier.classify_pdf(args.pdf)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved classification to {output_path}")


if __name__ == "__main__":
    main()
