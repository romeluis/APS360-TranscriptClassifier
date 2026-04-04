"""CLI entry point for single-PDF preprocessing.

Usage:
    python -m scripts.preprocess_single --pdf transcript.pdf --output transcript.md
"""

from __future__ import annotations

import argparse

from preprocessor import TranscriptPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a single PDF into markdown")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output markdown path")
    parser.add_argument(
        "--backend",
        choices=["marker", "opendataloader", "docling"],
        default="docling",
        help="PDF-to-Markdown backend package",
    )
    parser.add_argument("--batch-multiplier", type=int, default=2, help="Marker batch multiplier")
    parser.add_argument(
        "--disable-table-recognition",
        action="store_true",
        help="Disable marker table recognition processors",
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

    preprocessor = TranscriptPreprocessor(
        backend=args.backend,
        batch_multiplier=args.batch_multiplier,
        disable_table_recognition=args.disable_table_recognition,
        use_llm=args.use_llm,
        opendataloader_hybrid=args.opendataloader_hybrid,
        opendataloader_hybrid_mode=args.opendataloader_hybrid_mode,
        opendataloader_hybrid_url=args.opendataloader_hybrid_url.strip() or None,
        opendataloader_hybrid_timeout=args.opendataloader_hybrid_timeout.strip() or None,
        opendataloader_hybrid_fallback=args.opendataloader_hybrid_fallback,
    )
    preprocessor.convert(args.pdf, args.output)
    print(f"Saved markdown to {args.output}")


if __name__ == "__main__":
    main()
