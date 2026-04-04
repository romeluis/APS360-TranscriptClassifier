"""CLI entry point for transcript generation.

Usage:
    python -m scripts.generate --output data/output --count 200 --preprocess-md
"""

import argparse

from transcript_generator.config import GenerationConfig
from transcript_generator.generator import TranscriptGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transcripts")
    parser.add_argument("--output", default="data/output", help="Output directory")
    parser.add_argument("--count", type=int, default=200, help="Transcripts per template")
    parser.add_argument("--preprocess-md", action="store_true", help="Convert PDFs to markdown using selected backend")
    parser.add_argument("--html-to-md", action="store_true", help="Convert HTML directly to markdown (fast)")
    parser.add_argument(
        "--preprocess-backend",
        choices=["marker", "opendataloader", "docling"],
        default="docling",
        help="Backend package used when --preprocess-md is enabled",
    )
    parser.add_argument(
        "--preprocess-use-llm",
        action="store_true",
        default=True,
        help="Enable marker LLM-enhanced processing",
    )
    parser.add_argument(
        "--no-preprocess-use-llm",
        action="store_false",
        dest="preprocess_use_llm",
        help="Disable marker LLM-enhanced processing",
    )
    parser.add_argument(
        "--preprocess-batch-multiplier",
        type=int,
        default=2,
        help="Marker batch multiplier for preprocess-md",
    )
    parser.add_argument(
        "--preprocess-disable-table-recognition",
        action="store_true",
        help="Disable marker table recognition processors",
    )
    parser.add_argument(
        "--preprocess-opendataloader-hybrid",
        action="store_true",
        help="Enable OpenDataLoader hybrid mode",
    )
    parser.add_argument(
        "--preprocess-opendataloader-hybrid-mode",
        choices=["auto", "full"],
        default="auto",
        help="OpenDataLoader hybrid triage mode",
    )
    parser.add_argument(
        "--preprocess-opendataloader-hybrid-url",
        default="",
        help="OpenDataLoader hybrid server URL",
    )
    parser.add_argument(
        "--preprocess-opendataloader-hybrid-timeout",
        default="",
        help="OpenDataLoader hybrid timeout in milliseconds",
    )
    parser.add_argument(
        "--preprocess-opendataloader-hybrid-fallback",
        action="store_true",
        help="Enable OpenDataLoader Java fallback on hybrid errors",
    )
    parser.add_argument("--render-pdf", action="store_true", default=True, help="Render PDFs")
    parser.add_argument("--no-render-pdf", action="store_false", dest="render_pdf")
    parser.add_argument("--augment-noise", action="store_true", help="Apply noise augmentation")
    parser.add_argument("--noise-intensity", type=float, default=1.0, help="Noise intensity (0-1)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (0=auto)")
    parser.add_argument("--split-courses", action="store_true", help="Disjoint train/test course pools")
    parser.add_argument("--templates", nargs="*", help="Specific template names to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = GenerationConfig(
        count_per_template=args.count,
        render_pdf=args.render_pdf or args.preprocess_md,
        pdf_extract=False,
        preprocess_md=args.preprocess_md,
        html_to_md=args.html_to_md,
        augment_noise=args.augment_noise,
        noise_intensity=args.noise_intensity,
        workers=args.workers,
        split_courses=args.split_courses,
        template_names=args.templates,
        seed=args.seed,
        preprocess_backend=args.preprocess_backend,
        preprocess_use_llm=args.preprocess_use_llm,
        preprocess_batch_multiplier=args.preprocess_batch_multiplier,
        preprocess_disable_table_recognition=args.preprocess_disable_table_recognition,
        preprocess_opendataloader_hybrid=args.preprocess_opendataloader_hybrid,
        preprocess_opendataloader_hybrid_mode=args.preprocess_opendataloader_hybrid_mode,
        preprocess_opendataloader_hybrid_url=args.preprocess_opendataloader_hybrid_url.strip() or None,
        preprocess_opendataloader_hybrid_timeout=args.preprocess_opendataloader_hybrid_timeout.strip() or None,
        preprocess_opendataloader_hybrid_fallback=args.preprocess_opendataloader_hybrid_fallback,
    )

    gen = TranscriptGenerator(args.output)
    manifest = gen.generate(config)
    print(f"\nGenerated {manifest['total_transcripts']} transcripts.")


if __name__ == "__main__":
    main()
