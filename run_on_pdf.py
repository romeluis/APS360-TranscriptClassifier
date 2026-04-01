"""
Run the trained BERT NER model on a real transcript PDF.

Usage:
    python run_on_pdf.py path/to/transcript.pdf

The script:
  1. Extracts text from the PDF (via pypdf)
  2. Whitespace-tokenizes it
  3. Runs BertNERPredictor (model/checkpoints/best/)
  4. Reconstructs semester/course structure
  5. Renders a terminal table — the same view the iOS app would show

No ground-truth labels are needed; this works on any unseen PDF.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

# ── Ensure project root is on sys.path so model/evaluation imports work ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pypdf import PdfReader
from model.predict import BertNERPredictor
from evaluation.reconstructor import reconstruct_semesters

try:
    import pdfplumber
    _PDFPLUMBER = True
except ImportError:
    _PDFPLUMBER = False

# ANSI colours (same as visualize.py)
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
DIM   = "\033[2m"
RESET = "\033[0m"

try:
    from tabulate import tabulate
    _TABULATE = True
except ImportError:
    _TABULATE = False


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize extracted PDF text before tokenization.

    1. NFC unicode normalization: decomposes ligatures (ﬁ→fi, ﬀ→ff) and
       normalizes composed characters to their canonical form so the tokenizer
       never sees rare unicode codepoints absent from pretraining.
    2. Strip non-printable / control characters (except \\n and \\t): PDF
       extraction can embed form-feed, null bytes, or other control chars
       that split tokens unexpectedly.
    3. Collapse runs of multiple spaces on the same line (preserve newlines
       as structural separators between semester/course rows).
    """
    # NFC normalization decomposes ligatures and normalises accented chars
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except tab and newline
    text = re.sub(r"[^\x09\x0a\x20-\x7e\x80-\xff]", " ", text)
    # Collapse multiple spaces per line; preserve newlines
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using pdfplumber (preferred) or pypdf (fallback).

    pdfplumber uses character bounding boxes to reconstruct layout-aware text
    with proper line breaks, which preserves the tabular row structure of
    academic transcripts better than pypdf's character-order concatenation.
    """
    text = None

    if _PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if page_text:
                        pages.append(page_text)
            if pages:
                text = "\n".join(pages)
        except Exception as e:
            print(f"  pdfplumber failed ({e}); falling back to pypdf")
            text = None

    if text is None:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        text = "\n".join(pages)

    return _normalize_text(text)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_semesters(semesters: list[dict]):
    """Print extracted semester/course tables to the terminal."""
    if not semesters:
        print(f"  {DIM}(no semesters detected){RESET}")
        return

    for sem in semesters:
        print(f"\n  {CYAN}{BOLD}=== {sem['semester_name']} ==={RESET}")
        courses = sem["courses"]
        if not courses:
            print(f"    {DIM}(no courses){RESET}")
            continue

        rows = [
            [
                c["code"]  or f"{DIM}—{RESET}",
                c["name"]  or f"{DIM}—{RESET}",
                c["grade"] or f"{DIM}—{RESET}",
            ]
            for c in courses
        ]

        if _TABULATE:
            print(tabulate(rows, headers=["Code", "Course Name", "Grade"],
                           tablefmt="simple", stralign="left"))
        else:
            print(f"  {'Code':<12} {'Course Name':<40} {'Grade'}")
            print(f"  {'-'*12} {'-'*40} {'-'*5}")
            for row in rows:
                print(f"  {row[0]:<12} {row[1]:<40} {row[2]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run BERT NER model on a transcript PDF and display results."
    )
    parser.add_argument("pdf", help="Path to the transcript PDF")
    parser.add_argument(
        "--checkpoint", default=str(PROJECT_ROOT / "model/checkpoints/best"),
        help="Path to BERT checkpoint directory (default: model/checkpoints/best)",
    )
    parser.add_argument(
        "--show-tokens", action="store_true",
        help="Also print raw tokens with predicted tags (for debugging)",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: file not found — {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Extract text
    print(f"{BOLD}Loading PDF...{RESET}  {pdf_path.name}")
    raw_text = extract_text_from_pdf(str(pdf_path))
    tokens = raw_text.split()
    print(f"  Extracted {len(tokens)} tokens from {len(PdfReader(str(pdf_path)).pages)} page(s)\n")

    # 2. Load model
    print(f"{BOLD}Loading model...{RESET}  {args.checkpoint}")
    predictor = BertNERPredictor(checkpoint_dir=args.checkpoint)
    print(f"  Model ready on {predictor.device}\n")

    # 3. Run inference
    print(f"{BOLD}Running inference...{RESET}")
    tags = predictor.predict(tokens)
    entity_counts = {t: tags.count(t) for t in set(tags) if t != "O"}
    print(f"  Entities found: {entity_counts}\n")

    # 4. (Optional) show raw token/tag pairs
    if args.show_tokens:
        print(f"{BOLD}Token → Tag (non-O only):{RESET}")
        for tok, tag in zip(tokens, tags):
            if tag != "O":
                print(f"  {tok:<30} {GREEN}{tag}{RESET}")
        print()

    # 5. Reconstruct and display
    semesters = reconstruct_semesters(tokens, tags)
    total_courses = sum(len(s["courses"]) for s in semesters)

    print(f"{BOLD}Transcript Visualizer{RESET}")
    print("=" * 60)
    print(f"File:     {pdf_path.name}")
    print(f"Tokens:   {len(tokens)}")
    print(f"Semesters found: {len(semesters)}")
    print(f"Courses found:   {total_courses}")

    render_semesters(semesters)

    print(f"\n{'=' * 60}")
    print(f"{GREEN}{BOLD}Done.{RESET}")


if __name__ == "__main__":
    main()
