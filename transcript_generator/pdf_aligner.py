"""Align entity annotations from clean HTML text onto noisy PDF-extracted text.

The transcript generator knows the ground-truth entities (via data-entity spans
in the assembled HTML).  This module transfers those annotations to the text
that pypdf extracts from the rendered PDF, so the model trains on the same kind
of noisy text it will see at inference time.

Algorithm:
  1. Extract clean text + char-level annotations from HTML  (label_extractor)
  2. Extract raw text from the rendered PDF                 (pypdf)
  3. Character-level sequence alignment  (difflib.SequenceMatcher)
  4. Transfer entity annotations across the alignment
  5. Tokenise + assign BIO labels on the PDF text           (label_extractor)
"""

import difflib

from pypdf import PdfReader

from label_extractor import extract_text_and_annotations, text_to_bio_labels


# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------

def _build_char_entity_map(text, annotations):
    """Build a character-level entity array from (start, end, entity_type) annotations."""
    char_entity = [None] * len(text)
    for start, end, entity_type in annotations:
        for i in range(start, min(end, len(text))):
            char_entity[i] = entity_type
    return char_entity


def align_pdf_to_html(clean_text, annotations, pdf_text):
    """Transfer character-level entity annotations from clean HTML text to PDF text.

    Uses difflib.SequenceMatcher to find matching character blocks between the
    two texts, then copies entity assignments from clean positions to the
    corresponding PDF positions.

    Args:
        clean_text: visible text extracted from HTML (via extract_text_and_annotations)
        annotations: list of (start, end, entity_type) from HTML extraction
        pdf_text: raw text extracted from the rendered PDF via pypdf

    Returns:
        pdf_annotations: list of (start, end, entity_type) tuples on pdf_text
        coverage: float — fraction of entity characters successfully transferred
    """
    clean_entities = _build_char_entity_map(clean_text, annotations)

    # Track which PDF characters inherit an entity
    pdf_entities = [None] * len(pdf_text)

    # Character-level alignment — autojunk=False is critical so that common
    # characters (digits, single letters like grade "B") aren't discarded.
    matcher = difflib.SequenceMatcher(None, clean_text, pdf_text, autojunk=False)

    for clean_start, pdf_start, length in matcher.get_matching_blocks():
        for i in range(length):
            entity = clean_entities[clean_start + i]
            if entity is not None:
                pdf_entities[pdf_start + i] = entity

    # Convert the per-character entity array into (start, end, entity_type) spans
    pdf_annotations = []
    current_entity = None
    span_start = 0

    for i, entity in enumerate(pdf_entities):
        if entity != current_entity:
            if current_entity is not None:
                pdf_annotations.append((span_start, i, current_entity))
            current_entity = entity
            span_start = i
    if current_entity is not None:
        pdf_annotations.append((span_start, len(pdf_entities), current_entity))

    # Compute coverage: what fraction of entity characters in clean text
    # were successfully transferred to the PDF text?
    total_entity_chars = sum(1 for e in clean_entities if e is not None)
    transferred_chars = sum(1 for e in pdf_entities if e is not None)
    coverage = transferred_chars / total_entity_chars if total_entity_chars > 0 else 1.0

    return pdf_annotations, coverage


# ---------------------------------------------------------------------------
# PDF text extraction  (mirrors run_on_pdf.py)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF, page by page."""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

_FALLBACK_COVERAGE = 0.50   # below this, fall back to HTML extraction


def extract_pdf_labels(html_string, pdf_path):
    """Extract BIO-labelled tokens from a rendered PDF, aligned with HTML annotations.

    Args:
        html_string: assembled HTML with data-entity spans
        pdf_path: path to the rendered PDF file

    Returns:
        tokens: list[str]   — whitespace-tokenised PDF text
        labels: list[str]   — BIO labels aligned to those tokens
        stats:  dict        — alignment diagnostics
            - extraction_mode: "pdf" or "html_fallback"
            - alignment_coverage: float (0–1)
            - pdf_token_count: int
            - html_token_count: int
    """
    # Step 1: clean text + char annotations from HTML
    clean_text, annotations = extract_text_and_annotations(html_string)

    # Step 2: raw text from the rendered PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 3–4: align and transfer annotations
    pdf_annotations, coverage = align_pdf_to_html(clean_text, annotations, pdf_text)

    # Step 5: tokenise PDF text and assign BIO labels
    tokens, labels = text_to_bio_labels(pdf_text, pdf_annotations)

    # For comparison, count HTML tokens
    from label_extractor import extract_labels
    html_tokens, _ = extract_labels(html_string)

    stats = {
        "extraction_mode": "pdf",
        "alignment_coverage": round(coverage, 4),
        "pdf_token_count": len(tokens),
        "html_token_count": len(html_tokens),
    }

    # Quality gate: fall back to HTML extraction if coverage is too low
    if coverage < _FALLBACK_COVERAGE:
        html_tokens, html_labels = extract_labels(html_string)
        stats["extraction_mode"] = "html_fallback"
        return html_tokens, html_labels, stats

    return tokens, labels, stats
