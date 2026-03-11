"""Extract text and BIO labels from assembled HTML with entity-annotated spans."""

import re
from bs4 import BeautifulSoup, Comment, Doctype, NavigableString


def _find_running_classes(html_string):
    """Find CSS class names that use position: running(...) — these are
    repeated header/footer elements and should be excluded from text extraction."""
    running_classes = set()
    for match in re.finditer(
        r"\.([a-zA-Z][\w-]*)\s*\{[^}]*position\s*:\s*running\s*\([^)]+\)",
        html_string,
    ):
        running_classes.add(match.group(1))
    return running_classes


def _is_running_element(node, running_classes):
    """Check if a BeautifulSoup node has a class that uses CSS running positioning."""
    if not running_classes:
        return False
    classes = node.get("class", []) if hasattr(node, "get") else []
    return bool(set(classes) & running_classes)


# Block-level elements that introduce whitespace separation
_BLOCK_ELEMENTS = frozenset({
    "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "td", "th", "li", "br", "hr", "table", "section",
    "header", "footer", "article", "main", "nav", "aside",
    "ul", "ol", "dl", "dt", "dd", "blockquote", "pre",
    "thead", "tbody", "tfoot", "caption",
})


def extract_text_and_annotations(html_string):
    """
    Walk the assembled HTML tree and extract visible text with character-level
    entity annotations.

    Returns:
        text: str — the full visible text
        annotations: list of (start, end, entity_type) tuples
    """
    soup = BeautifulSoup(html_string, "html.parser")
    running_classes = _find_running_classes(html_string)

    # Remove non-visible elements
    for tag in soup.find_all("style"):
        tag.decompose()
    for tag in soup.find_all("script"):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    text_parts = []
    annotations = []
    current_offset = [0]  # mutable for nested access

    def _add_space():
        """Add a space separator if the last character isn't already a space."""
        if text_parts and text_parts[-1] and not text_parts[-1].endswith(" "):
            text_parts.append(" ")
            current_offset[0] += 1

    def walk(node, active_entity=None):
        if isinstance(node, Doctype):
            return
        if isinstance(node, NavigableString):
            text = str(node)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text)
            if not text.strip():
                _add_space()
                return

            start = current_offset[0]
            text_parts.append(text)
            current_offset[0] += len(text)

            if active_entity:
                annotations.append((start, current_offset[0], active_entity))
            return

        # Skip invisible/meta elements
        if getattr(node, "name", None) in ("style", "script", "meta", "link", "head"):
            return

        # Skip running header/footer elements (they repeat per page)
        if hasattr(node, "get") and _is_running_element(node, running_classes):
            return

        # Check for entity annotation
        entity = node.get("data-entity") if hasattr(node, "get") else None
        new_entity = entity or active_entity

        # Block elements add whitespace before
        if getattr(node, "name", None) in _BLOCK_ELEMENTS:
            _add_space()

        for child in node.children:
            walk(child, new_entity)

        # Block elements add whitespace after
        if getattr(node, "name", None) in _BLOCK_ELEMENTS:
            _add_space()

    walk(soup)
    full_text = "".join(text_parts).strip()
    return full_text, annotations


def text_to_bio_labels(text, annotations):
    """
    Convert extracted text + character-level annotations into token-level BIO labels.

    Uses whitespace tokenization (suitable for BERT pre-tokenization).

    Returns:
        tokens: list[str]
        labels: list[str]
    """
    # Build character-to-entity map
    char_entity = [None] * len(text)
    for start, end, entity_type in annotations:
        for i in range(start, min(end, len(text))):
            char_entity[i] = entity_type

    tokens = []
    labels = []

    for word_match in re.finditer(r"\S+", text):
        word = word_match.group()
        word_start = word_match.start()
        word_end = word_match.end()

        # Count entity types for this token's characters
        entity_counts = {}
        for i in range(word_start, word_end):
            e = char_entity[i]
            if e is not None:
                entity_counts[e] = entity_counts.get(e, 0) + 1

        if not entity_counts:
            label = "O"
        else:
            # Pick the entity type with the most characters in this token
            entity_type = max(entity_counts, key=entity_counts.get)

            # Determine B- vs I-: B if this is the start of a new entity span
            if labels and labels[-1] in (f"B-{entity_type}", f"I-{entity_type}"):
                # Check if there's a gap between this token and the previous one
                # by looking at the character positions
                label = f"I-{entity_type}"
            else:
                label = f"B-{entity_type}"

        tokens.append(word)
        labels.append(label)

    return tokens, labels


def extract_labels(html_string):
    """
    High-level function: extract tokens and BIO labels from assembled HTML.

    Returns:
        tokens: list[str]
        labels: list[str]
    """
    text, annotations = extract_text_and_annotations(html_string)
    tokens, labels = text_to_bio_labels(text, annotations)
    return tokens, labels


def validate_bio_labels(tokens, labels):
    """
    Validate BIO label consistency.

    Returns:
        errors: list of error message strings (empty if valid)
    """
    errors = []
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("I-"):
            entity_type = label[2:]
            if i == 0:
                errors.append(f"Token {i} '{token}' has {label} at start of sequence")
            elif labels[i - 1] not in (f"B-{entity_type}", f"I-{entity_type}"):
                errors.append(
                    f"Token {i} '{token}' has {label} but previous label is {labels[i-1]}"
                )
    return errors
