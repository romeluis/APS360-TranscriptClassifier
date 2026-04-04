"""Convert assembled transcript HTML directly to Markdown.

This is a fast alternative to running marker-pdf on rendered PDFs.
Since the HTML source is available at generation time, we can produce
clean Markdown instantly — no ML models, no PDF rendering needed.

The output mimics the structure marker would produce: pipe-delimited
tables, ``#`` headings, ``**bold**`` text, and paragraph breaks.
"""

from __future__ import annotations

import re
from bs4 import BeautifulSoup, Comment, Doctype, NavigableString, Tag


def _find_running_classes(html_string: str) -> set[str]:
    """Find CSS class names using ``position: running(...)`` — repeated
    header/footer elements that should be excluded."""
    running_classes: set[str] = set()
    for match in re.finditer(
        r"\.([a-zA-Z][\w-]*)\s*\{[^}]*position\s*:\s*running\s*\([^)]+\)",
        html_string,
    ):
        running_classes.add(match.group(1))
    return running_classes


def _is_running_element(node: Tag, running_classes: set[str]) -> bool:
    if not running_classes:
        return False
    classes = node.get("class", []) if hasattr(node, "get") else []
    return bool(set(classes) & running_classes)


# Block-level elements that introduce line breaks in output
_BLOCK_ELEMENTS = frozenset({
    "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "br", "hr", "section",
    "header", "footer", "article", "main", "nav", "aside",
    "ul", "ol", "dl", "dt", "dd", "blockquote", "pre",
})

_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _get_text(node) -> str:
    """Extract all visible text from a node, collapsing whitespace."""
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    parts = []
    for child in node.children:
        parts.append(_get_text(child))
    return " ".join(parts)


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _convert_table(table: Tag) -> str:
    """Convert an HTML <table> to a pipe-delimited Markdown table."""
    rows: list[list[str]] = []

    # Collect header rows from <thead>
    thead = table.find("thead")
    header_rows: list[list[str]] = []
    if thead:
        for tr in thead.find_all("tr", recursive=False):
            cells = [_clean_text(_get_text(cell)) for cell in tr.find_all(["th", "td"])]
            if any(cells):
                header_rows.append(cells)

    # Collect body rows from <tbody> or direct <tr> children
    body_rows: list[list[str]] = []
    tbody = table.find("tbody")
    body_source = tbody if tbody else table
    for tr in body_source.find_all("tr", recursive=False):
        # Skip rows already captured in thead
        if thead and tr.parent == thead:
            continue
        cells = [_clean_text(_get_text(cell)) for cell in tr.find_all(["th", "td"])]
        if any(cells):
            body_rows.append(cells)

    # If no thead, treat first row as header if it looks like one
    if not header_rows and body_rows:
        header_rows = [body_rows.pop(0)]

    if not header_rows:
        return ""

    rows = header_rows + body_rows

    # Determine column count and widths
    n_cols = max(len(r) for r in rows)
    # Pad short rows
    for r in rows:
        while len(r) < n_cols:
            r.append("")

    # Calculate column widths (min 3 for separator)
    col_widths = [3] * n_cols
    for r in rows:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(cell))

    def _format_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(f" {cell:<{col_widths[i]}} ")
        return "|" + "|".join(parts) + "|"

    lines: list[str] = []

    # Header row(s)
    for hr in header_rows:
        lines.append(_format_row(hr))

    # Separator
    sep_parts = ["-" * (w + 2) for w in col_widths]
    lines.append("|" + "|".join(sep_parts) + "|")

    # Body rows
    for br in body_rows:
        lines.append(_format_row(br))

    return "\n".join(lines)


def html_to_markdown(html_string: str) -> str:
    """Convert assembled transcript HTML to Markdown.

    Args:
        html_string: The assembled HTML (with ``data-entity`` spans, CSS, etc.)

    Returns:
        Markdown text resembling marker-pdf output.
    """
    running_classes = _find_running_classes(html_string)
    soup = BeautifulSoup(html_string, "html.parser")

    # Remove non-visible elements
    for tag in soup.find_all("style"):
        tag.decompose()
    for tag in soup.find_all("script"):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    output_lines: list[str] = []

    def _ensure_space():
        """Ensure there's a space at the end of current output (for inline separation)."""
        if output_lines and output_lines[-1] and not output_lines[-1][-1] in (" ", "\n"):
            output_lines.append(" ")

    def _walk(node):
        if isinstance(node, Doctype):
            return

        if isinstance(node, NavigableString):
            text = re.sub(r"\s+", " ", str(node))
            if text.strip():
                output_lines.append(text)
            elif text and output_lines:
                # Whitespace-only text node — preserve as single space
                _ensure_space()
            return

        if not isinstance(node, Tag):
            return

        tag_name = node.name or ""

        # Skip invisible/meta elements
        if tag_name in ("style", "script", "meta", "link", "head", "title"):
            return

        # Skip running header/footer elements
        if _is_running_element(node, running_classes):
            return

        # Handle tables
        if tag_name == "table":
            table_md = _convert_table(node)
            if table_md:
                output_lines.append("\n")
                output_lines.append(table_md)
                output_lines.append("\n")
            return

        # Handle headings
        if tag_name in _HEADING_LEVELS:
            level = _HEADING_LEVELS[tag_name]
            text = _clean_text(_get_text(node))
            if text:
                prefix = "#" * level
                output_lines.append("\n")
                output_lines.append(f"{prefix} {text}")
                output_lines.append("\n")
            return

        # Handle <strong>/<b>
        if tag_name in ("strong", "b"):
            text = _clean_text(_get_text(node))
            if text:
                output_lines.append(f"**{text}**")
            return

        # Handle <em>/<i>
        if tag_name in ("em", "i"):
            text = _clean_text(_get_text(node))
            if text:
                output_lines.append(f"*{text}*")
            return

        # Handle <br>
        if tag_name == "br":
            output_lines.append("\n")
            return

        # Handle <hr>
        if tag_name == "hr":
            output_lines.append("\n---\n")
            return

        # Block elements: add line break before/after
        is_block = tag_name in _BLOCK_ELEMENTS

        # Inline elements like <span>: add a space before if adjacent to
        # other inline content (prevents "codeNameGrade" concatenation)
        is_inline_container = tag_name in ("span", "a", "label")

        if is_block:
            output_lines.append("\n")
        elif is_inline_container:
            _ensure_space()

        for child in node.children:
            _walk(child)

        if is_block:
            output_lines.append("\n")
        elif is_inline_container:
            _ensure_space()

    _walk(soup)

    # Join and clean up the output
    raw = "".join(output_lines)

    # Normalize line breaks: collapse 3+ newlines to 2
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    # Clean up spaces around newlines
    raw = re.sub(r" *\n *", "\n", raw)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in raw.split("\n")]

    # Remove leading/trailing blank lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    return "\n\n".join(
        "\n".join(group)
        for group in _group_lines(lines)
    )


def _group_lines(lines: list[str]) -> list[list[str]]:
    """Group non-empty lines into paragraphs separated by blank lines."""
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    return groups
