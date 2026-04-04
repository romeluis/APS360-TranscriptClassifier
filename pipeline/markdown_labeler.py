"""
Assign BIO labels to whitespace-tokenized markdown using known entity values.

Given the markdown text (from marker) and the ground truth data (from the
transcript generator), this module finds each entity's tokens in the markdown
and assigns BIO tags.

Entity matching is ordered: semester names first (structural anchors), then
within each semester's scope: course codes, course names, then grades.
Course codes are the most reliable anchors (unique alphanumeric strings).
"""

import re
from typing import Optional


_MIN_COVERAGE = 0.70  # fall back to HTML labels if coverage is below this


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_token_span(
    md_tokens: list[str],
    entity_tokens: list[str],
    start: int = 0,
    end: Optional[int] = None,
    used: Optional[set] = None,
) -> Optional[tuple[int, int]]:
    """Find the first occurrence of entity_tokens as a contiguous subsequence
    in md_tokens[start:end], skipping positions in `used`.

    Returns (begin, end) indices or None.
    """
    if not entity_tokens:
        return None

    if end is None:
        end = len(md_tokens)
    if used is None:
        used = set()

    n_entity = len(entity_tokens)
    entity_lower = [t.lower() for t in entity_tokens]

    for i in range(start, end - n_entity + 1):
        if i in used:
            continue
        match = True
        for j in range(n_entity):
            pos = i + j
            if pos in used or pos >= end:
                match = False
                break
            if md_tokens[pos].lower() != entity_lower[j]:
                match = False
                break
        if match:
            return (i, i + n_entity)

    return None


def _find_token_span_fuzzy(
    md_tokens: list[str],
    entity_tokens: list[str],
    start: int = 0,
    end: Optional[int] = None,
    used: Optional[set] = None,
) -> Optional[tuple[int, int]]:
    """Like _find_token_span but strips markdown formatting chars (|, *, #, etc.)
    from tokens before comparing."""
    if not entity_tokens:
        return None

    if end is None:
        end = len(md_tokens)
    if used is None:
        used = set()

    def strip_md(t: str) -> str:
        return re.sub(r"[|*#_`~\[\](){}]", "", t).strip().lower()

    n_entity = len(entity_tokens)
    entity_stripped = [strip_md(t) for t in entity_tokens]
    # Skip empty entity tokens after stripping
    entity_stripped_nonempty = [t for t in entity_stripped if t]
    if not entity_stripped_nonempty:
        return None

    for i in range(start, end - n_entity + 1):
        if i in used:
            continue
        match = True
        for j in range(n_entity):
            pos = i + j
            if pos in used or pos >= end:
                match = False
                break
            if strip_md(md_tokens[pos]) != entity_stripped[j]:
                match = False
                break
        if match:
            return (i, i + n_entity)

    return None


def label_markdown(
    md_text: str,
    ground_truth: dict,
) -> tuple[list[str], list[str], float]:
    """Assign BIO labels to whitespace-tokenized markdown.

    Args:
        md_text: Markdown text from marker.
        ground_truth: Dict with key "semesters", each containing
            "semester_name" and "courses" (list of dicts with
            "code", "name", "grade").

    Returns:
        (tokens, bio_labels, coverage) where coverage is the fraction
        of expected entities that were successfully matched.
    """
    tokens = md_text.split()
    labels = ["O"] * len(tokens)
    used: set[int] = set()

    total_entities = 0
    matched_entities = 0

    semesters = ground_truth.get("semesters", [])

    # Track approximate position in the markdown for scoped matching.
    # We use a sliding scope: after matching a semester, courses are searched
    # from the semester position onward (but before the next semester).
    sem_positions: list[int] = []

    # Pass 1: Match all semester names to establish anchors
    search_from = 0
    for sem in semesters:
        sem_name = sem.get("semester_name", "")
        if not sem_name:
            sem_positions.append(search_from)
            continue

        total_entities += 1
        sem_tokens = sem_name.split()
        span = _find_token_span(tokens, sem_tokens, start=search_from, used=used)
        if span is None:
            span = _find_token_span_fuzzy(tokens, sem_tokens, start=search_from, used=used)

        if span is not None:
            matched_entities += 1
            begin, end = span
            labels[begin] = "B-SEM"
            for k in range(begin + 1, end):
                labels[k] = "I-SEM"
            for k in range(begin, end):
                used.add(k)
            sem_positions.append(begin)
            search_from = end
        else:
            sem_positions.append(search_from)

    # Compute scope boundaries: each semester's courses are searched
    # between its position and the next semester's position (or end of doc).
    scope_ends = []
    for i in range(len(semesters)):
        if i + 1 < len(sem_positions):
            scope_ends.append(len(tokens))  # extend to end, narrow below
        else:
            scope_ends.append(len(tokens))

    # More precise scope: from this semester to next semester
    for i in range(len(semesters)):
        if i + 1 < len(sem_positions):
            scope_ends[i] = sem_positions[i + 1]

    # Pass 2: Match courses within each semester's scope
    for sem_idx, sem in enumerate(semesters):
        scope_start = sem_positions[sem_idx] if sem_idx < len(sem_positions) else 0
        scope_end = scope_ends[sem_idx] if sem_idx < len(scope_ends) else len(tokens)

        for course in sem.get("courses", []):
            code = course.get("code", "")
            name = course.get("name", "")
            grade = course.get("grade", "")

            # Match course code
            if code:
                total_entities += 1
                code_tokens = code.split()
                span = _find_token_span(tokens, code_tokens, start=scope_start, end=scope_end, used=used)
                if span is None:
                    span = _find_token_span_fuzzy(tokens, code_tokens, start=scope_start, end=scope_end, used=used)
                if span is None:
                    # Expand scope to full document as fallback
                    span = _find_token_span(tokens, code_tokens, used=used)
                if span is not None:
                    matched_entities += 1
                    begin, end = span
                    labels[begin] = "B-CODE"
                    for k in range(begin + 1, end):
                        labels[k] = "I-CODE"
                    for k in range(begin, end):
                        used.add(k)

            # Match course name
            if name:
                total_entities += 1
                name_tokens = name.split()
                span = _find_token_span(tokens, name_tokens, start=scope_start, end=scope_end, used=used)
                if span is None:
                    span = _find_token_span_fuzzy(tokens, name_tokens, start=scope_start, end=scope_end, used=used)
                if span is None:
                    span = _find_token_span(tokens, name_tokens, used=used)
                if span is not None:
                    matched_entities += 1
                    begin, end = span
                    labels[begin] = "B-NAME"
                    for k in range(begin + 1, end):
                        labels[k] = "I-NAME"
                    for k in range(begin, end):
                        used.add(k)

            # Match grade
            if grade:
                total_entities += 1
                grade_tokens = grade.split()
                span = _find_token_span(tokens, grade_tokens, start=scope_start, end=scope_end, used=used)
                if span is None:
                    span = _find_token_span_fuzzy(tokens, grade_tokens, start=scope_start, end=scope_end, used=used)
                if span is None:
                    span = _find_token_span(tokens, grade_tokens, used=used)
                if span is not None:
                    matched_entities += 1
                    begin, end = span
                    labels[begin] = "B-GRADE"
                    for k in range(begin + 1, end):
                        labels[k] = "I-GRADE"
                    for k in range(begin, end):
                        used.add(k)

    coverage = matched_entities / total_entities if total_entities > 0 else 0.0
    return tokens, labels, coverage


def label_markdown_or_fallback(
    md_text: str,
    ground_truth: dict,
    html_tokens: list[str],
    html_labels: list[str],
    min_coverage: float = _MIN_COVERAGE,
) -> tuple[list[str], list[str], str]:
    """Try markdown labeling; fall back to HTML-extracted labels if coverage is too low.

    Returns:
        (tokens, labels, source) where source is "markdown" or "html_fallback".
    """
    tokens, labels, coverage = label_markdown(md_text, ground_truth)

    if coverage >= min_coverage:
        return tokens, labels, "markdown"
    else:
        return html_tokens, html_labels, "html_fallback"
