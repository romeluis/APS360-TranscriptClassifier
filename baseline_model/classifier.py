"""
Rule-based NER classifier for academic transcript tokens.

Uses a two-phase approach:
  Phase 1: Classify each token into a feature category (regex + vocabulary)
  Phase 2: State-machine left-to-right scan to assign BIO tags with context

Entity types: SEM (semester), CODE (course code), NAME (course name), GRADE (grade)
"""

import re

# ---------------------------------------------------------------------------
# Knowledge base — derived from transcript_generator/generators.py
# ---------------------------------------------------------------------------

DEPARTMENTS = {
    "CSC", "MAT", "STA", "PHY", "CHM", "BIO", "ENG", "HIS",
    "PHL", "SOC", "ECO", "PSY", "ECE", "MEC", "CIV",
}

LETTER_GRADES = {
    "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-",
    "D+", "D", "D-", "F",
}

SEASONS = {"Fall", "Winter", "Spring", "Summer", "Autumn"}

NAME_CONNECTORS = {"of", "and", "to", "the", "in", "for", "with"}

ROMAN_NUMERALS = {"I", "II", "III", "IV", "V"}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Course code — single token forms
RE_CODE_UOFT = re.compile(r"^[A-Z]{2,4}\d{3}[HY][15]$")       # ENG231Y1
RE_CODE_COMPACT = re.compile(r"^[A-Z]{2,4}\d{3}$")             # MEC110
RE_CODE_HYPHEN = re.compile(r"^[A-Z]{2,4}-\d{3}$")             # CSC-301

# Department prefix (standalone, for split codes like "MAT" "108")
RE_DEPT = re.compile(r"^[A-Z]{2,4}$")

# 3-digit course number (second half of split code)
RE_COURSE_NUM = re.compile(r"^\d{3}$")

# Year in expected range
RE_YEAR = re.compile(r"^\d{4}$")
YEAR_MIN, YEAR_MAX = 2000, 2030

# Grade patterns
RE_PERCENTAGE = re.compile(r"^\d{1,3}%$")
RE_DECIMAL = re.compile(r"^\d+\.\d+$")

# ---------------------------------------------------------------------------
# Phase 1: Token feature classification
# ---------------------------------------------------------------------------

# Feature labels
FEAT_CODE_FULL = "CODE_FULL"
FEAT_DEPT = "DEPT"
FEAT_COURSE_NUM = "COURSE_NUM"
FEAT_SEASON = "SEASON"
FEAT_YEAR = "YEAR"
FEAT_LETTER_GRADE = "LETTER_GRADE"
FEAT_PERCENTAGE = "PERCENTAGE"
FEAT_DECIMAL = "DECIMAL"
FEAT_CAPITALIZED = "CAPITALIZED"
FEAT_CONNECTOR = "CONNECTOR"
FEAT_OTHER = "OTHER"


def _classify_token(token: str) -> str:
    """Classify a single token into a feature category."""
    # Full course codes (highest priority — unambiguous)
    if RE_CODE_UOFT.match(token):
        return FEAT_CODE_FULL
    if RE_CODE_HYPHEN.match(token):
        return FEAT_CODE_FULL
    # Compact code (3-letter dept + 3-digit number, no suffix)
    if RE_CODE_COMPACT.match(token) and token[:3] in DEPARTMENTS:
        return FEAT_CODE_FULL

    # Standalone department prefix
    if RE_DEPT.match(token) and token in DEPARTMENTS:
        return FEAT_DEPT

    # Percentage grade (must check before DECIMAL)
    if RE_PERCENTAGE.match(token):
        return FEAT_PERCENTAGE

    # Decimal number (credits or GPA grade — ambiguous)
    if RE_DECIMAL.match(token):
        return FEAT_DECIMAL

    # Year
    if RE_YEAR.match(token):
        year = int(token)
        if YEAR_MIN <= year <= YEAR_MAX:
            return FEAT_YEAR

    # 3-digit course number (for split codes)
    if RE_COURSE_NUM.match(token):
        return FEAT_COURSE_NUM

    # Letter grade
    if token in LETTER_GRADES:
        return FEAT_LETTER_GRADE

    # Season word
    if token in SEASONS:
        return FEAT_SEASON

    # Pass grade
    if token == "P":
        return FEAT_LETTER_GRADE  # treat same as letter grade for tagging

    # Capitalized word (including hyphenated like "Human-Computer")
    if len(token) > 0 and token[0].isupper() and all(c.isalpha() or c == "-" for c in token):
        return FEAT_CAPITALIZED

    # Lowercase connector
    if token.lower() in NAME_CONNECTORS:
        return FEAT_CONNECTOR

    return FEAT_OTHER


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def _try_semester(tokens, features, i, n):
    """Try to match a semester at position i. Returns number of tokens (2) or 0."""
    if features[i] == FEAT_SEASON and i + 1 < n and features[i + 1] == FEAT_YEAR:
        return 2
    return 0


def _try_code(tokens, features, i, n):
    """Try to match a course code at position i. Returns 1 (single) or 2 (split) or 0."""
    if features[i] == FEAT_CODE_FULL:
        return 1
    if features[i] == FEAT_DEPT and i + 1 < n and features[i + 1] == FEAT_COURSE_NUM:
        return 2
    return 0


def _is_name_start(token, feat):
    """Can this token start a course name?"""
    return feat == FEAT_CAPITALIZED


def _is_name_continuation(token, feat, tokens, features, i, n):
    """Can this token continue an in-progress course name?"""
    # Roman numerals always continue
    if token in ROMAN_NUMERALS:
        return True

    # Multi-character capitalized words continue (unless they start a new entity)
    if feat == FEAT_CAPITALIZED:
        if _try_semester(tokens, features, i, n) > 0:
            return False
        if _try_code(tokens, features, i, n) > 0:
            return False
        return True

    # Connectors continue IF the next token is a name-like word
    if feat == FEAT_CONNECTOR or token.lower() in NAME_CONNECTORS:
        if i + 1 < n:
            nxt = tokens[i + 1]
            nxt_feat = features[i + 1]
            if nxt_feat == FEAT_CAPITALIZED:
                return True
            if nxt in ROMAN_NUMERALS:
                return True
            # "to 1500" — 4-digit historical number (not a year)
            if re.match(r"^\d{4}$", nxt):
                val = int(nxt)
                if val < YEAR_MIN:
                    return True
            # Connector followed by another connector (rare but handle: "of the")
            if nxt.lower() in NAME_CONNECTORS and i + 2 < n:
                if features[i + 2] == FEAT_CAPITALIZED or tokens[i + 2] in ROMAN_NUMERALS:
                    return True
        return False

    # Numbers like "1500" in "World History to 1500"
    if re.match(r"^\d+$", token) and i > 0:
        prev = tokens[i - 1].lower()
        if prev in {"to", "of", "in", "for"}:
            val = int(token)
            if 100 < val < YEAR_MIN:
                return True

    return False


def _is_unambiguous_grade(token, feat):
    """Is this token unambiguously a grade (not a decimal)?"""
    return feat in (FEAT_LETTER_GRADE, FEAT_PERCENTAGE)


def _classify_decimal_role(tokens, features, i, n):
    """
    When a decimal appears after a course name, determine if it's credits (O) or grade (B-GRADE).

    Lookahead rule: if the next token is also grade-like, this decimal is credits.
    Otherwise, this decimal is the grade itself.
    """
    if i + 1 >= n:
        return "GRADE"

    nxt_feat = features[i + 1]
    nxt_tok = tokens[i + 1]

    # Next is an unambiguous grade → current is credits
    if nxt_feat in (FEAT_LETTER_GRADE, FEAT_PERCENTAGE):
        return "CREDIT"
    # Next is also a decimal → current is credits, next is grade
    if nxt_feat == FEAT_DECIMAL:
        return "CREDIT"
    # Next is P (pass)
    if nxt_tok == "P":
        return "CREDIT"

    # Next is a code, semester, or non-entity → current is grade
    return "GRADE"


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(tokens: list[str]) -> list[str]:
    """
    Predict BIO NER tags for a list of transcript tokens.

    Args:
        tokens: whitespace-tokenized text from a transcript PDF

    Returns:
        list of BIO tags, same length as tokens
    """
    n = len(tokens)
    if n == 0:
        return []

    features = [_classify_token(t) for t in tokens]
    tags = ["O"] * n

    # Find the first semester to skip preamble
    preamble_end = n
    for i in range(n):
        if _try_semester(tokens, features, i, n) > 0:
            preamble_end = i
            break

    # State machine
    i = preamble_end
    state = "SCAN"

    while i < n:
        feat = features[i]
        tok = tokens[i]

        if state == "SCAN":
            # Look for semester
            sem_len = _try_semester(tokens, features, i, n)
            if sem_len > 0:
                tags[i] = "B-SEM"
                for j in range(1, sem_len):
                    tags[i + j] = "I-SEM"
                i += sem_len
                state = "AFTER_SEM"
                continue

            # Look for course code
            code_len = _try_code(tokens, features, i, n)
            if code_len > 0:
                tags[i] = "B-CODE"
                for j in range(1, code_len):
                    tags[i + j] = "I-CODE"
                i += code_len
                state = "AFTER_CODE"
                continue

            # Skip everything else as O
            i += 1

        elif state == "AFTER_SEM":
            # After semester: skip headers, look for first code
            code_len = _try_code(tokens, features, i, n)
            if code_len > 0:
                tags[i] = "B-CODE"
                for j in range(1, code_len):
                    tags[i + j] = "I-CODE"
                i += code_len
                state = "AFTER_CODE"
                continue

            sem_len = _try_semester(tokens, features, i, n)
            if sem_len > 0:
                tags[i] = "B-SEM"
                for j in range(1, sem_len):
                    tags[i + j] = "I-SEM"
                i += sem_len
                state = "AFTER_SEM"
                continue

            # Header/boilerplate → O
            i += 1

        elif state == "AFTER_CODE":
            # Expect course name start
            if _is_name_start(tok, feat):
                tags[i] = "B-NAME"
                i += 1
                state = "IN_NAME"
                continue

            # Single-word names that are connectors won't happen, but handle
            # unexpected tokens by trying grade directly (template with no name?)
            if _is_unambiguous_grade(tok, feat):
                tags[i] = "B-GRADE"
                i += 1
                state = "AFTER_GRADE"
                continue

            if feat == FEAT_DECIMAL:
                # Could be credits before grade (no name in this template?)
                role = _classify_decimal_role(tokens, features, i, n)
                if role == "CREDIT":
                    i += 1
                    state = "AFTER_CREDIT"
                else:
                    tags[i] = "B-GRADE"
                    i += 1
                    state = "AFTER_GRADE"
                continue

            # Unexpected — try to recover
            i += 1
            state = "SCAN"

        elif state == "IN_NAME":
            if _is_name_continuation(tok, feat, tokens, features, i, n):
                tags[i] = "I-NAME"
                i += 1
                continue

            # Name ended — re-process this token in AFTER_NAME
            state = "AFTER_NAME"
            # Don't increment i

        elif state == "AFTER_NAME":
            # Unambiguous grade
            if _is_unambiguous_grade(tok, feat):
                tags[i] = "B-GRADE"
                i += 1
                state = "AFTER_GRADE"
                continue

            # Decimal: credits or GPA grade?
            if feat == FEAT_DECIMAL:
                role = _classify_decimal_role(tokens, features, i, n)
                if role == "CREDIT":
                    i += 1  # skip as O
                    state = "AFTER_CREDIT"
                else:
                    tags[i] = "B-GRADE"
                    i += 1
                    state = "AFTER_GRADE"
                continue

            # Nothing matched — recover
            state = "SCAN"

        elif state == "AFTER_CREDIT":
            # Expect grade
            if _is_unambiguous_grade(tok, feat):
                tags[i] = "B-GRADE"
                i += 1
                state = "AFTER_GRADE"
                continue

            if feat == FEAT_DECIMAL:
                # This decimal is the grade (GPA scale)
                tags[i] = "B-GRADE"
                i += 1
                state = "AFTER_GRADE"
                continue

            # Unexpected — recover
            state = "SCAN"

        elif state == "AFTER_GRADE":
            # Expect next code, next semester, or summary text
            code_len = _try_code(tokens, features, i, n)
            if code_len > 0:
                tags[i] = "B-CODE"
                for j in range(1, code_len):
                    tags[i + j] = "I-CODE"
                i += code_len
                state = "AFTER_CODE"
                continue

            sem_len = _try_semester(tokens, features, i, n)
            if sem_len > 0:
                tags[i] = "B-SEM"
                for j in range(1, sem_len):
                    tags[i + j] = "I-SEM"
                i += sem_len
                state = "AFTER_SEM"
                continue

            # Summary/boilerplate → O, go back to SCAN
            i += 1
            state = "SCAN"

    return tags
