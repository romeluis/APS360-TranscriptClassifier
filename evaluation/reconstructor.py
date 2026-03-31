"""
Reconstruct structured semester/course data from BIO-tagged tokens.

Used by:
  - metrics.py  (course extraction rate comparison)
  - visualize.py (table display of extracted data)
"""

import re


_SEASONS = {"fall", "winter", "spring", "summer"}


def _is_plausible_semester(name: str) -> bool:
    """Return True if the semester name looks like a real semester, not header/footer noise.

    Rejects things like:
      "2026-03-30,"  (timestamp from page header)
      "2021 Fall-2026 Winter:"  (registration date range with colon)
      "Fall-2026"    (bare-hyphen fragments)
      "https://acorn..."  (URL footer)
    Accepts:
      "Fall 2021", "2021 Fall", "Winter 2022", "2025 Summer"
    """
    if any(s in name for s in ("http", "://", "AM", "PM")):
        return False
    if ":" in name or "," in name:
        return False
    # Reject bare hyphens: "Fall-2026", "2021-Fall" (allow " - " with spaces)
    if re.search(r"\S-\S", name):
        return False
    # Reject ACORN registration-history fragments like "In Progress - 2021 Fall"
    # or "Dean's Honour List" — these are O-context occurrences of year+season
    if re.search(r"(In\s+Progress|Dean|Honours?|Honour\s+List)", name, re.IGNORECASE):
        return False
    # Must contain a year in plausible academic range
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", name)
    if not years or not (1990 <= int(years[0]) <= 2035):
        return False
    # Must contain a recognized season word
    if not any(s in name.lower() for s in _SEASONS):
        return False
    # 1–5 tokens (e.g. "Fall 2021" = 2, "2021 Fall" = 2, "Fall 2021 - " = 3)
    if not (1 <= len(name.split()) <= 5):
        return False
    return True


def _clean_semester_name(name: str) -> str:
    """Strip trailing noise from a semester name.

    Handles cases where the model includes separator punctuation
    (' -', ' -', ' — ') that trails after the year/season tokens.
    E.g. "2022 Winter -" → "2022 Winter"
    """
    return re.sub(r"[\s\-–—]+$", "", name).strip()


def _merge_adjacent_sem_tags(tokens: list[str], tags: list[str]) -> list[str]:
    """Convert consecutive B-SEM B-SEM pairs into B-SEM I-SEM when the two
    tokens together form a plausible semester name (e.g. "2022" + "Fall").

    Fixes model outputs where year and season are tagged as separate B-SEM
    entities instead of one multi-token span.
    """
    tags = list(tags)
    for i in range(len(tags) - 1):
        if tags[i] == "B-SEM" and tags[i + 1] == "B-SEM":
            candidate = tokens[i] + " " + tokens[i + 1]
            if _is_plausible_semester(candidate):
                tags[i + 1] = "I-SEM"
    return tags


def reconstruct_semesters(tokens: list[str], tags: list[str]) -> list[dict]:
    """Parse BIO tags into structured semester/course data.

    Walks tokens left-to-right, grouping entities into semesters and courses.
    Phantom semesters from header/footer noise are filtered using
    `_is_plausible_semester()`.

    Orphan courses from phantom semesters are assigned using a
    **nearest-neighbour rule**: they go to the last valid semester
    that appeared before the phantom (backward assignment), or to the
    next valid semester if none has appeared yet (forward assignment).
    This correctly handles formats like ACORN where the page header
    "2026-03-30," creates a phantom immediately after the semester header
    but before the section's courses.

    Args:
        tokens: whitespace-tokenized transcript text
        tags:   BIO labels aligned with tokens

    Returns:
        List of semester dicts::

            [
                {
                    "semester_name": "Fall 2019",
                    "courses": [
                        {"code": "CSC301H1", "name": "Machine Learning", "grade": "A-"},
                        ...
                    ]
                },
                ...
            ]
    """
    tags = _merge_adjacent_sem_tags(tokens, tags)

    semesters: list[dict] = []
    current_semester: dict | None = None
    current_course: dict | None = None

    def _finalize_course():
        nonlocal current_course
        if current_course is not None:
            if current_semester is not None:
                current_semester["courses"].append(current_course)
            current_course = None

    for token, tag in zip(tokens, tags):
        if tag == "B-SEM":
            _finalize_course()
            current_semester = {"semester_name": token, "courses": []}
            semesters.append(current_semester)

        elif tag == "I-SEM":
            if current_semester is not None:
                current_semester["semester_name"] += " " + token

        elif tag == "B-CODE":
            _finalize_course()
            if current_semester is None:
                current_semester = {"semester_name": "Unknown Semester", "courses": []}
                semesters.append(current_semester)
            current_course = {"code": token, "name": None, "grade": None}

        elif tag == "I-CODE":
            if current_course is not None:
                current_course["code"] += " " + token

        elif tag == "B-NAME":
            if current_course is not None:
                current_course["name"] = token

        elif tag == "I-NAME":
            if current_course is not None and current_course["name"] is not None:
                current_course["name"] += " " + token

        elif tag == "B-GRADE":
            if current_course is not None:
                current_course["grade"] = token

        elif tag == "I-GRADE":
            if current_course is not None and current_course["grade"] is not None:
                current_course["grade"] += " " + token

    _finalize_course()

    # ── Clean semester names (strip trailing separators like " -") ────────────
    for sem in semesters:
        sem["semester_name"] = _clean_semester_name(sem["semester_name"])

    # ── Dedup consecutive identical semester names ────────────────────────────
    if semesters:
        deduped: list[dict] = [semesters[0]]
        for sem in semesters[1:]:
            if sem["semester_name"] == deduped[-1]["semester_name"]:
                deduped[-1]["courses"].extend(sem["courses"])
            else:
                deduped.append(sem)
        semesters = deduped

    # ── Filter phantom semesters; assign orphan courses to nearest valid ───────
    # Strategy: backward assignment — orphan courses from a phantom semester
    # go to the last valid semester seen before the phantom (if any), otherwise
    # they are held and forwarded to the next valid semester.
    #
    # Rationale: page-break headers ("2026-03-30,") appear AFTER the section
    # header ("2021 Fall") but BEFORE the section's courses. Backward assignment
    # puts those courses back in the correct semester.
    pending_orphans: list[dict] = []   # orphans waiting for a valid semester (forward)
    filtered: list[dict] = []

    for sem in semesters:
        name = sem["semester_name"]
        if _is_plausible_semester(name):
            # Attach any forward-pending orphans (before first valid semester)
            if pending_orphans:
                sem["courses"] = pending_orphans + sem["courses"]
                pending_orphans = []
            filtered.append(sem)
        else:
            # Phantom — assign its courses backward or hold forward
            if filtered:
                # Backward: give courses to the last valid semester
                filtered[-1]["courses"].extend(sem["courses"])
            else:
                # No valid semester yet — hold for next valid one
                pending_orphans.extend(sem["courses"])

    # Any still-pending orphans at the end go to the last valid semester
    if pending_orphans:
        if filtered:
            filtered[-1]["courses"].extend(pending_orphans)
        else:
            filtered = [{"semester_name": "Unknown Semester", "courses": pending_orphans}]

    # ── Second dedup pass (names may now match after cleaning) ───────────────
    if filtered:
        deduped2: list[dict] = [filtered[0]]
        for sem in filtered[1:]:
            if sem["semester_name"] == deduped2[-1]["semester_name"]:
                deduped2[-1]["courses"].extend(sem["courses"])
            else:
                deduped2.append(sem)
        filtered = deduped2

    # Drop semesters with no courses (navigation phantoms that had none)
    filtered = [s for s in filtered if s["courses"]]

    return filtered


def flatten_courses(semesters: list[dict]) -> list[dict]:
    """Flatten semester structure into a flat list of courses with semester_name."""
    flat = []
    for sem in semesters:
        for course in sem["courses"]:
            flat.append({"semester": sem["semester_name"], **course})
    return flat
