"""
Reconstruct structured semester/course data from BIO-tagged tokens.

Used by:
  - metrics.py  (course extraction rate comparison)
  - visualize.py (table display of extracted data)
"""

import re


def _is_plausible_semester(name: str) -> bool:
    """Return True if the semester name looks like a real semester, not header/footer noise.

    Rejects things like:
      "2026-03-30,"  (timestamp from page header)
      "2021 Fall-2026 Winter:"  (registration date range with colon)
      "Fall-2026"    (bare-hyphen date fragments from registration banners)
      "https://acorn..."  (URL footer)
    Accepts:
      "Fall 2021", "2021 Fall", "Winter 2022", "2021 - Fall", "2025 Summer"
    """
    # Reject anything containing a URL or time-of-day marker
    if any(s in name for s in ("http", "://", "AM", "PM")):
        return False

    # Reject names with colons (timestamps, label separators like "GPA:")
    if ":" in name:
        return False

    # Reject names with commas (e.g. "2026-03-30,")
    if "," in name:
        return False

    # Reject bare hyphens (no spaces around them): "Fall-2026", "2021-Fall"
    # Allow " - " (with spaces) which is a valid season separator like "2021 - Fall"
    if re.search(r"\S-\S", name):
        return False

    # Must contain a year in a plausible academic range
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", name)
    if not years:
        return False
    year = int(years[0])
    if not (1990 <= year <= 2035):
        return False

    # Must contain a recognized season word
    seasons = {"fall", "winter", "spring", "summer"}
    if not any(s in name.lower() for s in seasons):
        return False

    # Must be between 1 and 5 tokens long (e.g. "Fall 2021" = 2, "2021 - Fall" = 3)
    if not (1 <= len(name.split()) <= 5):
        return False

    return True


def _merge_adjacent_sem_tags(tokens: list[str], tags: list[str]) -> list[str]:
    """Convert consecutive B-SEM B-SEM pairs into B-SEM I-SEM when the two
    tokens together form a plausible semester name (e.g. "2022" + "Fall").

    This fixes model outputs where year and season are tagged as two separate
    B-SEM entities instead of one multi-token span.
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
    `_is_plausible_semester()` — their courses are merged into the nearest
    valid semester.

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

        Courses with missing fields have None for those fields.
    """
    # Pre-pass: merge adjacent B-SEM B-SEM pairs into B-SEM I-SEM when the
    # pair forms a valid semester name (fixes split year/season tagging).
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
            # Ensure there's a semester to attach to
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

        # O tags are skipped

    # Finalize any remaining course
    _finalize_course()

    # ── Dedup consecutive identical semester names ────────────────────────────
    # ACORN and similar systems repeat semester headers as page navigation.
    # Merge any consecutive semesters with the same name into one.
    if semesters:
        deduped: list[dict] = [semesters[0]]
        for sem in semesters[1:]:
            if sem["semester_name"] == deduped[-1]["semester_name"]:
                deduped[-1]["courses"].extend(sem["courses"])
            else:
                deduped.append(sem)
        semesters = deduped

    # ── Post-process: remove phantom semesters from header/footer noise ──────
    # Courses from phantom semesters are forwarded to the next valid semester.
    orphan_courses: list[dict] = []
    filtered: list[dict] = []

    for sem in semesters:
        if _is_plausible_semester(sem["semester_name"]):
            # Prepend any orphaned courses that came before this valid semester
            if orphan_courses:
                sem["courses"] = orphan_courses + sem["courses"]
                orphan_courses = []
            filtered.append(sem)
        else:
            # Phantom semester — save its courses for the next valid one
            orphan_courses.extend(sem["courses"])

    # Any remaining orphans go to the last valid semester.
    # If NO valid semesters were found at all, collect everything into one bucket
    # so the caller still receives the extracted courses.
    if orphan_courses:
        if filtered:
            filtered[-1]["courses"].extend(orphan_courses)
        else:
            filtered = [{"semester_name": "Unknown Semester", "courses": orphan_courses}]

    # Drop valid-name semesters that ended up with zero courses (navigation phantoms).
    filtered = [s for s in filtered if s["courses"]]

    return filtered


def flatten_courses(semesters: list[dict]) -> list[dict]:
    """Flatten semester structure into a flat list of courses.

    Each returned dict includes the semester_name alongside the course fields.
    """
    flat = []
    for sem in semesters:
        for course in sem["courses"]:
            flat.append({
                "semester": sem["semester_name"],
                **course,
            })
    return flat
