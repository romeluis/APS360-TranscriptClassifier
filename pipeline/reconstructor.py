"""
Reconstruct structured semester/course data from BIO-tagged tokens.

Converts flat (tokens, BIO tags) into:
    [{"semester_name": "Fall 2021", "courses": [{"code": "CSC301", "name": "...", "grade": "A-"}, ...]}, ...]
"""

import re


_SEASONS = {"fall", "winter", "spring", "summer"}


def _is_plausible_semester(name: str) -> bool:
    """Return True if the semester name looks like a real semester."""
    if any(s in name for s in ("http", "://", "AM", "PM")):
        return False
    if ":" in name or "," in name:
        return False
    if re.search(r"\S-\S", name):
        return False
    if re.search(r"(In\s+Progress|Dean|Honours?|Honour\s+List)", name, re.IGNORECASE):
        return False
    if not any(s in name.lower() for s in _SEASONS):
        return False
    if not (1 <= len(name.split()) <= 5):
        return False
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", name)
    if years and not (1990 <= int(years[0]) <= 2035):
        return False
    return True


def _clean_semester_name(name: str) -> str:
    """Strip trailing separators like ' -' from a semester name."""
    return re.sub(r"[\s\-\u2013\u2014]+$", "", name).strip()


def _merge_adjacent_sem_tags(tokens: list[str], tags: list[str]) -> list[str]:
    """Convert consecutive B-SEM B-SEM into B-SEM I-SEM when they form a plausible semester."""
    tags = list(tags)
    for i in range(len(tags) - 1):
        if tags[i] == "B-SEM" and tags[i + 1] == "B-SEM":
            candidate = tokens[i] + " " + tokens[i + 1]
            if _is_plausible_semester(candidate):
                tags[i + 1] = "I-SEM"
    return tags


def reconstruct_semesters(tokens: list[str], tags: list[str]) -> list[dict]:
    """Parse BIO tags into structured semester/course data.

    Walks tokens left-to-right, grouping entities. Phantom semesters from
    header/footer noise are filtered, and their orphan courses are assigned
    to the nearest valid semester.
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

    # Clean semester names
    for sem in semesters:
        sem["semester_name"] = _clean_semester_name(sem["semester_name"])

    # Dedup consecutive identical semester names
    if semesters:
        deduped: list[dict] = [semesters[0]]
        for sem in semesters[1:]:
            if sem["semester_name"] == deduped[-1]["semester_name"]:
                deduped[-1]["courses"].extend(sem["courses"])
            else:
                deduped.append(sem)
        semesters = deduped

    # Filter phantom semesters; assign orphan courses to nearest valid
    pending_orphans: list[dict] = []
    filtered: list[dict] = []

    for sem in semesters:
        name = sem["semester_name"]
        if _is_plausible_semester(name):
            if pending_orphans:
                sem["courses"] = pending_orphans + sem["courses"]
                pending_orphans = []
            filtered.append(sem)
        else:
            if filtered:
                filtered[-1]["courses"].extend(sem["courses"])
            else:
                pending_orphans.extend(sem["courses"])

    if pending_orphans:
        if filtered:
            filtered[-1]["courses"].extend(pending_orphans)
        else:
            filtered = [{"semester_name": "Unknown Semester", "courses": pending_orphans}]

    # Second dedup pass
    if filtered:
        deduped2: list[dict] = [filtered[0]]
        for sem in filtered[1:]:
            if sem["semester_name"] == deduped2[-1]["semester_name"]:
                deduped2[-1]["courses"].extend(sem["courses"])
            else:
                deduped2.append(sem)
        filtered = deduped2

    # Drop semesters with no courses
    filtered = [s for s in filtered if s["courses"]]

    return filtered


def flatten_courses(semesters: list[dict]) -> list[dict]:
    """Flatten semester structure into a flat list of courses with semester_name."""
    flat = []
    for sem in semesters:
        for course in sem["courses"]:
            flat.append({"semester": sem["semester_name"], **course})
    return flat
