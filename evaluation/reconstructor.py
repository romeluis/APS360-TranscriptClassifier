"""
Reconstruct structured semester/course data from BIO-tagged tokens.

Used by:
  - metrics.py  (course extraction rate comparison)
  - visualize.py (table display of extracted data)
"""


def reconstruct_semesters(tokens: list[str], tags: list[str]) -> list[dict]:
    """Parse BIO tags into structured semester/course data.

    Walks tokens left-to-right, grouping entities into semesters and courses.

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

        # O tags are skipped

    # Finalize any remaining course
    _finalize_course()

    return semesters


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
