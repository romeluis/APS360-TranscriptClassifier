"""HTML assembly — expand repeating blocks and replace tags with entity wrapping."""

import html
import re

# Tags that map to BIO entity types
ENTITY_TAGS = {
    "CODE": "CODE",
    "COURSE_NAME": "NAME",
    "GRADE": "GRADE",
    "SEMESTER": "SEM",
}


def _wrap_entity(entity_type, value):
    """Wrap a value in an entity-annotated span (for BIO label extraction)."""
    escaped = html.escape(str(value))
    return f'<span data-entity="{entity_type}">{escaped}</span>'


def _replace_tag(text, tag_name, value):
    """Replace {TAG_NAME} with the value, wrapping entity tags in annotated spans."""
    placeholder = "{" + tag_name + "}"
    if tag_name in ENTITY_TAGS:
        replacement = _wrap_entity(ENTITY_TAGS[tag_name], value)
    else:
        replacement = html.escape(str(value))
    return text.replace(placeholder, replacement)


def assemble(raw_html, config, blocks, data):
    """
    Assemble a complete HTML document from template + generated data.

    Args:
        raw_html: the full template HTML string
        config: parsed template config (unused here, kept for interface consistency)
        blocks: dict of block name -> block HTML content
        data: generated transcript data from generators.generate_transcript_data()

    Returns:
        Complete HTML string with all blocks expanded and tags replaced.
    """
    semester_block_template = blocks["SEMESTER_BLOCK"]
    course_block_template = blocks["COURSE_BLOCK"]

    # Build all semester HTML
    all_semester_html = []
    for sem_data in data["semesters"]:
        sem_html = semester_block_template

        # Expand COURSE_BLOCK for each course in this semester
        course_rows = []
        for course in sem_data["courses"]:
            row = course_block_template
            row = _replace_tag(row, "CODE", course["code"])
            row = _replace_tag(row, "COURSE_NAME", course["name"])
            row = _replace_tag(row, "GRADE", course["grade"])
            row = _replace_tag(row, "CREDITS", course["credits"])
            row = _replace_tag(row, "GRADE_POINTS", course.get("grade_points", ""))
            row = _replace_tag(row, "COURSE_AVG", course.get("course_avg", ""))
            course_rows.append(row)

        # Replace the COURSE_BLOCK region with expanded course rows
        sem_html = re.sub(
            r"<!-- BEGIN:COURSE_BLOCK -->.*?<!-- END:COURSE_BLOCK -->",
            "\n".join(course_rows),
            sem_html,
            flags=re.DOTALL,
        )

        # Replace semester-level tags
        sem_html = _replace_tag(sem_html, "SEMESTER", sem_data["semester_name"])
        sem_html = sem_html.replace("{SEMESTER_GPA}", html.escape(sem_data.get("semester_gpa", "")))
        sem_html = sem_html.replace("{SEMESTER_CREDITS}", html.escape(sem_data.get("semester_credits", "")))
        sem_html = sem_html.replace("{STANDING}", html.escape(sem_data.get("standing", "")))

        all_semester_html.append(sem_html)

    # Replace the SEMESTER_BLOCK region with all expanded semesters
    assembled = re.sub(
        r"<!-- BEGIN:SEMESTER_BLOCK -->.*?<!-- END:SEMESTER_BLOCK -->",
        "\n".join(all_semester_html),
        raw_html,
        flags=re.DOTALL,
    )

    # Replace student info tags
    for tag_name, value in data["student_info"].items():
        assembled = assembled.replace("{" + tag_name + "}", html.escape(str(value)))

    # Replace summary tags
    for tag_name, value in data["summary"].items():
        assembled = assembled.replace("{" + tag_name + "}", html.escape(str(value)))

    # Remove CONFIG comment from output
    assembled = re.sub(r"<!--\s*CONFIG\s*.*?\s*-->", "", assembled, flags=re.DOTALL)

    return assembled
