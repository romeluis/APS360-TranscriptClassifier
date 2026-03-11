"""Parse HTML template files — extract config and structural blocks."""

import json
import re


def parse_template(template_path):
    """
    Parse an HTML template file.

    Returns:
        raw_html: str — the full template HTML
        config: dict — the parsed CONFIG JSON block
        blocks: dict — mapping of block name to its HTML content
                       e.g. {"SEMESTER_BLOCK": "...", "COURSE_BLOCK": "..."}
    """
    with open(template_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    # Extract CONFIG block
    config_match = re.search(r"<!--\s*CONFIG\s*(.*?)\s*-->", raw_html, re.DOTALL)
    if config_match:
        config = json.loads(config_match.group(1))
    else:
        config = {}

    # Extract blocks recursively (COURSE_BLOCK is nested inside SEMESTER_BLOCK)
    blocks = {}

    # First find the outer SEMESTER_BLOCK
    sem_match = re.search(
        r"<!-- BEGIN:SEMESTER_BLOCK -->(.*?)<!-- END:SEMESTER_BLOCK -->",
        raw_html,
        re.DOTALL,
    )
    if not sem_match:
        raise ValueError(f"Template {template_path} missing <!-- BEGIN:SEMESTER_BLOCK --> block")
    blocks["SEMESTER_BLOCK"] = sem_match.group(1)

    # Then find COURSE_BLOCK inside the semester block content
    course_match = re.search(
        r"<!-- BEGIN:COURSE_BLOCK -->(.*?)<!-- END:COURSE_BLOCK -->",
        blocks["SEMESTER_BLOCK"],
        re.DOTALL,
    )
    if not course_match:
        raise ValueError(f"Template {template_path} missing <!-- BEGIN:COURSE_BLOCK --> block")
    blocks["COURSE_BLOCK"] = course_match.group(1)

    return raw_html, config, blocks
