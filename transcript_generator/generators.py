"""Random academic data generation for synthetic transcripts."""

import json
import math
import random
import os
from pathlib import Path
from faker import Faker

fake = Faker()

# Load course names from data file
_DATA_DIR = Path(__file__).parent / "data"
with open(_DATA_DIR / "course_names.json") as f:
    COURSE_NAMES = json.load(f)

DEPARTMENTS = list(COURSE_NAMES.keys())

LETTER_GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]
# GPA points for each letter grade (4.0 scale)
LETTER_TO_GPA = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
}
# Weights favor middle-range grades (realistic distribution)
LETTER_WEIGHTS = [3, 8, 10, 12, 14, 12, 10, 8, 6, 4, 3, 2, 1]

INSTITUTIONS = [
    "University of Toronto",
    "McGill University",
    "University of British Columbia",
    "University of Alberta",
    "Western University",
    "Queen's University",
    "McMaster University",
    "University of Waterloo",
    "University of Ottawa",
    "Dalhousie University",
    "York University",
    "Carleton University",
    "Simon Fraser University",
    "University of Victoria",
    "University of Calgary",
    "Ryerson University",
    "University of Manitoba",
    "University of Saskatchewan",
    "Memorial University",
    "Concordia University",
]

PROGRAMS = [
    "Bachelor of Science, Computer Science",
    "Bachelor of Science, Mathematics",
    "Bachelor of Science, Physics",
    "Bachelor of Science, Chemistry",
    "Bachelor of Science, Biology",
    "Bachelor of Science, Statistics",
    "Bachelor of Arts, Economics",
    "Bachelor of Arts, Psychology",
    "Bachelor of Arts, English",
    "Bachelor of Arts, History",
    "Bachelor of Arts, Philosophy",
    "Bachelor of Arts, Sociology",
    "Bachelor of Engineering, Electrical Engineering",
    "Bachelor of Engineering, Mechanical Engineering",
    "Bachelor of Engineering, Civil Engineering",
    "Bachelor of Engineering, Computer Engineering",
    "Bachelor of Commerce, Finance",
    "Bachelor of Commerce, Accounting",
    "Bachelor of Science, Environmental Science",
    "Bachelor of Science, Neuroscience",
]

FACULTIES = [
    "Faculty of Arts and Science",
    "Faculty of Engineering",
    "Faculty of Applied Science",
    "Faculty of Science",
    "Faculty of Arts",
    "School of Engineering",
    "College of Science",
    "College of Arts and Sciences",
    "Faculty of Mathematics",
    "School of Computing",
]

STANDINGS = [
    "Good Standing",
    "Dean's Honour List",
    "Academic Probation",
    "Good Standing",
    "Good Standing",
    "Dean's Honour List",
    "Good Standing",
    "Good Standing",
]


# ---------- Config defaults ----------

DEFAULT_CONFIG = {
    "grade_scale": "letter",
    "courses_per_semester": [3, 6],
    "semesters": [2, 8],
    "course_code_format": "3letter_3digit",
    "has_credits": True,
    "has_semester_gpa": True,
    "has_cumulative_gpa": True,
    "has_standing": False,
    "credit_values": [0.5, 1.0],
    "seasons": ["Fall", "Winter", "Spring", "Summer"],
    "year_range": [2018, 2024],
}


def merge_config(template_config):
    """Merge template config with defaults."""
    merged = dict(DEFAULT_CONFIG)
    merged.update(template_config)
    return merged


# ---------- Individual generators ----------

def generate_course_code(config):
    """Generate a course code based on config format."""
    dept = random.choice(DEPARTMENTS)
    fmt = config.get("course_code_format", "3letter_3digit")

    if fmt == "3letter_3digit":
        level = random.choice([1, 2, 3, 4]) * 100 + random.randint(0, 99)
        return dept, f"{dept}{level}"
    elif fmt == "4letter_3digit":
        level = random.randint(100, 499)
        return dept, f"{dept} {level}"
    elif fmt == "dept_hyphen_num":
        level = random.randint(100, 499)
        return dept, f"{dept}-{level}"
    elif fmt == "uoft_style":
        level = random.choice([1, 2, 3, 4]) * 100 + random.randint(0, 99)
        suffix = random.choice(["H1", "Y1", "H5"])
        return dept, f"{dept}{level}{suffix}"
    else:
        # Default fallback
        level = random.randint(100, 499)
        return dept, f"{dept}{level}"


def generate_course_name(dept):
    """Pick a random course name for the given department."""
    if dept in COURSE_NAMES:
        return random.choice(COURSE_NAMES[dept])
    # Fallback for unknown departments
    all_names = [n for names in COURSE_NAMES.values() for n in names]
    return random.choice(all_names)


def generate_grade(config):
    """Generate a grade value based on the configured scale. Returns (display_value, gpa_points)."""
    scale = config.get("grade_scale", "letter")

    if scale == "letter":
        grade = random.choices(LETTER_GRADES, weights=LETTER_WEIGHTS, k=1)[0]
        return grade, LETTER_TO_GPA[grade]
    elif scale == "percentage":
        pct = random.choices(
            range(40, 101),
            weights=[1] * 10 + [2] * 10 + [4] * 10 + [6] * 10 + [8] * 11 + [5] * 10,
            k=1
        )[0]
        gpa = min(4.0, max(0.0, (pct - 50) / 12.5))
        return f"{pct}%", round(gpa, 1)
    elif scale == "gpa_point":
        gpa = round(random.triangular(0.0, 4.0, 3.0), 1)
        return str(gpa), gpa
    elif scale == "pass_fail":
        passed = random.random() > 0.15
        return ("P" if passed else "F"), (3.0 if passed else 0.0)
    else:
        grade = random.choices(LETTER_GRADES, weights=LETTER_WEIGHTS, k=1)[0]
        return grade, LETTER_TO_GPA[grade]


def generate_semesters(config):
    """Generate a chronologically ordered list of semester names."""
    seasons = config.get("seasons", ["Fall", "Winter", "Spring", "Summer"])
    year_range = config.get("year_range", [2018, 2024])
    sem_range = config.get("semesters", [2, 8])
    n_semesters = random.randint(sem_range[0], sem_range[1])

    start_year = random.randint(year_range[0], year_range[1] - n_semesters // len(seasons))
    season_idx = random.randint(0, len(seasons) - 1)

    semesters = []
    year = start_year
    for _ in range(n_semesters):
        semesters.append(f"{seasons[season_idx]} {year}")
        season_idx += 1
        if season_idx >= len(seasons):
            season_idx = 0
            year += 1
    return semesters


def generate_student_info(config):
    """Generate random student metadata."""
    return {
        "STUDENT_NAME": fake.name(),
        "STUDENT_ID": str(random.randint(100000000, 999999999)),
        "INSTITUTION": random.choice(INSTITUTIONS),
        "PROGRAM": random.choice(PROGRAMS),
        "FACULTY": random.choice(FACULTIES),
        "DOB": fake.date_of_birth(minimum_age=18, maximum_age=30).strftime("%B %d, %Y"),
        "ISSUE_DATE": fake.date_between(start_date="-1y", end_date="today").strftime("%B %d, %Y"),
    }


# ---------- Full transcript generation ----------

def generate_transcript_data(config):
    """
    Generate all random data for one transcript.

    Returns a dict with:
      - student_info: dict of non-entity tag values
      - semesters: list of semester dicts
      - summary: dict with cumulative stats
    """
    config = merge_config(config)
    student_info = generate_student_info(config)
    semester_names = generate_semesters(config)

    credit_values = config.get("credit_values", [0.5, 1.0])
    course_range = config.get("courses_per_semester", [3, 6])

    all_gpa_points = []
    all_credits = []
    semesters = []
    used_codes = set()

    for sem_name in semester_names:
        n_courses = random.randint(course_range[0], course_range[1])
        courses = []
        sem_gpa_sum = 0.0
        sem_credit_sum = 0.0

        for _ in range(n_courses):
            # Generate unique course code per transcript
            for _ in range(20):
                dept, code = generate_course_code(config)
                if code not in used_codes:
                    used_codes.add(code)
                    break

            name = generate_course_name(dept)
            grade_display, gpa_points = generate_grade(config)
            credits = random.choice(credit_values)

            courses.append({
                "code": code,
                "name": name,
                "grade": grade_display,
                "credits": str(credits),
                "grade_points": str(gpa_points),
            })

            sem_gpa_sum += gpa_points * credits
            sem_credit_sum += credits

        sem_gpa = round(sem_gpa_sum / sem_credit_sum, 2) if sem_credit_sum > 0 else 0.0
        all_gpa_points.append(sem_gpa_sum)
        all_credits.append(sem_credit_sum)

        # Determine standing based on GPA
        if config.get("has_standing", False):
            if sem_gpa >= 3.5:
                standing = "Dean's Honour List"
            elif sem_gpa < 1.5:
                standing = "Academic Probation"
            else:
                standing = "Good Standing"
        else:
            standing = ""

        semesters.append({
            "semester_name": sem_name,
            "courses": courses,
            "semester_gpa": f"{sem_gpa:.2f}",
            "semester_credits": str(sem_credit_sum),
            "standing": standing,
        })

    total_gpa_points = sum(all_gpa_points)
    total_credits = sum(all_credits)
    cumulative_gpa = round(total_gpa_points / total_credits, 2) if total_credits > 0 else 0.0

    summary = {
        "CUMULATIVE_GPA": f"{cumulative_gpa:.2f}",
        "TOTAL_CREDITS": str(total_credits),
        "STANDING": semesters[-1]["standing"] if semesters else "Good Standing",
    }

    return {
        "student_info": student_info,
        "semesters": semesters,
        "summary": summary,
    }
