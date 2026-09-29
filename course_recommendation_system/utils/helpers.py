"""
utils/helpers.py
----------------
Shared constants, text-normalisation helpers, dataset loading/validation and
filtering utilities used by the recommender and the Streamlit app.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "courses.csv"

# --------------------------------------------------------------------------- #
# Constants used by the UI and the recommender
# --------------------------------------------------------------------------- #
LEVELS = ["Beginner", "Intermediate", "Advanced"]
LEVEL_INDEX = {level: i for i, level in enumerate(LEVELS)}

DOMAINS = [
    "Data Structures & Algorithms",
    "Artificial Intelligence",
    "Machine Learning",
    "Data Science",
    "Web Development",
    "App Development",
    "Cyber Security",
    "Cloud Computing",
    "DevOps",
    "Database Management",
    "Python Programming",
    "Java Programming",
    "C++ Programming",
    "UI/UX Design",
]

CAREER_BY_DOMAIN = {
    "Data Structures & Algorithms": "Software Engineer",
    "Artificial Intelligence": "AI Engineer",
    "Machine Learning": "Machine Learning Engineer",
    "Data Science": "Data Scientist",
    "Web Development": "Web Developer",
    "App Development": "Mobile App Developer",
    "Cyber Security": "Security Analyst",
    "Cloud Computing": "Cloud Engineer",
    "DevOps": "DevOps Engineer",
    "Database Management": "Database Engineer",
    "Python Programming": "Python Developer",
    "Java Programming": "Java Developer",
    "C++ Programming": "C++ Developer",
    "UI/UX Design": "UI/UX Designer",
}
DOMAIN_BY_CAREER = {career: domain for domain, career in CAREER_BY_DOMAIN.items()}

NOT_SURE = "Not sure yet"
CAREER_GOALS = [NOT_SURE] + [CAREER_BY_DOMAIN[d] for d in DOMAINS]

EDUCATION_LEVELS = ["Diploma", "B.Tech / B.E.", "BCA / B.Sc", "M.Tech / MCA / M.Sc", "Other"]
SEMESTERS = [f"Semester {i}" for i in range(1, 9)] + ["Postgraduate"]

LANGUAGES = ["Python", "Java", "C", "C++", "JavaScript", "TypeScript", "C#",
             "Kotlin", "Swift", "Go", "R", "SQL", "PHP", "Dart"]

SKILL_OPTIONS = [
    "HTML", "CSS", "Git", "Linux", "SQL", "Data Structures", "Algorithms", "OOP",
    "Problem Solving", "NumPy", "Pandas", "Matplotlib", "Statistics",
    "Machine Learning", "Deep Learning", "React", "Node.js", "Django", "Flask",
    "Docker", "AWS", "Networking", "Cryptography", "Figma", "MongoDB",
    "REST APIs", "Excel", "Power BI",
]

LEARNING_PREFS = ["Video lectures", "Hands-on projects", "Reading & documentation", "Mixed"]

REQUIRED_COLUMNS = [
    "course_id", "course_name", "category", "description", "skills", "difficulty",
    "duration_hours", "platform", "rating", "enrollments", "prerequisites",
    "career_domain", "course_url",
]


class DatasetError(Exception):
    """Raised when the course dataset is missing, unreadable or invalid."""


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
# Special tokens are rewritten *before* punctuation is stripped so that
# "C++" does not collapse into "c" and "Node.js" does not split in two.
_REPLACEMENTS = [
    ("c++", "cpp"), ("c#", "csharp"), ("node.js", "nodejs"), ("react.js", "react"),
    ("ui/ux", "uiux"), ("ci/cd", "cicd"), ("asp.net", "aspnet"), (".net", "dotnet"),
]
_DISPLAY_TERMS = {"cpp": "C++", "csharp": "C#", "nodejs": "Node.js", "uiux": "UI/UX", "cicd": "CI/CD"}


def normalize_text(text) -> str:
    """Lower-case text, protect special tokens and strip punctuation."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = str(text).lower()
    for old, new in _REPLACEMENTS:
        value = value.replace(old, new)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def display_term(term: str) -> str:
    """Convert a normalised TF-IDF term back to a readable form."""
    return " ".join(_DISPLAY_TERMS.get(word, word) for word in term.split())


def split_list(value, sep: str = ";") -> List[str]:
    """Split a delimiter-separated string into a clean list."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(sep) if part.strip()]


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            out.append(str(item).strip())
    return out


def skill_matches(user_skills: set, course_skill: str) -> bool:
    """
    True if a course skill is covered by any of the student's (normalised) skills.
    Whole-word containment is allowed, e.g. student 'sql' covers 'Advanced SQL',
    but 'java' does not cover 'JavaScript'.
    """
    course_norm = normalize_text(course_skill)
    if not course_norm:
        return False
    padded_course = f" {course_norm} "
    for user in user_skills:
        if not user:
            continue
        if user == course_norm or f" {user} " in padded_course or padded_course in f" {user} ":
            return True
    return False


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def safe_url(value) -> str:
    """Return a valid http(s) URL or an empty string."""
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()
    return text if re.match(r"^https?://\S+$", text) else ""


def estimate_weeks(hours: float, weekly_hours: int) -> int:
    return max(1, math.ceil(float(hours) / max(int(weekly_hours), 1)))


def format_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_courses(path=DATA_PATH) -> pd.DataFrame:
    """Load, validate and clean the course dataset. Raises DatasetError on failure."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found at '{path}'. Run `python data/generate_courses.py` "
            "to regenerate it."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The dataset file is empty.") from exc
    except Exception as exc:  # malformed CSV, encoding problems, ...
        raise DatasetError(f"Could not read the dataset: {exc}") from exc

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {', '.join(missing)}")

    text_cols = ["course_id", "course_name", "category", "description", "skills",
                 "platform", "prerequisites", "career_domain", "course_url"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["course_name"] != ""].reset_index(drop=True)
    blank_id = df["course_id"] == ""
    df.loc[blank_id, "course_id"] = [f"AUTO{i:03d}" for i in df.index[blank_id]]
    df = df.drop_duplicates(subset="course_id", keep="first").reset_index(drop=True)
    if df.empty:
        raise DatasetError("The dataset contains no valid courses.")

    df["difficulty"] = df["difficulty"].fillna("").astype(str).str.strip().str.title()
    df.loc[~df["difficulty"].isin(LEVELS), "difficulty"] = "Beginner"

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    rating_median = df["rating"].median()
    df["rating"] = df["rating"].fillna(0.0 if pd.isna(rating_median) else rating_median).clip(0, 5)

    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    hours_median = df["duration_hours"].median()
    df["duration_hours"] = df["duration_hours"].fillna(10.0 if pd.isna(hours_median) else hours_median).clip(lower=1)

    df["enrollments"] = pd.to_numeric(df["enrollments"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["course_url"] = df["course_url"].apply(safe_url)

    if "learning_format" not in df.columns:
        df["learning_format"] = "Video lectures"
    df["learning_format"] = df["learning_format"].fillna("Video lectures").astype(str)

    fallback_order = df.groupby("category").cumcount() + 1
    if "path_order" in df.columns:
        df["path_order"] = pd.to_numeric(df["path_order"], errors="coerce").fillna(fallback_order).astype(int)
    else:
        df["path_order"] = fallback_order
    return df


# --------------------------------------------------------------------------- #
# Validation and filtering
# --------------------------------------------------------------------------- #
def validate_profile(profile) -> List[str]:
    """Return a list of human-readable problems with a student profile (empty = valid)."""
    errors: List[str] = []
    if not (profile.languages or profile.skills or profile.interests):
        errors.append("Select at least one programming language, skill or area of interest.")
    if profile.skill_level not in LEVELS:
        errors.append("Please choose a valid skill level.")
    if not 1 <= int(profile.weekly_hours) <= 80:
        errors.append("Weekly study hours must be between 1 and 80.")
    return errors


def apply_filters(
    df: pd.DataFrame,
    domains: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    max_hours: Optional[float] = None,
    min_rating: float = 0.0,
    platforms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply sidebar filters. An empty multiselect means 'no restriction'."""
    out = df
    if domains:
        out = out[out["category"].isin(domains)]
    if difficulties:
        out = out[out["difficulty"].isin(difficulties)]
    if max_hours is not None:
        out = out[out["duration_hours"] <= max_hours]
    if min_rating:
        out = out[out["rating"] >= min_rating]
    if platforms:
        out = out[out["platform"].isin(platforms)]
    return out
