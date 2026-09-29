"""
models/recommender.py
---------------------
Content-based course recommender.

METHODOLOGY (summary)
1. Feature engineering: every course is turned into ONE text document built from
   name, category, skills, career domain, description and prerequisites.
   Important fields (skills, category, career) are repeated so they receive a
   higher weight after TF-IDF.
2. TF-IDF: the corpus of course documents is vectorised with TfidfVectorizer
   (unigrams + bigrams, English stop-words removed, sub-linear term frequency).
3. Student profile -> text document -> same TF-IDF space (vectorizer.transform).
4. Cosine similarity between the profile vector and every course vector gives the
   *content similarity* (0..1).
5. Hybrid weighted score: content similarity is combined with explicit signals
   (skill overlap, interest match, career match, difficulty fit, format fit).
   Final score = sum(weight_i * component_i) * 100.
6. Explainability: every recommendation carries matching skills, the strongest
   shared TF-IDF terms and a natural-language reason.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import (
    DOMAIN_BY_CAREER,
    LEVEL_INDEX,
    NOT_SURE,
    display_term,
    normalize_text,
    skill_matches,
    split_list,
)

COMPONENTS = ["content_similarity", "skill_match", "interest_match",
              "career_match", "difficulty_fit", "format_fit"]


# --------------------------------------------------------------------------- #
# Student profile
# --------------------------------------------------------------------------- #
@dataclass
class StudentProfile:
    name: str = "Student"
    education_level: str = ""
    semester: str = ""
    languages: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    skill_level: str = "Beginner"
    interests: List[str] = field(default_factory=list)
    career_goal: str = NOT_SURE
    weekly_hours: int = 6
    learning_preference: str = "Mixed"

    @property
    def known_skills(self) -> set:
        """Normalised set of everything the student says they know."""
        return {normalize_text(s) for s in list(self.languages) + list(self.skills) if normalize_text(s)}

    def to_text(self) -> str:
        """
        Convert the profile into a text document for TF-IDF.
        Skills are repeated 3x, languages/interests/career 2x (same idea as the
        course documents) so that stronger signals carry more weight.
        """
        parts: List[str] = []
        parts += [normalize_text(" ".join(self.languages))] * 2
        parts += [normalize_text(" ".join(self.skills))] * 3
        parts += [normalize_text(" ".join(self.interests))] * 2
        if self.career_goal and self.career_goal != NOT_SURE:
            parts += [normalize_text(self.career_goal)] * 2
            domain = DOMAIN_BY_CAREER.get(self.career_goal)
            if domain:
                parts += [normalize_text(domain)] * 2
        return " ".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
class CourseRecommender:
    # Weights of the hybrid score (sum = 1.0)
    WEIGHTS: Dict[str, float] = {
        "content_similarity": 0.40,
        "skill_match": 0.15,
        "interest_match": 0.15,
        "career_match": 0.10,
        "difficulty_fit": 0.15,
        "format_fit": 0.05,
    }
    MASTERED_THRESHOLD = 0.8   # student already knows >= 80% of the course skills
    MASTERED_PENALTY = 0.7     # ...so the course is ranked lower (nothing new to learn)
    FOUNDATIONS = {"Machine Learning": "python", "Data Science": "python",
                   "Artificial Intelligence": "python"}

    def __init__(self, courses: pd.DataFrame):
        if courses is None or courses.empty:
            raise ValueError("The course dataset is empty.")
        self.courses = courses.reset_index(drop=True).copy()
        self._engineer_features()
        self._fit_tfidf()

    # ---- feature engineering ------------------------------------------------ #
    def _engineer_features(self) -> None:
        c = self.courses
        c["skill_list"] = c["skills"].apply(split_list)
        c["skill_set"] = c["skill_list"].apply(lambda xs: frozenset(normalize_text(x) for x in xs))
        c["level_index"] = c["difficulty"].map(LEVEL_INDEX).fillna(0).astype(int)
        c["text_feature"] = c.apply(self._course_text, axis=1)

    @staticmethod
    def _course_text(row) -> str:
        """Weighted text document: (field, repetitions)."""
        fields = [
            (row["course_name"], 2),
            (row["category"], 2),
            (row["skills"], 3),
            (row["career_domain"], 2),
            (row["description"], 1),
            (row["prerequisites"], 1),
        ]
        return " ".join(" ".join([normalize_text(text)] * weight) for text, weight in fields if text)

    def _fit_tfidf(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.course_matrix = self.vectorizer.fit_transform(self.courses["text_feature"])
        self.feature_names = self.vectorizer.get_feature_names_out()

    # ---- scoring components -------------------------------------------------- #
    @staticmethod
    def _difficulty_fit(user_level: int, course_level: int) -> float:
        diff = course_level - user_level
        if diff == 0:
            return 1.0
        if diff == 1:
            return 0.6   # a manageable stretch
        if diff == -1:
            return 0.4   # revision material
        return 0.1       # too easy / too hard

    @staticmethod
    def _format_fit(preference: str, course_format: str) -> float:
        if preference == "Mixed":
            return 0.5
        return 1.0 if preference.lower() == str(course_format).lower() else 0.0

    def _top_terms(self, profile_vec, n: int = 3) -> List[List[str]]:
        """Strongest TF-IDF terms shared by the profile and each course (explainability)."""
        n_courses = self.course_matrix.shape[0]
        if profile_vec is None:
            return [[] for _ in range(n_courses)]
        product = self.course_matrix.multiply(profile_vec).tocsr()
        result: List[List[str]] = []
        for i in range(n_courses):
            start, end = product.indptr[i], product.indptr[i + 1]
            if end == start:
                result.append([])
                continue
            idx, val = product.indices[start:end], product.data[start:end]
            chosen: List[str] = []
            for j in np.argsort(val)[::-1]:
                term = self.feature_names[idx[j]]
                if any(term in c or c in term for c in chosen):
                    continue
                chosen.append(term)
                if len(chosen) == n:
                    break
            result.append([display_term(t) for t in chosen])
        return result

    # ---- main API -------------------------------------------------------------- #
    def recommend(self, profile: StudentProfile, explain: bool = True) -> pd.DataFrame:
        """Score and rank ALL courses for a profile. Filtering/top-N is done by the caller."""
        df = self.courses.copy()
        n = len(df)

        # 1) content similarity (TF-IDF + cosine)
        text = profile.to_text()
        profile_vec = self.vectorizer.transform([text]) if text else None
        cos = cosine_similarity(profile_vec, self.course_matrix).ravel() if profile_vec is not None else np.zeros(n)
        max_cos = cos.max() if n else 0.0
        content = cos / max_cos if max_cos > 0 else cos   # rescale so the best match = 1
        df["raw_cosine"] = cos

        # 2) explicit signals
        known = profile.known_skills
        interests = {normalize_text(i) for i in profile.interests}
        user_level = LEVEL_INDEX.get(profile.skill_level, 0)

        matched = [[s for s in skills if skill_matches(known, s)] for skills in df["skill_list"]]
        ratio = np.array([len(m) / len(s) if s else 0.0 for m, s in zip(matched, df["skill_list"])])
        interest = df["category"].map(lambda cat: 1.0 if normalize_text(cat) in interests else 0.0).to_numpy()
        if profile.career_goal and profile.career_goal != NOT_SURE:
            career = (df["career_domain"] == profile.career_goal).astype(float).to_numpy()
        else:
            career = np.zeros(n)
        difficulty = df["level_index"].map(lambda lvl: self._difficulty_fit(user_level, lvl)).to_numpy()
        fmt = df["learning_format"].map(lambda f: self._format_fit(profile.learning_preference, f)).to_numpy()

        df["matching_skills"] = matched
        df["content_similarity"] = content
        df["skill_match"] = ratio
        df["interest_match"] = interest
        df["career_match"] = career
        df["difficulty_fit"] = difficulty
        df["format_fit"] = fmt

        # 3) hybrid weighted score (0..100) with a penalty for already-mastered courses
        mastered = ratio >= self.MASTERED_THRESHOLD
        factor = np.where(mastered, self.MASTERED_PENALTY, 1.0)
        for comp in COMPONENTS:
            df[f"contrib_{comp}"] = self.WEIGHTS[comp] * df[comp].to_numpy() * factor * 100
        df["score"] = df[[f"contrib_{c}" for c in COMPONENTS]].sum(axis=1).clip(0, 100)
        df["mastered"] = mastered

        # 4) explanations
        if explain:
            terms = self._top_terms(profile_vec)
            df["reason"] = [self._build_reason(profile, df.iloc[i], terms[i]) for i in range(n)]
        else:
            df["reason"] = ""

        df = df.sort_values(["score", "rating"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, n + 1))
        return df

    @staticmethod
    def _build_reason(profile: StudentProfile, row, terms: List[str]) -> str:
        parts: List[str] = []
        matched = row["matching_skills"]
        if matched:
            parts.append(f"Builds on your existing skills in {', '.join(matched[:4])}.")
        if row["interest_match"] > 0:
            parts.append(f"Matches your interest in {row['category']}.")
        if row["career_match"] > 0:
            parts.append(f"Aligned with your career goal ({row['career_domain']}).")
        if terms:
            parts.append(f"Shares key topics with your profile: {', '.join(terms)}.")
        fit = row["difficulty_fit"]
        if fit == 1.0:
            parts.append(f"Its {row['difficulty']} level matches your {profile.skill_level} level.")
        elif fit == 0.6:
            parts.append("A manageable step up from your current level.")
        elif fit == 0.4:
            parts.append("Useful revision material below your current level.")
        else:
            parts.append("Its level differs a lot from yours, so check the prerequisites first.")
        if row["mastered"]:
            parts.append("You already know most of its skills, so it is ranked lower.")
        if row["format_fit"] == 1.0 and profile.learning_preference != "Mixed":
            parts.append(f"Delivered as {str(row['learning_format']).lower()}, which suits your learning preference.")
        return " ".join(parts) if parts else "General match with your profile."

    # ---- skill gap analysis ------------------------------------------------------ #
    def skill_gap_analysis(self, profile: StudentProfile, recs: pd.DataFrame, max_missing: int = 10) -> dict:
        """Compare the student's skills with the skills demanded by the recommended courses."""
        empty = {"existing": [], "missing": [], "path": pd.DataFrame()}
        if recs is None or recs.empty:
            return empty
        known = profile.known_skills
        existing_counter, missing_counter, display = Counter(), Counter(), {}
        for skills in recs["skill_list"]:
            for skill in skills:
                key = normalize_text(skill)
                display.setdefault(key, skill)
                (existing_counter if skill_matches(known, skill) else missing_counter)[key] += 1

        existing = [display[k] for k, _ in existing_counter.most_common()]
        missing = [(display[k], cnt) for k, cnt in missing_counter.most_common(max_missing)]

        user_level = LEVEL_INDEX.get(profile.skill_level, 0)
        picks: Dict[str, dict] = {}
        for skill, _ in missing:
            key = normalize_text(skill)
            cand = self.courses[self.courses["skill_set"].apply(lambda s, k=key: k in s)]
            if cand.empty:
                continue
            cand = cand.assign(gap=(cand["level_index"] - user_level).abs())
            best = cand.sort_values(["gap", "rating"], ascending=[True, False]).iloc[0]
            picks.setdefault(best["course_id"], {"row": best, "skills": []})["skills"].append(skill)

        rows = [{
            "level_index": e["row"]["level_index"], "path_order": e["row"]["path_order"],
            "course_name": e["row"]["course_name"], "category": e["row"]["category"],
            "difficulty": e["row"]["difficulty"], "duration_hours": e["row"]["duration_hours"],
            "platform": e["row"]["platform"], "course_url": e["row"]["course_url"],
            "covers_missing_skills": ", ".join(e["skills"]),
        } for e in picks.values()]
        path = pd.DataFrame(rows)
        if not path.empty:
            path = path.sort_values(["level_index", "path_order"]).reset_index(drop=True)
        return {"existing": existing, "missing": missing, "path": path}

    # ---- learning roadmap ------------------------------------------------------------ #
    def build_roadmap(self, profile: StudentProfile, category: str) -> pd.DataFrame:
        """Beginner -> Intermediate -> Advanced path for one domain, tailored to what the student knows."""
        domain_df = self.courses[self.courses["category"] == category]
        if domain_df.empty:
            return pd.DataFrame()
        domain_df = domain_df.sort_values(["level_index", "path_order"])
        known = profile.known_skills
        user_level = LEVEL_INDEX.get(profile.skill_level, 0)

        def coverage(skills: List[str]) -> float:
            return sum(skill_matches(known, s) for s in skills) / len(skills) if skills else 0.0

        keep = [r for _, r in domain_df.iterrows()
                if not (coverage(r["skill_list"]) >= 0.6 and r["level_index"] <= user_level)]
        if len(keep) < 3:   # never return an almost-empty roadmap
            keep = [r for _, r in domain_df.iterrows() if r["level_index"] >= user_level] or \
                   [r for _, r in domain_df.iterrows()]

        steps = []
        foundation = self.FOUNDATIONS.get(category)
        if foundation and not any(foundation in k for k in known):
            py = self.courses[self.courses["category"] == "Python Programming"].sort_values("path_order")
            if not py.empty:
                steps.append((py.iloc[0], "Foundation: you need this before the rest of the path"))
        steps += [(r, "") for r in keep]

        rows = [{
            "step": i, "stage": r["difficulty"], "course_name": r["course_name"], "category": r["category"],
            "difficulty": r["difficulty"], "duration_hours": r["duration_hours"], "platform": r["platform"],
            "course_url": r["course_url"], "note": note,
        } for i, (r, note) in enumerate(steps, start=1)]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Offline evaluation (no ground-truth labels available)
# --------------------------------------------------------------------------- #
def evaluate_offline(rec: CourseRecommender, k: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Simulated 'hidden course' evaluation.
    For every course we create a synthetic student who knows a random 50% of that
    course's skills (and, in scenario B, also states its category and career goal).
    We then check whether the hidden course is retrieved in the top-K.

    Metrics: Hit Rate@K, Mean Reciprocal Rank (MRR) and Category Precision@K.
    These measure retrieval consistency on synthetic profiles, NOT real-user
    satisfaction, so they must not be reported as 'accuracy'.
    """
    rng = np.random.default_rng(seed)
    records = []
    for scenario in ("A: skills only", "B: skills + interest + career"):
        hits, rr, cat_prec = [], [], []
        for _, course in rec.courses.iterrows():
            skills = list(course["skill_list"])
            if not skills:
                continue
            keep = max(1, int(round(len(skills) * 0.5)))
            kept = [str(s) for s in rng.choice(skills, size=keep, replace=False)]
            profile = StudentProfile(
                name="synthetic", skills=kept, skill_level=course["difficulty"],
                interests=[course["category"]] if scenario.startswith("B") else [],
                career_goal=course["career_domain"] if scenario.startswith("B") else NOT_SURE,
            )
            top = rec.recommend(profile, explain=False).head(k)
            ids = top["course_id"].tolist()
            hit = course["course_id"] in ids
            hits.append(1.0 if hit else 0.0)
            rr.append(1.0 / (ids.index(course["course_id"]) + 1) if hit else 0.0)
            cat_prec.append(float((top["category"] == course["category"]).mean()))
        records.append({
            "Scenario": scenario, "Profiles tested": len(hits), f"Hit Rate@{k}": round(float(np.mean(hits)), 3),
            "MRR": round(float(np.mean(rr)), 3), f"Category Precision@{k}": round(float(np.mean(cat_prec)), 3),
        })
    return pd.DataFrame(records)
