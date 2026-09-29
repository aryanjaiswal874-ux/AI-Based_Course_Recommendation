"""
AI-Based Course Recommendation System for Students
Run with:  streamlit run app.py
"""
from __future__ import annotations

import itertools
import re

import pandas as pd
import streamlit as st

from models.recommender import COMPONENTS, CourseRecommender, StudentProfile, evaluate_offline
from utils import charts
from utils.helpers import (
    CAREER_GOALS, DATA_PATH, DOMAIN_BY_CAREER, EDUCATION_LEVELS, LANGUAGES, LEARNING_PREFS,
    LEVELS, NOT_SURE, SEMESTERS, SKILL_OPTIONS, DatasetError, apply_filters, estimate_weeks,
    format_int, load_courses, unique_preserve_order, validate_profile,
)

st.set_page_config(page_title="AI Course Recommender", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .hero {padding: 1.4rem 1.6rem; border-radius: 14px;
           background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%); color: white; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: .3rem 0 0 0; opacity: .92;}
    .arrow {text-align: center; font-size: 1.6rem; line-height: 1; color: #7c3aed; margin: .1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Model loading (cached) with graceful error handling
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading course dataset and fitting the TF-IDF model...")
def get_recommender(path: str, mtime: float) -> CourseRecommender:
    return CourseRecommender(load_courses(path))


def load_model() -> CourseRecommender:
    try:
        mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
        return get_recommender(str(DATA_PATH), mtime)
    except DatasetError as exc:
        st.error(f"⚠️ Dataset problem: {exc}")
    except Exception as exc:  # never let the app crash on load
        st.error(f"⚠️ Unexpected error while loading the recommender: {exc}")
    st.stop()


model = load_model()
courses_df = model.courses


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def chips(items) -> str:
    return " ".join(f"`{i}`" for i in items) if items else "_none_"


def course_link(url: str, label: str = "Open course ↗") -> str:
    return f"[{label}]({url})" if url else "_URL not available_"


def default_category(profile: StudentProfile, ranked: pd.DataFrame) -> str:
    if profile.career_goal in DOMAIN_BY_CAREER:
        return DOMAIN_BY_CAREER[profile.career_goal]
    for interest in profile.interests:
        if interest in set(courses_df["category"]):
            return interest
    return ranked.iloc[0]["category"]


def render_card(row: pd.Series, profile: StudentProfile, rank: int) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f"### {rank}. {row['course_name']}")
            st.caption(f"{row['category']} • {row['difficulty']} • {row['duration_hours']:.0f} hrs • "
                       f"{row['platform']} • ⭐ {row['rating']:.1f}")
        right.metric("Match", f"{row['score']:.0f}%")
        st.progress(min(max(row["score"] / 100, 0.0), 1.0), text=f"Recommendation Score: {row['score']:.0f}%")
        st.markdown(f"**Matching skills:** {chips(row['matching_skills']) if row['matching_skills'] else '_none yet — this course introduces new skills_'}")
        st.markdown(f"> 💡 **Why recommended:** {row['reason']}")
        with st.expander("Course details"):
            st.write(row["description"])
            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**Prerequisites**  \n{row['prerequisites'] or 'None'}")
            d2.markdown(f"**Career domain**  \n{row['career_domain']}")
            d3.markdown(f"**Enrollments (sample)**  \n{format_int(row['enrollments'])}")
            st.markdown(f"**All skills:** {chips(row['skill_list'])}")
            weeks = estimate_weeks(row["duration_hours"], profile.weekly_hours)
            st.markdown(f"⏱️ About **{weeks} week(s)** at {profile.weekly_hours} hrs/week • Format: {row['learning_format']}")
            breakdown = pd.DataFrame({
                "Component": [charts.COMPONENT_LABELS[f"contrib_{c}"] for c in COMPONENTS],
                "Points": [round(row[f"contrib_{c}"], 1) for c in COMPONENTS],
            })
            st.dataframe(breakdown, hide_index=True)
        st.markdown(course_link(row["course_url"]))


_chart_ids = itertools.count()  # reset on every rerun -> stable, unique widget keys


def show_chart(fig, empty_msg: str = "Not enough data to draw this chart.") -> None:
    if fig is None:
        st.info(empty_msg)
    else:
        st.plotly_chart(fig, key=f"chart_{next(_chart_ids)}")


def need_profile() -> None:
    st.info("👈 Fill in your profile in the sidebar and click **Generate Recommendations** to see this section.")


# --------------------------------------------------------------------------- #
# Sidebar: student form + filters
# --------------------------------------------------------------------------- #
st.sidebar.title("🎓 Student Information")
with st.sidebar.form("profile_form"):
    name = st.text_input("Name", placeholder="e.g. Aaru")
    education = st.selectbox("Education level", EDUCATION_LEVELS, index=1)
    semester = st.selectbox("Current semester / year", SEMESTERS, index=4)
    languages = st.multiselect("Programming languages known", LANGUAGES)
    skills_sel = st.multiselect("Existing technical skills", SKILL_OPTIONS)
    skills_extra = st.text_input("Other skills (comma separated)", placeholder="e.g. Pandas, Flask, Kotlin")
    skill_level = st.selectbox("Skill level", LEVELS)
    interests = st.multiselect("Areas of interest", sorted(courses_df["category"].unique()))
    career = st.selectbox("Preferred career / domain", CAREER_GOALS)
    weekly_hours = st.slider("Weekly study hours", 1, 40, 8)
    learning_pref = st.selectbox("Learning preference", LEARNING_PREFS, index=3)
    submitted = st.form_submit_button("🚀 Generate Recommendations", type="primary", width="stretch")

if submitted:
    extra = [s.strip() for s in re.split(r"[,;\n]", skills_extra or "") if s.strip()]
    new_profile = StudentProfile(
        name=(name or "").strip()[:50] or "Student",
        education_level=education, semester=semester,
        languages=unique_preserve_order(languages),
        skills=unique_preserve_order(list(skills_sel) + extra),
        skill_level=skill_level, interests=list(interests), career_goal=career,
        weekly_hours=int(weekly_hours), learning_preference=learning_pref,
    )
    problems = validate_profile(new_profile)
    if problems:
        for p in problems:
            st.sidebar.error(p)
    else:
        st.session_state["profile"] = new_profile
        st.sidebar.success("Profile saved!")

st.sidebar.title("🎛️ Filters")
st.sidebar.caption("Leave a filter empty to include everything.")
f_domains = st.sidebar.multiselect("Domain", sorted(courses_df["category"].unique()))
f_levels = st.sidebar.multiselect("Difficulty", LEVELS)
min_h, max_h = int(courses_df["duration_hours"].min()), int(courses_df["duration_hours"].max())
f_hours = st.sidebar.slider("Max duration (hours)", min_h, max_h, max_h) if min_h < max_h else max_h
f_rating = st.sidebar.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)
f_platforms = st.sidebar.multiselect("Platform", sorted(courses_df["platform"].unique()))
top_n = st.sidebar.slider("Number of recommendations", 5, 10, 6)

# --------------------------------------------------------------------------- #
# Recommendation pipeline (runs only when a profile exists)
# --------------------------------------------------------------------------- #
profile: StudentProfile | None = st.session_state.get("profile")
ranked = filtered = top = pd.DataFrame()
gap = {"existing": [], "missing": [], "path": pd.DataFrame()}
if profile is not None:
    try:
        ranked = model.recommend(profile)
        filtered = apply_filters(ranked, f_domains, f_levels, f_hours, f_rating, f_platforms)
        top = filtered.head(top_n)
        gap = model.skill_gap_analysis(profile, top)
    except Exception as exc:
        st.error(f"Something went wrong while generating recommendations: {exc}")
        st.stop()

# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
st.markdown(
    """<div class="hero"><h1>🎓 AI-Based Course Recommendation System</h1>
    <p>Content-based recommendations using TF-IDF and cosine similarity, with skill-gap analysis and learning roadmaps.</p></div>""",
    unsafe_allow_html=True,
)

tab_home, tab_profile, tab_recs, tab_gap, tab_road, tab_stats, tab_about = st.tabs(
    ["🏠 Home", "👤 Student Profile", "🎯 Recommendations", "📊 Skill Gap",
     "🗺️ Learning Roadmap", "📈 Analytics", "ℹ️ About Project"]
)

# ------------------------------- Home / dashboard --------------------------- #
with tab_home:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Courses in dataset", len(courses_df))
    c2.metric("Domains", courses_df["category"].nunique())
    c3.metric("Platforms", courses_df["platform"].nunique())
    c4.metric("Skills covered", len({s for xs in courses_df["skill_list"] for s in xs}))
    st.caption("⚠️ The dataset is a clearly labelled **sample** of illustrative courses. Ratings and enrollments are synthetic; "
               "links open platform search pages. No course is claimed to be official or certified.")

    if profile is None:
        st.subheader("How it works")
        st.markdown(
            "1. **Enter your profile** in the sidebar (skills, level, interests, career goal).\n"
            "2. The system converts your profile and every course into **TF-IDF vectors**.\n"
            "3. **Cosine similarity** plus skill, interest, career, difficulty and format signals produce a score.\n"
            "4. Explore **recommendations, skill gaps, a roadmap and analytics** in the tabs above."
        )
        need_profile()
    else:
        st.subheader(f"Dashboard for {profile.name}")
        if top.empty:
            st.warning("No courses match your current filters. Relax the filters in the sidebar.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Best match", f"{top.iloc[0]['score']:.0f}%")
            k2.metric("Skills you have (in top picks)", len(gap["existing"]))
            k3.metric("Skills to learn", len(gap["missing"]))
            k4.metric("Skill level", profile.skill_level)
            a, b = st.columns([3, 2])
            with a:
                st.markdown("**Top recommendations**")
                show_chart(charts.score_bar(top))
            with b:
                st.markdown("**Skill balance**")
                show_chart(charts.gap_donut(len(gap["existing"]), len(gap["missing"])), "No skills to compare yet.")
            st.markdown("**Roadmap preview**")
            preview = model.build_roadmap(profile, default_category(profile, ranked))
            if preview.empty:
                st.info("No roadmap available for this domain.")
            else:
                st.markdown(" ➜ ".join(preview["course_name"].tolist()))

# ------------------------------- Student profile ---------------------------- #
with tab_profile:
    if profile is None:
        need_profile()
    else:
        st.subheader("Profile summary")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Name", profile.name)
        p2.metric("Education", profile.education_level)
        p3.metric("Semester", profile.semester)
        p4.metric("Weekly study hours", profile.weekly_hours)
        st.markdown(f"**Skill level:** {profile.skill_level}  \n"
                    f"**Career goal:** {profile.career_goal}  \n"
                    f"**Learning preference:** {profile.learning_preference}")
        st.markdown(f"**Languages:** {chips(profile.languages)}")
        st.markdown(f"**Skills:** {chips(profile.skills)}")
        st.markdown(f"**Interests:** {chips(profile.interests)}")
        filled = sum(bool(x) for x in [profile.languages, profile.skills, profile.interests,
                                       profile.career_goal != NOT_SURE, profile.name != "Student"])
        st.progress(filled / 5, text=f"Profile completeness: {filled}/5 — a richer profile gives better recommendations")
        with st.expander("Text document the model builds from your profile (TF-IDF input)"):
            st.code(profile.to_text() or "(empty)")

# ------------------------------- Recommendations ---------------------------- #
with tab_recs:
    if profile is None:
        need_profile()
    elif top.empty:
        st.warning("No courses match your filters. Try clearing some filters in the sidebar.")
    else:
        st.subheader(f"Top {len(top)} courses for {profile.name}")
        if f_domains or f_levels or f_platforms or f_rating or f_hours < max_h:
            st.caption(f"Filters applied: {len(filtered)} of {len(ranked)} courses remain.")
        for position, (_, row) in enumerate(top.iterrows(), start=1):
            render_card(row, profile, position)
        export = top[["course_name", "category", "difficulty", "duration_hours", "platform", "rating",
                      "score", "reason", "course_url"]]
        st.download_button("⬇️ Download recommendations (CSV)", export.to_csv(index=False).encode("utf-8"),
                           file_name="recommendations.csv", mime="text/csv")

# ------------------------------- Skill gap ------------------------------------ #
with tab_gap:
    if profile is None:
        need_profile()
    elif top.empty:
        st.warning("No recommended courses available, so no skill gap can be computed.")
    else:
        st.subheader("Skill Gap Analysis")
        st.caption("Compares your skills with the skills taught by your top recommended courses.")
        e_col, m_col = st.columns(2)
        with e_col:
            st.markdown("#### ✅ Skills you already have")
            st.markdown(chips(gap["existing"]) if gap["existing"] else "_None of the recommended courses' skills match yet._")
        with m_col:
            st.markdown("#### 🎯 Skills you need to learn")
            st.markdown(chips([m[0] for m in gap["missing"]]) if gap["missing"] else "_Great — nothing missing!_")
        g1, g2 = st.columns([3, 2])
        with g1:
            show_chart(charts.skill_gap_bar(gap["missing"]), "No missing skills to chart.")
        with g2:
            show_chart(charts.gap_donut(len(gap["existing"]), len(gap["missing"])))
        st.markdown("#### 📚 Recommended learning path for the missing skills")
        path = gap["path"]
        if path.empty:
            st.info("No matching courses found for the missing skills.")
        else:
            for i, (_, r) in enumerate(path.iterrows(), start=1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {r['course_name']}** — {r['difficulty']} • {r['duration_hours']:.0f} hrs • {r['platform']}")
                    st.caption(f"Covers: {r['covers_missing_skills']}")
                    st.markdown(course_link(r["course_url"]))

# ------------------------------- Roadmap ---------------------------------------- #
with tab_road:
    if profile is None:
        need_profile()
    else:
        st.subheader("Personalised Learning Roadmap")
        categories = sorted(courses_df["category"].unique())
        start = default_category(profile, ranked)
        chosen = st.selectbox("Roadmap domain", categories, index=categories.index(start) if start in categories else 0)
        road = model.build_roadmap(profile, chosen)
        if road.empty:
            st.info("No courses available for this domain.")
        else:
            total_h = float(road["duration_hours"].sum())
            r1, r2, r3 = st.columns(3)
            r1.metric("Steps", len(road))
            r2.metric("Total hours", f"{total_h:.0f}")
            r3.metric("Estimated weeks", estimate_weeks(total_h, profile.weekly_hours))
            for stage, icon in zip(LEVELS, ["🟢", "🟠", "🔴"]):
                stage_df = road[road["stage"] == stage]
                if stage_df.empty:
                    continue
                st.markdown(f"#### {icon} {stage}")
                for _, r in stage_df.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**Step {int(r['step'])}: {r['course_name']}**")
                        st.caption(f"{r['duration_hours']:.0f} hrs • {r['platform']}" + (f" • {r['note']}" if r["note"] else ""))
                        st.markdown(course_link(r["course_url"]))
                    if int(r["step"]) < len(road):
                        st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
            st.markdown("**Time per step**")
            show_chart(charts.roadmap_chart(road))

# ------------------------------- Analytics ---------------------------------------- #
with tab_stats:
    st.subheader("Dataset analytics")
    a1, a2 = st.columns(2)
    with a1:
        show_chart(charts.category_pie(courses_df, "Course category distribution (dataset)"))
    with a2:
        show_chart(charts.difficulty_bar(courses_df, "Difficulty distribution (dataset)"))
    show_chart(charts.platform_bar(courses_df, "Courses per platform (dataset)"))
    if profile is not None and not top.empty:
        st.subheader("Your recommendations")
        b1, b2 = st.columns(2)
        with b1:
            show_chart(charts.category_pie(top, "Categories in your recommendations"))
        with b2:
            show_chart(charts.difficulty_bar(top, "Difficulty of your recommendations"))
        st.markdown("**Recommendation scores**")
        show_chart(charts.score_bar(top))
        st.markdown("**How each score is built (points per component)**")
        show_chart(charts.breakdown_bar(top))
        st.markdown("**Rating vs. score**")
        show_chart(charts.rating_scatter(top))
    else:
        st.info("Generate recommendations to see charts about your personal results.")

# ------------------------------- About ----------------------------------------------- #
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Goal:** help students find suitable online courses from a large catalogue and explain *why* each course is suggested.

**Why content-based filtering?** There is no login and no interaction history, so collaborative filtering
would suffer from the *cold-start problem*. Content-based filtering only needs a description of the student
and of the courses, and it is easy to explain.

**Pipeline**
1. *Feature engineering* – each course becomes one text document (name, category, skills, career domain,
   description, prerequisites). Important fields are repeated to weigh them higher.
2. *TF-IDF* – `TfidfVectorizer` gives each term a weight that is high when the term is frequent in a course
   but rare in the catalogue: `tf-idf(t, d) = tf(t, d) × idf(t)`.
3. *Cosine similarity* – the student profile is vectorised in the same space and compared with each course:
   `cos(θ) = (A · B) / (‖A‖ ‖B‖)`.
4. *Hybrid scoring* – `score = 100 × Σ weight × component`, with the weights below.
5. *Explanation* – matching skills, shared TF-IDF terms and level fit are turned into a sentence.
        """
    )
    weights = pd.DataFrame({
        "Component": [charts.COMPONENT_LABELS[f"contrib_{c}"] for c in COMPONENTS],
        "Weight": [CourseRecommender.WEIGHTS[c] for c in COMPONENTS],
        "Meaning": [
            "Cosine similarity of profile and course TF-IDF vectors (rescaled so the best match = 1)",
            "Share of the course's skills the student already has",
            "1 if the course category is one of the student's interests",
            "1 if the course targets the student's career goal",
            "1.0 same level, 0.6 one level up, 0.4 one level down, 0.1 otherwise",
            "1 if the delivery format matches the learning preference (0.5 for 'Mixed')",
        ],
    })
    st.dataframe(weights, hide_index=True)
    st.caption("Courses whose skills the student already knows (≥ 80%) are multiplied by 0.7 because there is little new to learn.")

    st.markdown("### Evaluation methodology")
    st.markdown(
        """
There is **no labelled ground truth** (no real students who rated courses), so classification accuracy,
precision or recall against true labels are **not meaningful** here and no accuracy number is claimed.
Instead, an **offline "hidden course" simulation** is provided: for every course a synthetic student is created
who knows a random half of that course's skills. We check whether the hidden course appears in the top-K
(Hit Rate@K, MRR) and how many top-K results share its category (Category Precision@K).
These figures measure retrieval consistency on synthetic profiles, **not** real user satisfaction.
Real evaluation would need user feedback (ratings, clicks, completions).
        """
    )
    k_val = st.select_slider("K", options=[3, 5, 10], value=10)
    if st.button("▶️ Run offline evaluation"):
        with st.spinner("Simulating profiles..."):
            try:
                st.dataframe(evaluate_offline(model, k=int(k_val)), hide_index=True)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    st.markdown("### Limitations")
    st.markdown(
        "- Sample dataset with synthetic ratings/enrollments.\n"
        "- Text matching cannot understand deep semantics (synonyms) beyond shared words.\n"
        "- No user feedback loop yet; weights are hand-tuned, not learned."
    )
