"""
utils/charts.py
---------------
Plotly figure builders. Every function returns None when there is nothing to plot
so the UI can show a friendly message instead of crashing.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import LEVELS

TEMPLATE = "plotly_white"
LEVEL_COLORS = {"Beginner": "#2ca02c", "Intermediate": "#ff9f1c", "Advanced": "#d62728"}
COMPONENT_LABELS = {
    "contrib_content_similarity": "Content similarity (TF-IDF)",
    "contrib_skill_match": "Skill match",
    "contrib_interest_match": "Interest match",
    "contrib_career_match": "Career match",
    "contrib_difficulty_fit": "Difficulty fit",
    "contrib_format_fit": "Format fit",
}


def _empty(df: Optional[pd.DataFrame]) -> bool:
    return df is None or len(df) == 0


def score_bar(recs: pd.DataFrame) -> Optional[go.Figure]:
    if _empty(recs):
        return None
    d = recs.sort_values("score")
    fig = px.bar(d, x="score", y="course_name", orientation="h", color="difficulty",
                 color_discrete_map=LEVEL_COLORS, range_x=[0, 100], template=TEMPLATE,
                 text=d["score"].round(0).astype(int).astype(str) + "%",
                 labels={"score": "Recommendation score (%)", "course_name": ""})
    fig.update_layout(height=max(320, 60 * len(d)), margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Difficulty")
    return fig


def breakdown_bar(recs: pd.DataFrame) -> Optional[go.Figure]:
    if _empty(recs):
        return None
    cols = list(COMPONENT_LABELS)
    d = recs[["course_name"] + cols].melt(id_vars="course_name", var_name="component", value_name="points")
    d["component"] = d["component"].map(COMPONENT_LABELS)
    fig = px.bar(d, x="points", y="course_name", color="component", orientation="h", template=TEMPLATE,
                 labels={"points": "Score points", "course_name": ""})
    fig.update_layout(height=max(320, 60 * len(recs)), margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(autorange="reversed"), legend_title_text="Component")
    return fig


def skill_gap_bar(missing: List[Tuple[str, int]]) -> Optional[go.Figure]:
    if not missing:
        return None
    d = pd.DataFrame(missing, columns=["skill", "courses"]).sort_values("courses")
    fig = px.bar(d, x="courses", y="skill", orientation="h", template=TEMPLATE,
                 labels={"courses": "Recommended courses requiring this skill", "skill": ""},
                 color_discrete_sequence=["#d62728"])
    fig.update_layout(height=max(300, 38 * len(d)), margin=dict(l=10, r=10, t=30, b=10))
    return fig


def gap_donut(n_existing: int, n_missing: int) -> Optional[go.Figure]:
    if n_existing + n_missing == 0:
        return None
    fig = go.Figure(go.Pie(labels=["Skills you have", "Skills to learn"], values=[n_existing, n_missing],
                           hole=0.55, marker=dict(colors=["#2ca02c", "#d62728"])))
    fig.update_layout(template=TEMPLATE, height=320, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def category_pie(df: pd.DataFrame, title: str = "") -> Optional[go.Figure]:
    if _empty(df):
        return None
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "courses"]
    fig = px.pie(counts, names="category", values="courses", hole=0.4, template=TEMPLATE, title=title)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def difficulty_bar(df: pd.DataFrame, title: str = "") -> Optional[go.Figure]:
    if _empty(df):
        return None
    counts = df["difficulty"].value_counts().reindex(LEVELS, fill_value=0).reset_index()
    counts.columns = ["difficulty", "courses"]
    fig = px.bar(counts, x="difficulty", y="courses", color="difficulty", color_discrete_map=LEVEL_COLORS,
                 template=TEMPLATE, title=title, text="courses")
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def platform_bar(df: pd.DataFrame, title: str = "") -> Optional[go.Figure]:
    if _empty(df):
        return None
    counts = df["platform"].value_counts().reset_index()
    counts.columns = ["platform", "courses"]
    fig = px.bar(counts, x="platform", y="courses", template=TEMPLATE, title=title, text="courses",
                 color_discrete_sequence=["#4c78a8"])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def rating_scatter(recs: pd.DataFrame) -> Optional[go.Figure]:
    if _empty(recs):
        return None
    fig = px.scatter(recs, x="rating", y="score", size="duration_hours", color="category",
                     hover_name="course_name", template=TEMPLATE,
                     labels={"rating": "Course rating", "score": "Recommendation score (%)"})
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def roadmap_chart(road: pd.DataFrame) -> Optional[go.Figure]:
    if _empty(road):
        return None
    d = road.copy()
    d["label"] = d["step"].astype(str) + ". " + d["course_name"]
    fig = px.bar(d, x="duration_hours", y="label", color="stage", orientation="h", template=TEMPLATE,
                 color_discrete_map=LEVEL_COLORS, text="duration_hours",
                 labels={"duration_hours": "Estimated hours", "label": ""})
    fig.update_layout(height=max(320, 50 * len(d)), yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Stage")
    return fig
