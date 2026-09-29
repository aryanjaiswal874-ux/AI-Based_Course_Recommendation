# 🎓 AI-Based Course Recommendation System for Students

An AI/ML minor project (CSE) built with **Python + Streamlit**. It recommends online courses to a student
using a **content-based recommender (TF-IDF + cosine similarity)**, explains *why* each course was chosen,
shows a **skill-gap analysis**, generates a **learning roadmap** and presents everything in an interactive dashboard.

> ⚠️ **Dataset notice:** `data/courses.csv` is a clearly labelled **sample dataset** of 112 illustrative courses.
> Ratings and enrollments are synthetic, links are platform search/landing pages, and no course is claimed to be
> official, certified or affiliated with any platform.

## Project overview
Students face thousands of courses across many platforms. This system takes a student profile (skills, level,
interests, career goal, study time, learning preference) and ranks the sample catalogue by relevance, with a
transparent score breakdown so the result can be explained in a viva.

## Problem statement
Choosing the right course is hard: catalogues are huge, course titles are vague, prerequisites are unclear and
popularity does not mean suitability. A wrong choice wastes time. A personalised, explainable recommender helps
students pick courses that fit their current level and goals.

## Objectives
1. Build a student-profile form that captures skills, level, interests and goals.
2. Create a structured course dataset (100+ courses, 14 domains).
3. Implement a content-based recommender using TF-IDF and cosine similarity.
4. Combine similarity with skill, interest, career, difficulty and format signals.
5. Explain every recommendation in plain language.
6. Provide skill-gap analysis and a personalised roadmap.
7. Offer filters and an interactive Plotly dashboard.
8. Describe an honest offline evaluation methodology.

## Features
- Sidebar student form + live filters (domain, difficulty, duration, rating, platform)
- Top 5–10 recommendations as cards: score progress bar, matching skills, reason, URL, expandable details
- Skill Gap Analysis: skills you have / need, and a learning path covering missing skills
- Beginner → Intermediate → Advanced roadmap (adds a Python foundation step for AI/ML/DS if needed)
- Analytics: category, difficulty and platform distributions, score breakdown, rating vs score
- Robust error handling: empty input, missing/invalid dataset, no matches, missing URLs
- CSV export of recommendations
- Tabs: Home, Student Profile, Recommendations, Skill Gap, Learning Roadmap, Analytics, About Project

## Technologies used
Python 3.9+, Streamlit, Pandas, NumPy, Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), Plotly.

## Machine-learning methodology
**Why content-based?** No login or interaction history exists, so collaborative filtering would face the
*cold-start* problem. Content-based filtering needs only descriptions of students and courses and is explainable.

**1. Feature engineering.** Each course becomes one text document from name (×2), category (×2), skills (×3),
career domain (×2), description (×1) and prerequisites (×1). Repetition acts as a field weight. Special tokens
are normalised (`C++ → cpp`, `Node.js → nodejs`) so they survive tokenisation.

**2. TF-IDF.** `TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)` converts documents to
vectors. `tf-idf(t, d) = tf(t, d) × idf(t)` – terms that are frequent in a course but rare in the catalogue get high weight.

**3. Cosine similarity.** The profile text (languages ×2, skills ×3, interests ×2, career ×2) is transformed into the
same vector space; `cos(θ) = A·B / (‖A‖‖B‖)` measures similarity. It is rescaled by the maximum so the best match = 1.

**4. Hybrid scoring.** `score = 100 × Σ wᵢ · componentᵢ`

| Component | Weight | Definition |
|---|---|---|
| Content similarity | 0.40 | Rescaled TF-IDF cosine similarity |
| Skill match | 0.15 | Share of course skills the student already has |
| Interest match | 0.15 | Course category ∈ student interests |
| Career match | 0.10 | Course career domain = selected career goal |
| Difficulty fit | 0.15 | 1.0 same level, 0.6 one up, 0.4 one down, 0.1 otherwise |
| Format fit | 0.05 | Course format = learning preference (0.5 for "Mixed") |

Courses whose skills the student already knows (≥ 80%) are multiplied by 0.7. Weights are hand-tuned design choices,
not learned parameters.

**5. Explainability.** Matching skills, the strongest shared TF-IDF terms and difficulty fit are converted into a sentence.

**6. Evaluation.** No labelled ground truth exists, so accuracy/precision/recall against true labels are not meaningful and
**no accuracy figure is claimed**. The About tab has an offline *hidden-course* simulation: a synthetic student knowing a random
half of a course's skills should retrieve that course in the top-K. Reported metrics are Hit Rate@K, MRR and Category
Precision@K, in two scenarios (skills only; skills + interest + career). They show retrieval consistency, not user satisfaction.
Real evaluation would require user ratings, clicks or completions.

## Dataset description
`data/courses.csv` – 112 rows, 8 per domain, ordered from foundational to advanced (`path_order`).

| Column | Description |
|---|---|
| course_id, course_name, category, description | Identification and text |
| skills, prerequisites, career_domain | `;`-separated skills, prerequisite text, target career |
| difficulty, duration_hours | Beginner/Intermediate/Advanced, estimated hours |
| platform, rating, enrollments | Platform name; **synthetic** rating (4.0–4.9) and enrollments |
| course_url | Platform search/landing link (not a pinned course page) |
| learning_format, path_order, data_source | Delivery format, order in domain roadmap, sample-data label |

Regenerate with `python data/generate_courses.py`.

## System architecture
```
Student form (sidebar) ──► StudentProfile ──► profile text ──┐
                                                             ▼
courses.csv ──► load_courses() ──► course text ──► TF-IDF vectors ──► cosine similarity
                                                             │
                     skill / interest / career / difficulty / format signals
                                                             ▼
                              weighted score ──► ranking ──► filters ──► top-N
                                                             │
        cards + reasons  |  skill-gap analysis  |  roadmap  |  Plotly analytics
```

## Project structure
```
course_recommendation_system/
├── app.py                    # Streamlit UI (tabs, sidebar, cards, charts)
├── data/
│   ├── courses.csv           # sample dataset (112 courses)
│   └── generate_courses.py   # regenerates the dataset
├── models/
│   ├── __init__.py
│   └── recommender.py        # TF-IDF, cosine similarity, scoring, skill gap, roadmap, evaluation
├── utils/
│   ├── __init__.py
│   ├── helpers.py            # constants, text normalisation, dataset loading, filters
│   └── charts.py             # Plotly figures
├── docs/PROJECT_DOCUMENTATION.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation instructions
```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## How to run
```bash
streamlit run app.py
```
Open the URL shown in the terminal (usually http://localhost:8501), fill in the sidebar form and click **Generate Recommendations**.

## Screenshots
_Add your screenshots here after running the app._
- Home / dashboard: `docs/screenshots/home.png`
- Recommendations: `docs/screenshots/recommendations.png`
- Skill gap: `docs/screenshots/skill_gap.png`
- Roadmap: `docs/screenshots/roadmap.png`
- Analytics: `docs/screenshots/analytics.png`

## Future enhancements
Collaborative filtering, hybrid recommender, sentence-embedding / deep-learning text models, user accounts, live course API
integration, real-time availability, student feedback and rating loop, learned scoring weights.

## Limitations
Sample dataset with synthetic ratings; keyword-based similarity cannot capture synonyms; weights are hand-tuned; no real-user
evaluation; links are search pages, not verified course pages.

## Conclusion
The project demonstrates a complete, explainable content-based recommender: profile → TF-IDF vectors → cosine similarity →
hybrid score → explanations, skill gaps and roadmaps, wrapped in a clean Streamlit dashboard suitable for a CSE AI/ML minor project.
