# Minor Project Documentation
**Title:** AI-Based Course Recommendation System for Students
**Department:** Computer Science & Engineering (AI/ML)

## Abstract
The rapid growth of online learning platforms has given students access to thousands of courses, yet choosing a course that matches their current knowledge, interests and career goals remains difficult. This project presents an AI-based course recommendation system that provides personalised, explainable suggestions through a web application built with Python and Streamlit. A student describes their programming languages, technical skills, skill level, areas of interest, career goal, weekly study time and learning preference. The system represents each course and the student profile as text documents, converts them into numerical vectors with Term Frequency–Inverse Document Frequency (TF-IDF), and measures relevance using cosine similarity. This content-based similarity is combined with skill overlap, interest match, career-goal match, difficulty compatibility and learning-format fit into a weighted hybrid score. Every recommendation is accompanied by matching skills and a natural-language explanation. Additional modules perform skill-gap analysis, generate a Beginner-to-Advanced learning roadmap and present interactive Plotly analytics. Because no labelled ground truth is available, the project describes an offline "hidden-course" simulation instead of reporting classification accuracy. The system uses a clearly labelled sample dataset of 112 illustrative courses across 14 domains and can be extended with real course data and user feedback.

**Keywords:** recommender system, content-based filtering, TF-IDF, cosine similarity, skill-gap analysis, Streamlit.

## Problem Statement
Students today can choose from thousands of online courses offered by many platforms. Course titles are often vague, prerequisites are unclear, and popularity or rating alone does not indicate suitability for a particular learner. Beginners may start courses that are too advanced, while experienced students may waste time on material they already know. Manual searching across platforms is time-consuming and leads to unfocused learning and dropped courses. There is a need for a system that understands a student's background and goals, recommends relevant courses, explains its choices and shows what to learn next.

## Objectives
1. To collect a student's skills, level, interests, career goal, study time and learning preference through a simple form.
2. To build a structured course dataset with at least 100 courses covering 14 technical domains.
3. To implement a content-based recommender using TF-IDF vectorisation and cosine similarity.
4. To improve ranking with a weighted hybrid score (skills, interests, career, difficulty and format).
5. To explain each recommendation so that students can trust and understand it.
6. To identify skill gaps and suggest courses that cover the missing skills.
7. To generate a personalised Beginner → Intermediate → Advanced learning roadmap.
8. To provide filters and interactive dashboards, and to describe an honest offline evaluation methodology.

## Scope
The system is designed for college students (diploma, B.Tech, BCA/B.Sc, M.Tech/MCA) who want structured, personalised learning paths in programming, AI/ML, data, web, mobile, security, cloud, DevOps, databases and design. It works without login or a database, so it can be used instantly. It supports self-assessment of skills, discovery of suitable courses, planning study time in weeks and preparing for internships and placements. The current version uses a sample catalogue but the architecture (CSV loader + vectoriser) can accept real course data.

## Future Scope
- **Collaborative filtering** using ratings and enrolment history from real students.
- **Hybrid recommendation system** combining content-based and collaborative signals.
- **Deep learning / sentence embeddings** (e.g., transformer models) to capture synonyms and semantics.
- **User accounts** to store profiles, progress and completed courses.
- **Course API integration** with learning platforms for up-to-date catalogues.
- **Real-time course availability**, pricing and enrolment status.
- **Student feedback loop** (likes, ratings, completions) to learn scoring weights and enable real evaluation.
- Prerequisite graphs and adaptive roadmaps that update as skills improve.

## Viva Preparation – Likely Questions
| Question | Short answer |
|---|---|
| Why content-based filtering? | No user history (cold start); needs only text descriptions; explainable. |
| What is TF-IDF? | Weights a term by its frequency in a document times its rarity across documents: `tf × idf`. |
| Why cosine similarity? | Compares the *direction* of vectors, so document length does not dominate. |
| How did you weight fields? | By repeating text (skills ×3, name/category/career ×2) before vectorising. |
| How is the final score computed? | `100 × Σ weight × component`; weights 0.40/0.15/0.15/0.10/0.15/0.05. |
| Why are the weights not learned? | No labelled feedback data; they are documented design choices. |
| Why no accuracy? | No ground-truth labels; classification accuracy is not meaningful for ranking without them. |
| How do you evaluate then? | Offline hidden-course simulation (Hit Rate@K, MRR, Category Precision@K); real evaluation needs user feedback. |
| What is the cold-start problem? | A recommender cannot personalise for new users/items with no history. |
| Is the dataset real? | No – a labelled sample with synthetic ratings/enrollments and search-page links. |
| What are the limitations? | Keyword matching (no deep semantics), sample data, hand-tuned weights. |
