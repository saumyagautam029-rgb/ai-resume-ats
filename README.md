# AI Resume ATS System

Resume screening system using TF-IDF semantic matching and cosine similarity.

## How It Works
- Extracts keywords from resume and job description
- Calculates TF-IDF vectors for semantic comparison
- Scores 0-100 based on 60% semantic similarity + 40% keyword overlap
- Provides actionable feedback on missing skills

## Tech Stack
- Streamlit
- scikit-learn (TfidfVectorizer, cosine_similarity)
- SQLite
- NLP keyword extraction

## Live Demo
[Your URL]

## Tested On
- Data Scientist job descriptions from LinkedIn
- Weak resume: 16.67/100 | Keyword-rich resume: 41.81/100
