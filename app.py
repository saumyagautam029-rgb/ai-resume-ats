import streamlit as st
import re
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.set_page_config(page_title="AI Resume ATS", layout="wide")

DB_PATH = "ats.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            job_title TEXT,
            ats_score REAL,
            date TEXT,
            time TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_scan(name, job, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    c.execute("INSERT INTO scans (candidate_name, job_title, ats_score, date, time) VALUES (?, ?, ?, ?, ?)",
              (name, job, score, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM scans ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text):
    common_skills = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker',
        'kubernetes', 'machine learning', 'deep learning', 'tensorflow', 'pytorch',
        'data analysis', 'excel', 'tableau', 'powerbi', 'git', 'linux', 'agile',
        'scrum', 'communication', 'leadership', 'project management', 'html', 'css',
        'flask', 'django', 'fastapi', 'mongodb', 'postgresql', 'redis', 'kafka',
        'spark', 'hadoop', 'pandas', 'numpy', 'scikit-learn', 'opencv', 'nlp',
        'rest api', 'graphql', 'microservices', 'ci/cd', 'jenkins', 'terraform'
    ]
    text_lower = text.lower()
    found = [skill for skill in common_skills if skill in text_lower]
    return found

def calculate_ats_score(resume_text, job_desc):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_desc)
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        similarity = 0.0
    
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_desc))
    
    if len(job_skills) > 0:
        match_ratio = len(resume_skills & job_skills) / len(job_skills)
    else:
        match_ratio = 0.0
    
    final_score = (similarity * 0.6 + match_ratio * 0.4) * 100
    return round(min(final_score, 100), 2), resume_skills, job_skills

def generate_feedback(resume_text, job_desc, score, resume_skills, job_skills):
    feedback = []
    missing = job_skills - resume_skills
    
    if score >= 80:
        feedback.append("Excellent match! Your resume aligns well with the job description.")
    elif score >= 60:
        feedback.append("Good match, but there is room for improvement.")
    elif score >= 40:
        feedback.append("Moderate match. Consider tailoring your resume more specifically.")
    else:
        feedback.append("Low match. Significant improvements recommended.")
    
    if missing:
        feedback.append(f"Missing keywords/skills: {', '.join(list(missing)[:10])}")
    
    sections = ['experience', 'education', 'skills', 'projects', 'certifications']
    resume_lower = resume_text.lower()
    missing_sections = [s for s in sections if s not in resume_lower]
    if missing_sections:
        feedback.append(f"Consider adding these sections: {', '.join(missing_sections)}")
    
    return feedback

st.title("AI Resume ATS System")
st.caption("Smart resume screening with semantic matching and ATS scoring")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    job_title = st.text_input("Job Title", placeholder="Senior Python Developer")
    job_desc = st.text_area("Paste Job Description", height=300, placeholder="Paste the full job description here...")

with col2:
    st.subheader("Candidate Resume")
    candidate_name = st.text_input("Candidate Name", placeholder="John Doe")
    resume_text = st.text_area("Paste Resume Text", height=300, placeholder="Paste the resume text here...")

if st.button("Analyze Resume", type="primary", use_container_width=True):
    if job_desc and resume_text and candidate_name and job_title:
        with st.spinner("Analyzing with AI..."):
            score, resume_skills, job_skills = calculate_ats_score(resume_text, job_desc)
            feedback = generate_feedback(resume_text, job_desc, score, resume_skills, job_skills)
            
            save_scan(candidate_name, job_title, score)
            
            st.divider()
            col_score, col_skills = st.columns([1, 2])
            
            with col_score:
                st.metric("ATS Score", f"{score}/100")
                if score >= 80:
                    st.success("Strong Match")
                elif score >= 60:
                    st.info("Good Match")
                elif score >= 40:
                    st.warning("Moderate Match")
                else:
                    st.error("Weak Match")
            
            with col_skills:
                st.subheader("Skills Analysis")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Resume Skills:")
                    st.caption(", ".join(resume_skills) if resume_skills else "None detected")
                with c2:
                    st.write("Job Skills:")
                    st.caption(", ".join(job_skills) if job_skills else "None detected")
            
            st.subheader("AI Feedback")
            for item in feedback:
                st.write(f"• {item}")
            
            st.subheader("Recommendations")
            st.write("1. Add quantifiable achievements (e.g., 'Increased revenue by 20%')")
            st.write("2. Use action verbs: Led, Developed, Implemented, Optimized")
            st.write("3. Mirror keywords from the job description naturally")
            st.write("4. Keep formatting simple for ATS parsing")
    else:
        st.error("Please fill in all fields.")

st.divider()
st.subheader("Scan History")

history = get_history()
if history:
    df = pd.DataFrame(history, columns=["ID", "Name", "Job", "Score", "Date", "Time"])
    st.dataframe(df.drop("ID", axis=1), use_container_width=True)
else:
    st.info("No scans yet.")
