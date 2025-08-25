import os
import re
import pdfplumber
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

STANDARD_SKILLS = {
    "python": ["python3", "py"],
    "javascript": ["js", "java script"],
    "react": ["react.js", "reactjs"],
    "php": ["php7", "php8"],
    "machine learning": ["ml", "ai", "deep learning"]
}

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_fields(text, filename):
    email = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = re.findall(r"\+?\d[\d -]{8,12}\d", text)

    first_line = text.split("\n")[0].strip()
    name = first_line if len(first_line.split()) <= 4 else os.path.splitext(filename)[0]

    skills_found = []
    for skill, variations in STANDARD_SKILLS.items():
        if re.search(skill, text, re.IGNORECASE):
            skills_found.append(skill)
        else:
            for var in variations:
                if re.search(var, text, re.IGNORECASE):
                    skills_found.append(skill)

    exp_match = re.findall(r"(\d+)\s+year", text.lower())
    experience = max([int(x) for x in exp_match], default=0)

    return {
        "filename": filename,
        "name": name,
        "email": email[0] if email else None,
        "phone": phone[0] if phone else None,
        "skills": list(set(skills_found)),
        "experience": experience
    }

def skill_match_score(candidate_skills, job_skills):
    if not candidate_skills:
        return 0
    score = 0
    for js in job_skills:
        for cs in candidate_skills:
            emb_sim = util.cos_sim(
                model.encode(js, convert_to_tensor=True),
                model.encode(cs, convert_to_tensor=True)
            ).item()
            fuzzy_sim = fuzz.token_set_ratio(js.lower(), cs.lower()) / 100
            score += max(emb_sim, fuzzy_sim)
    return score / len(job_skills)

def rank_candidates(uploaded_files, job_description):
    job_skills = [s.strip().lower() for s in job_description["skills"]]
    results = []

    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        fields = extract_fields(text, uploaded_file.name)

        skill_score = skill_match_score(fields["skills"], job_skills)
        exp_score = min(fields["experience"] / job_description["min_exp"], 1.0)
        edu_score = 1.0 if job_description.get("degree", "").lower() in text.lower() else 0.5

        final_score = (skill_score * 0.6) + (exp_score * 0.3) + (edu_score * 0.1)

        results.append({
            "filename": fields["filename"],
            "name": fields["name"],
            "email": fields["email"],
            "phone": fields["phone"],
            "skills": ", ".join(fields["skills"]),
            "experience": fields["experience"],
            "score": round(final_score * 100, 2)
        })

    df = pd.DataFrame(results).sort_values(by="score", ascending=False).head(5)
    return df

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Resume Parser ATS", layout="wide")
st.title("📄 Resume Parser & Candidate Ranking")

# Upload resumes
uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

# Job description inputs
st.subheader("Job Description")
job_skills = st.text_input("Required Skills (comma-separated)", "Python, React, Machine Learning")
min_exp = st.number_input("Minimum Experience (years)", min_value=0, value=3)
degree = st.text_input("Required Degree", "B.Tech")

if st.button("Process Resumes"):
    if uploaded_files:
        jd = {"skills": job_skills.split(","), "min_exp": min_exp, "degree": degree}
        df = rank_candidates(uploaded_files, jd)

        st.success("Top 5 Candidates Ranked ✅")
        st.dataframe(df)

        # Download as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Top 5 as CSV", csv, "top_5_candidates.csv", "text/csv")
    else:
        st.warning("Please upload at least one PDF resume.")
