import streamlit as st
import pdfplumber
import re
import pandas as pd
from rapidfuzz import fuzz

# -------------------------------
# Resume Text Extraction
# -------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# -------------------------------
# Resume Parsing (Simple NLP)
# -------------------------------
def parse_resume(text):
    skills_list = ["python", "java", "sql", "excel", "machine learning",
                   "deep learning", "react", "node", "php", "cloud",
                   "aws", "azure", "docker", "kubernetes", "nlp"]
    found_skills = [s for s in skills_list if s.lower() in text.lower()]
    return {
        "skills": found_skills
    }

# -------------------------------
# JD Parsing
# -------------------------------
def parse_jd(jd_text):
    skills_list = ["python", "java", "sql", "excel", "machine learning",
                   "deep learning", "react", "node", "php", "cloud",
                   "aws", "azure", "docker", "kubernetes", "nlp"]
    required_skills = [s for s in skills_list if s.lower() in jd_text.lower()]
    return {"skills_required": required_skills}

# -------------------------------
# Matching Resume to JD
# -------------------------------
def match_resume_to_jd(resume_text, jd_text):
    return fuzz.token_set_ratio(resume_text.lower(), jd_text.lower())

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Resume Parser", layout="wide")

st.title("📄 AI Resume Parser & Ranking Tool")

uploaded_files = st.file_uploader("📂 Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)
jd_text = st.text_area("📝 Paste Job Description Here")

# Add Submit Button
if uploaded_files and jd_text:
    if st.button("🚀 Submit for Processing"):
        jd_parsed = parse_jd(jd_text)
        st.subheader("🔎 Job Description Parsed Skills")
        st.write(jd_parsed)

        results = []
        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            parsed = parse_resume(resume_text)
            score = match_resume_to_jd(resume_text, jd_text)

            # coverage check
            matched = [s for s in parsed["skills"] if s in jd_parsed["skills_required"]]
            missing = [s for s in jd_parsed["skills_required"] if s not in parsed["skills"]]

            results.append({
                "name": file.name,
                "score": score,
                "skills": parsed["skills"],
                "matched": matched,
                "missing": missing
            })

        # Rank candidates
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.subheader("📊 Candidate Ranking")
        for r in results:
            st.markdown(f"**{r['name']}** — Match Score: `{r['score']}`")
            st.write("✅ Matched Skills:", r["matched"])
            st.write("❌ Missing Skills:", r["missing"])
            st.write("---")

        # --- Export Top 5 Resumes ---
        top_n = min(5, len(results))
        top_candidates = results[:top_n]

        df = pd.DataFrame([{
            "File Name": r["name"],
            "Score": r["score"],
            "Matched Skills": ", ".join(r["matched"]),
            "Missing Skills": ", ".join(r["missing"]),
            "All Extracted Skills": ", ".join(r["skills"])
        } for r in top_candidates])

        st.subheader("⬇️ Export Top Candidates")
        st.dataframe(df)

        # CSV Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Top Candidates as CSV",
            data=csv,
            file_name="top_candidates.csv",
            mime="text/csv"
        )

        # Excel Download
        excel_file = "top_candidates.xlsx"
        df.to_excel(excel_file, index=False)
        with open(excel_file, "rb") as f:
            st.download_button(
                label="Download Top Candidates as Excel",
                data=f,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
