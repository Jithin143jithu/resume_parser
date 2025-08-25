import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import re

# Load embedding model (small & fast)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- Acquire: extract text from PDF ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# --- Yield: simple parsing (skills, education, experience) ---
def parse_resume(text):
    skills = re.findall(r"\b(Python|Java|C\+\+|SQL|React|AWS|Docker|Kubernetes)\b", text, re.I)
    education = re.findall(r"(B\.?Tech|M\.?Tech|BE|MBA|BSc|MSc)", text, re.I)
    experience_years = len(re.findall(r"\b20[0-9]{2}\b", text))  # rough proxy
    return {
        "skills": list(set([s.lower() for s in skills])),
        "education": education,
        "experience_years": experience_years
    }

def parse_jd(text):
    skills = re.findall(r"\b(Python|Java|C\+\+|SQL|React|AWS|Docker|Kubernetes)\b", text, re.I)
    return {"skills_required": list(set([s.lower() for s in skills]))}

# --- Yield: matching using embeddings ---
def match_resume_to_jd(resume_text, jd_text):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    score = util.cos_sim(emb_resume, emb_jd).item()
    return round(score, 3)

# --- Streamlit UI (Present) ---
st.title("📄 AI Resume Parser & Matcher")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
jd_text = st.text_area("Paste Job Description")

if uploaded_files and jd_text:
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
