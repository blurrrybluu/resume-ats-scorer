import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    return text.lower().strip()

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc 
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 2]
    return list(set(keywords))

def calculate_ats_score(resume_text, job_desc):
    if not resume_text or not job_desc:
        return 0, [], []
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_desc)
    resume_kw = extract_keywords(resume_clean)
    job_kw = extract_keywords(job_clean)
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([resume_clean, job_clean])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        keyword_match = len(set(resume_kw) & set(job_kw)) / len(job_kw) if job_kw else 0
        score = int((similarity * 0.6 + keyword_match * 0.4) * 100)
        return max(0, min(100, score)), resume_kw, job_kw
    except:
        return 0, [], []

# ———— UI ————
st.set_page_config(page_title="Resume ATS Scorer", page_icon="Resume")
st.title("Resume ATS Scorer")
st.markdown("**Upload your resume (PDF) + Paste job description → Get ATS Score!**")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("**Upload Resume (PDF)**", type="pdf")

with col2:
    job_desc = st.text_area("**Paste Job Description**", height=200, 
                            placeholder="e.g., We are looking for a Python developer with Django, AWS...")

if resume_file and job_desc:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text_from_pdf(resume_file)
        if resume_text.strip():
            score, resume_kw, job_kw = calculate_ats_score(resume_text, job_desc)
            matched = set(resume_kw) & set(job_kw)
            missing = set(job_kw) - set(resume_kw)
            
            st.success(f"### **ATS Score: {score}/100**")
            
            if score >= 80:
                st.balloons()
                st.markdown("**Excellent!** Your resume is highly optimized.")
            elif score >= 60:
                st.markdown("**Good!** Minor improvements needed.")
            elif score >= 40:
                st.markdown("**Fair.** Add more job-specific keywords.")
            else:
                st.markdown("**Needs Work.** Rewrite with keywords from the job.")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Matched Keywords**: {len(matched)}")
                if matched:
                    st.write(", ".join(list(matched)[:10]))
            with col2:
                st.write(f"**Missing Keywords**: {len(missing)}")
                if missing:
                    st.write(", ".join(list(missing)[:10]))
        else:
            st.error("Could not read PDF. Try another file.")
else:
    st.info("Upload **resume** + paste **job description** to get started!")

st.markdown("---")
st.caption("Built with spaCy • scikit-learn • Streamlit | By **Ann Mariya**")