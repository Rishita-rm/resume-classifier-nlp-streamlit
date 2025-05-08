import streamlit as st
import pandas as pd
import pickle
import re
import docx
import PyPDF2
import numpy as np
import os
import zipfile
from PIL import Image
import plotly.express as px

# ------------------- Load Models ------------------- #
def load_clf_from_dropbox():
    url = "https://www.dropbox.com/scl/fi/afrgzbojcbon1lai7geoo/clf.pkl?rlkey=oo532gq5qbvh6a4452mbwwk5b&st=9vmmq1xt&dl=1"  # 🔁 Replace with your real Dropbox link
    filename = "clf.pkl"
    if not os.path.exists(filename):
        st.info("📥 Downloading model from Dropbox...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        st.success("✅ Model downloaded successfully!")
    return pickle.load(open(filename, "rb"))

# Load TF-IDF and encoder from local files
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))
clf = load_clf_from_dropbox()

# ------------------- Utils ------------------- #
def clean_resume(txt):
    txt = re.sub(r"http\S+|www\S+|https\S+", '', txt)
    txt = re.sub(r'@\w+|#', '', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub('\s+', ' ', txt).strip()
    return txt

def handle_file_upload(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return "".join([p.extract_text() for p in PyPDF2.PdfReader(file).pages])
    elif ext == 'docx':
        return "\n".join([p.text for p in docx.Document(file).paragraphs])
    elif ext == 'txt':
        try:
            return file.read().decode("utf-8")
        except:
            return file.read().decode("latin-1")
    return ""

# ------------------- Sidebar ------------------- #
st.set_page_config(page_title="Resume Classifier", layout="wide")
st.sidebar.title("\U0001F4DA Resume Classifier")
st.sidebar.markdown("Made with ❤️ by Rishita Makkar")

menu = st.sidebar.radio("Go to", [
    "🏠 Home", "📄 Upload Resumes", "⚖️ Compare with JD", "📖 About", "💬 Feedback"
])

# ------------------- Home ------------------- #
if menu == "🏠 Home":
    st.title("AI-Powered Resume Classifier")
    st.write("Upload your resume to get predicted industry categories using NLP and Machine Learning.")
    st.markdown("This tool helps job seekers and recruiters match resumes to domains using AI.")

# ------------------- Upload Resumes ------------------- #
elif menu == "📄 Upload Resumes":
    st.title("📄 Upload & Analyze Resumes")
    show_raw = st.checkbox("📄 Show Extracted Text")
    uploaded_files = st.file_uploader("Upload your resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    resume_data = []
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"🔍 Analyzing `{file.name}`..."):
                text = handle_file_upload(file)
                cleaned = clean_resume(text)

                if not cleaned.strip():
                    st.warning(f"{file.name} appears empty or unreadable.")
                    continue

                vector = tfidf.transform([cleaned])
                probs = clf.predict_proba(vector)[0]
                top_idx = np.argmax(probs)
                category = le.inverse_transform([top_idx])[0]
                confidence = round(probs[top_idx] * 100, 2)

                resume_data.append({
                    "Filename": file.name,
                    "Predicted Category": category,
                    "Confidence (%)": confidence,
                    "Raw Text": text[:300] + "..." if show_raw else "Hidden"
                })

        df = pd.DataFrame(resume_data).sort_values("Confidence (%)", ascending=False)
        st.success(f"✅ Processed {len(df)} resume(s)")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 Confidence Comparison")
        fig = px.bar(df, x="Filename", y="Confidence (%)", color="Predicted Category", title="Resume Ranking")
        st.plotly_chart(fig)

# ------------------- Compare with JD ------------------- #
elif menu == "⚖️ Compare with JD":
    st.title("📝 Resume vs Job Description")
    jd_text = st.text_area("Paste Job Description")
    resume_file = st.file_uploader("Upload a single resume", type=["pdf", "docx", "txt"])

    if jd_text and resume_file:
        resume_txt = handle_file_upload(resume_file)
        resume_cleaned = clean_resume(resume_txt)
        jd_cleaned = clean_resume(jd_text)

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_cleaned, jd_cleaned])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        st.success(f"Match Score: {round(score * 100, 2)}%")
        if score >= 0.7:
            st.info("✅ Great match!")
        elif score >= 0.5:
            st.warning("⚠️ Moderate match. Some tweaking needed.")
        else:
            st.error("❌ Weak match. Consider rephrasing your resume.")

# ------------------- About ------------------- #
elif menu == "📖 About":
    st.title("👩‍💻 About This App")
    st.markdown("""
This AI-powered resume classifier helps:

* 🔍 Categorize resumes into industries.
* 📊 Rank multiple resumes by confidence.
* 🧠 Compare resumes with job descriptions.

**Built with:** Python, Streamlit, Scikit-learn, Plotly  
**Developer:** Rishita Makkar
""")

# ------------------- Feedback ------------------- #
elif menu == "💬 Feedback":
    st.title("💬 We Value Your Feedback")
    name = st.text_input("Name")
    comment = st.text_area("What's on your mind?")
    if st.button("Submit"):
        st.success("✅ Thanks for your feedback!")
