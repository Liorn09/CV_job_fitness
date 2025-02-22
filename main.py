import streamlit as st
import requests

API_URL = "http://localhost:8000/upload/"

st.title("📄 AI-Powered Resume Screening")

cv_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if cv_file is not None and job_desc_file is not None:
    st.write("Uploading...")

    # Save files temporarily
    with open("temp_cv.pdf", "wb") as f:
        f.write(cv_file.getbuffer())

    with open("temp_job_desc.pdf", "wb") as f:
        f.write(job_desc_file.getbuffer())

    # Send both files to FastAPI backend
    with open("temp_cv.pdf", "rb") as f_cv, open("temp_job_desc.pdf",
                                                 "rb") as f_job:
        files = {"cv": f_cv, "job_desc": f_job}
        response = requests.post(API_URL, files=files)

    # Show match score
    if response.status_code == 200:
        result = response.json()
        st.success(f"✅ Match Score: {result['fitness score']}%")
        st.write(f"📄 Name: {result['candidate_name']}")
        st.write(f"📧 Email: {result['candidate_email']}")
        st.write(f"📚 Status: {result['candidate_fitness']}")
    else:
        st.error("⚠️ Failed to process the documents. Try again.")
