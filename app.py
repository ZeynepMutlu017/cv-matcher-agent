import os
import json
import pickle
import numpy as np
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview"
)

CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

with open("cv_matcher_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


def get_embedding(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding)


def predict_job_role(cv_text):
    embedding = get_embedding(cv_text)
    X = embedding.reshape(1, -1)
    prediction = model.predict(X)[0]
    predicted_role = le.inverse_transform([prediction])[0]
    return predicted_role


def analyze_cv(cv_text, job_text, predicted_role):
    prompt = f"""
Analyze the CV and job description. ML model predicted the candidate's closest job role as: {predicted_role}

CV:
{cv_text}

Job Description:
{job_text}

Respond in the following JSON format only:
{{
  "match_score": <0-100 integer>,
  "matched_skills": ["skill 1", "skill 2"],
  "missing_skills": ["skill 1", "skill 2"],
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["suggestion 1", "suggestion 2"],
  "summary": "3-4 sentence overall evaluation"
}}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a career advisor. Return only JSON, no extra text."},
            {"role": "user", "content": prompt}
        ]
    )
    raw = response.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
# Streamlit UI
st.set_page_config(page_title="CV Matcher", layout="wide")
st.title("CV & Job Description Matcher")
st.markdown("Powered by **Azure OpenAI** + **Machine Learning**")

col1, col2 = st.columns(2)

with col1:
    cv_text = st.text_area("Paste your CV here", height=400, placeholder="Enter your CV")
with col2:
    job_text = st.text_area("Paste the job description here", height=400, placeholder="Enter job description")

if st.button("Analyze", use_container_width=True):
    if not cv_text or not job_text:
        st.warning("Please fill in both fields.")
    else:
        with st.spinner("Running analysis..."):
            predicted_role = predict_job_role(cv_text)
            analysis = analyze_cv(cv_text, job_text, predicted_role)

        st.divider()

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("ML Predicted Role", predicted_role)
        with col4:
            st.metric("Match Score", f"{analysis['match_score']}/100")
        with col5:
            score = analysis['match_score']
            if score >= 75:
                st.metric("Status", "Strong Match")
            elif score >= 50:
                st.metric("Status", "Moderate Match")
            else:
                st.metric("Status", "Weak Match")

        col6, col7 = st.columns(2)
        with col6:
            st.subheader("Matched Skills")
            for skill in analysis['matched_skills']:
                st.success(skill)
            st.subheader("Strengths")
            for s in analysis['strengths']:
                st.info(s)

        with col7:
            st.subheader("Missing Skills")
            for skill in analysis['missing_skills']:
                st.error(skill)
            st.subheader("Improvement Suggestions")
            for i in analysis['improvements']:
                st.warning(i)

        st.divider()
        st.subheader("Summary")
        st.write(analysis['summary'])                    