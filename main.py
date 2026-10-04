import os
import json
import pickle
import numpy as np
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

# Modeli ve label encoder'ı yükle
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
    probabilities = model.predict_proba(X)[0]
    predicted_role = le.inverse_transform([prediction])[0]
    top3_idx = probabilities.argsort()[-3:][::-1]
    top3 = [(le.inverse_transform([i])[0], probabilities[i]) for i in top3_idx]
    return predicted_role, top3


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


def match_cv(cv_text, job_text):
    print("Running ML model...")
    predicted_role, top3 = predict_job_role(cv_text)

    print("Running GPT analysis...")
    analysis = analyze_cv(cv_text, job_text, predicted_role)

    print("\n" + "=" * 50)
    print(f"  ML PREDICTED ROLE: {predicted_role}")
    print(f"  MATCH SCORE: {analysis['match_score']}/100")
    print("=" * 50)
    print(f"\nMatched Skills: {', '.join(analysis['matched_skills'])}")
    print(f"Missing Skills: {', '.join(analysis['missing_skills'])}")
    print(f"\nStrengths:")
    for s in analysis['strengths']:
        print(f"   - {s}")
    print(f"\nImprovement Suggestions:")
    for i in analysis['improvements']:
        print(f"   - {i}")
    print(f"\nSummary: {analysis['summary']}")

match_cv(cv, job)                                                                                                                                                 
