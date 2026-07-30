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

#Test Data
cv = """
ZEYNEP MUTLU
Computer Engineer
EXPERIENCE
Data Engineering Intern Bentego
 07/2025 - 09/2025
- Processed and analyzed large-scale telecom and finance datasets using Apache Spark and Python, improving overall data pipeline efficiency
- Developed and deployed 5+ ETL pipelines extracting data from multiple sources into PostgreSQL, reducing manual data handling and error rates
-Redesigned daily ETL processes to extract only current-day data rather than performing full table scans on large datasets, reducing processing time from 8 hours to 5 hours and improving system efficiency
TECHNICAL SKILLS
Programming: Python, SQL, JavaScript, TypeScript, Java, C++ 
Machine Learning & AI: scikit-learn, XGBoost, TensorFlow, Keras, Gradient Boosting, Neural Networks, SHAP
Data Science & Engineering: Pandas, NumPy, Matplotlib, Seaborn, Apache Spark, ETL Pipelines, Data Processing, Data Modeling
Databases: PostgreSQL, MySQL, Firebase Firestore
Cloud & DevOps: AWS (Lambda, DynamoDB, Cognito, CloudFront, CodePipeline), Docker, Linux, Google Cloud Storage
Tools: Git, Retool, Jupyter Notebook, Power BI
Robotics: ROS2, CMake
PROJECTS
AI-Powered Library Book Recommendation System
- Developed serverless backend with AWS Lambda functions (Node.js) for books catalog and reading lists management 
- Integrated AWS Bedrock AI (Claude 3 Haiku) for natural language book recommendations 
- Built automated CI/CD pipeline (CodePipeline + CodeBuild) 
- Implemented JWT authentication with AWS Cognito and user data isolation in DynamoDB 
- Deployed globally via CloudFront CDN with 100% serverless architecture

House Price Prediction Using Machine Learning & Deep Learning
Graduation Project
- Developed end-to-end ML/DL system comparing 4 algorithms on 21,600+ records — best model achieved 99.5% accuracy (R²=0.9956) with $9,124 average prediction error
- Implemented end-to-end pipeline: data preprocessing, feature engineering (8 derived features), hyperparameter tuning via GridSearchCV, 5-fold cross-validation and SHAP explainability analysis 

- Built TensorFlow/Keras neural network (128-64-32) with dropout regularization achieving R²=0.9586; identified ensemble methods outperform deep learning on tabular data

AI-Powered Data-Driven News Video Automation System
- Built automated news video pipeline using Remotion  and TypeScript
- Integrated Google Gemini AI for intelligent news scoring and content analysis across 4 metrics
- Optimized API efficiency with batch scoring (20x fewer API calls vs. individual requests)
- Designed modular, scalable video composition architecture with 14 dynamic scenes
- Automated end-to-end pipeline: data fetch  AI analysis  video render
EDUCATION
Bachelor's Degree in Computer Engineering (English-medium program)
Okan University, Istanbul, Turkey 09/2021 - 06/2026
Relevant Coursework: Data Structures, Algorithms, Computer Networks, Database Management Systems, Big Data Processing, Kalman Filtering & Sensor Fusion, Robot Operating System (ROS2), Machine Learning
CERTIFICATIONS

 https://www.linkedin.com/in/zeynep-mutlu-058347294/details/certifications/

 https://learning.veribilimiokulu.com/certificates/gxokvqw6pm
LANGUAGES
Turkish: Native
English: Advanced (Professional working proficiency)
German: Beginner (Basic conversational skills)
 

"""

job = """
About the Position

We are looking for a team member to join our data science team, with experience or a desire to build a career in regression, forecasting, classification, clustering, and other modeling projects.

Responsibilities:

* Develop projects using data science techniques such as regression, forecasting, classification, and clustering.
* Perform in-depth analyses on data and derive meaningful insights.
* Effectively use Python, R, and other relevant tools in model development processes.
* Clean, prepare, and analyze datasets.
* Collaborate with cross-functional teams and provide regular progress reports on projects.
* Follow relevant data science research and propose innovative solutions.

Required Qualifications:

* Bachelor's or Master's degree in Statistics, Mathematics, Computer Science, Engineering, or a related field.
* Experience in regression, forecasting, classification, and clustering.
* Knowledge of Python, R, and data science libraries (Pandas, NumPy, scikit-learn, TensorFlow, Keras).
* Experience working with SQL and NoSQL databases.
* Analytical thinking, problem-solving, and decision-making skills.
* Ability to explain technical concepts to non-technical team members.
* Team-oriented with strong communication skills.

Preferred Qualifications:

* Proven achievements and portfolio in relevant data science projects.
* Advanced certifications or training in data science and machine learning.
* Knowledge of big data platforms (Hadoop, Spark).
"""
match_cv(cv, job)                                                                                                                                                 