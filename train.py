import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview"
)

EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

def get_embedding(text):
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding)

print("Loading dataset...")
df = pd.read_csv("candidate_job_role_dataset.csv")

df["experience_level"] = df["experience_level"].str.strip().str.rstrip(",")

role_counts = df["job_role"].value_counts()
valid_roles = role_counts[role_counts >= 10].index
df = df[df["job_role"].isin(valid_roles)].reset_index(drop=True)

df["text"] = df["skills"] + " " + df["qualification"] + " " + df["experience_level"]

# Label encoding
le = LabelEncoder()
df["label"] = le.fit_transform(df["job_role"])

print(f"Classes: {list(le.classes_)}")
print(f"Dataset size: {len(df)}")

# Calculating embeddings
cache_file = "embeddings_cache3.npy"
labels_file = "labels_cache3.npy"

if os.path.exists(cache_file):
    print("Loading embeddings from cache...")
    X = np.load(cache_file)
    y = np.load(labels_file)
else:
    print("Calculating embeddings...")
    embeddings = []
    for i, text in enumerate(df["text"]):
        emb = get_embedding(text)
        embeddings.append(emb)
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(df)} done")

    X = np.array(embeddings)
    y = df["label"].values
    np.save(cache_file, X)
    np.save(labels_file, y)
    print("Embeddings saved to cache.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
print("Training model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

# Evaluation
print("\nModel Performance:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and label encoder
with open("cv_matcher_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model saved.")