# ============================
# Movie Genre Classification
# CodSoft Internship - Task 1
# ============================

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset folder path
dataset_path = "Dataset"

# File paths
train_file = os.path.join(dataset_path, "train_data.txt")
test_file = os.path.join(dataset_path, "test_data.txt")

# Read training dataset
train_df = pd.read_csv(
    train_file,
    sep=r"\s*:::\s*",
    engine="python",
    header=None,
    names=["ID", "Title", "Genre", "Description"]
)

# Display dataset information
print("Dataset Loaded Successfully!")
print("\nShape of Dataset:")
print(train_df.shape)

print("\nFirst 5 Rows:")
print(train_df.head())

# -------------------------------
# Check Dataset Information
# -------------------------------

print("\nColumn Names:")
print(train_df.columns)

print("\nMissing Values:")
print(train_df.isnull().sum())

print("\nGenre Distribution:")
print(train_df["Genre"].value_counts().head(10))

# ==========================================
# Preparing Data
# ==========================================

# Input (Movie Description)
X = train_df["Description"]

# Output (Genre)
y = train_df["Genre"]

print("\nTotal Samples:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Completed!")
print("Training Shape:", X_train_tfidf.shape)
print("Testing Shape:", X_test_tfidf.shape)
# ==========================================
# Train Logistic Regression Model
# ==========================================

model = LogisticRegression(max_iter=1000)

print("\nTraining Model...")

model.fit(X_train_tfidf, y_train)

print("Model Training Completed!")

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
# ==========================================
# Save Model and Vectorizer
# ==========================================

joblib.dump(model, "model/movie_genre_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")