"""
Train a predictive model on the Titanic dataset and save the model + preprocessing pipeline.
Usage:
    python train_model.py
Outputs:
    - model_pipeline.pkl
    - metrics.txt
"""

# import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------
# Config
# -------------------------
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
OUTPATH = Path("model_artifacts")
OUTPATH.mkdir(exist_ok=True)

# -------------------------
# Load
# -------------------------
df = pd.read_csv(DATA_URL)

# -------------------------
# Feature engineering (simple & explainable)
# -------------------------
df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
# Map rare titles to 'Other'
title_counts = df['Title'].value_counts()
common_titles = title_counts[title_counts > 5].index
df['Title'] = df['Title'].where(df['Title'].isin(common_titles), other='Other')

# Choose target and features
target = "Survived"
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']

X = df[features]
y = df[target]

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Preprocessing pipelines
# -------------------------
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -------------------------
# Model pipeline
# -------------------------
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# -------------------------
# Train
# -------------------------
clf.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_proba)
}

# Save artifacts
joblib.dump(clf, OUTPATH / "model_pipeline.pkl")

with open(OUTPATH / "metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

print("Model trained and saved to", OUTPATH / "model_pipeline.pkl")
print("Metrics:", metrics)
