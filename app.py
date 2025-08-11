"""
Streamlit app to load the saved pipeline and make predictions.
Run: streamlit run app.py
"""

# import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("model_artifacts/model_pipeline.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("Titanic Survival Predictor")
st.markdown("Upload a CSV (with same feature columns) or fill in a single passenger to predict survival probability.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Single input form
st.header("Predict single passenger")
with st.form("single_form"):
    pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=32.2)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])
    submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked,
            'Title': title
        }])
        proba = model.predict_proba(row)[0, 1]
        pred = model.predict(row)[0]
        st.metric("Survival probability", f"{proba*100:.1f}%")
        st.info("Prediction: " + ("Survived" if pred == 1 else "Did not survive"))

st.markdown("---")
st.header("Batch predictions (CSV)")
uploaded = st.file_uploader("Upload CSV with columns: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    if not set(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']).issubset(df.columns):
        st.error("CSV is missing required columns.")
    else:
        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)
        df['survival_prob'] = probs
        df['pred_survived'] = preds
        st.write(df.head())
        st.download_button("Download predictions CSV", df.to_csv(index=False), "predictions.csv", "text/csv")
