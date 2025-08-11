# Titanic Survival Predictor

This project trains a Random Forest model to predict Titanic passenger survival using key features, and provides an interactive Streamlit web app for predictions.

---

## Project Structure

- `train_model.py`: Script to load data, preprocess, train, evaluate, and save the model pipeline.
- `app.py`: Streamlit app to predict survival from user input or batch CSV uploads.
- `model_artifacts/`: Stores the saved model pipeline and metrics.
- `data/`: Optional folder to store datasets locally.
- `requirements.txt`: Python dependencies.

---

## How to Run

1. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies

  ```bash
  pip install -r requirements.txt

2. Install dependencies

  ```bash
  pip install -r requirements.txt



## Features
 - Interactive UI for single passenger survival prediction.
 - Batch CSV upload for multiple predictions.
 - Clear model evaluation metrics saved during training.
