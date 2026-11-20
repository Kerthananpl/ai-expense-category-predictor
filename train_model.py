"""Train a text-classification model for expense descriptions."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "training_expenses.csv"
MODEL = ROOT / "expense_category_model.joblib"

def build_model(data):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(data["description"].astype(str), data["category"].astype(str))
    return pipeline

def main():
    data = pd.read_csv(DATA)
    pipeline = build_model(data)
    joblib.dump(pipeline, MODEL)
    print(f"Model trained on {len(data)} example transactions.")
    print(f"Saved model to: {MODEL}")

if __name__ == "__main__":
    main()
