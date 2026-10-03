"""Train a lightweight scam-message classifier and export it for on-device use.

Uses data/dataset.csv if present (e.g. the UCI SMS Spam Collection), otherwise
falls back to the small synthetic sample in data/sample_messages.csv.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SPAM_LABELS = {"scam", "spam", "phishing", "1"}


def load():
    path = Path("data/dataset.csv")
    if not path.exists():
        path = Path("data/sample_messages.csv")
        print("data/dataset.csv not found: using the small synthetic sample.")
    df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"} <= set(df.columns):  # UCI SMS Spam Collection format
        df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna()
    y = df["label"].astype(str).str.lower().isin(SPAM_LABELS).astype(int)
    return df["text"].astype(str), y


def main():
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te), target_names=["safe", "scam"], zero_division=0))

    Path("model").mkdir(exist_ok=True)
    joblib.dump(model, "model/model.joblib")

    # Plain-JSON export: small enough to port to TFLite / Android / browser.
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    export = {
        "vocabulary": {t: int(i) for t, i in tfidf.vocabulary_.items()},
        "idf": [float(v) for v in tfidf.idf_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
    }
    Path("model/model.json").write_text(json.dumps(export), encoding="utf-8")
    print("Saved model/model.joblib and model/model.json")


if __name__ == "__main__":
    main()
