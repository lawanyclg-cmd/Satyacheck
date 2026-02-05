# -*- coding: utf-8 -*-


from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Text cleaning (shared)
# -----------------------------
def ensure_nltk_resources() -> None:
    """Ensure required NLTK resources are available (call once at startup)."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.word_tokenize("test")
    except LookupError:
        nltk.download("punkt", quiet=True)


_STEMMER = PorterStemmer()
_STOP_WORDS = None


def _get_stop_words() -> set:
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS


def clean_text(text: str) -> str:
    """Clean a news article into a normalized string for TF-IDF."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    cleaned: List[str] = []
    stop_words = _get_stop_words()
    for word in words:
        if word not in stop_words and len(word) > 2:
            cleaned.append(_STEMMER.stem(word))
    return " ".join(cleaned)


# -----------------------------
# Training + saving
# -----------------------------
def train_and_save(
    fake_csv_path: str = 'D:/Codezen_New/Data/Fake.csv',
    real_csv_path: str = 'D:/Codezen_New/Data/True.csv',
    model_path: str = "model.pkl",
    vectorizer_path: str = "vectorizer.pkl",
    important_words_path: str = "important_words.json",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    top_n: int = 20,
) -> None:
    """
    Train a TF-IDF + Logistic Regression model and save artifacts.
    Intended to be run offline (not during prediction).
    """
    ensure_nltk_resources()

    df_fake = pd.read_csv(fake_csv_path)
    df_real = pd.read_csv(real_csv_path)
    df_fake["label"] = 0  # 0 = Fake
    df_real["label"] = 1  # 1 = Real
    df = pd.concat([df_fake, df_real], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    important_words = extract_important_words(model, vectorizer, top_n=top_n)
    with open(important_words_path, "w", encoding="utf-8") as f:
        json.dump(important_words, f, indent=2)


def extract_important_words(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    top_n: int = 20,
) -> Dict[str, List[str]]:
    """Extract top positive/negative words from model coefficients."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_.ravel()

    top_pos_idx = coefs.argsort()[-top_n:][::-1]
    top_neg_idx = coefs.argsort()[:top_n]

    return {
        "top_real_words": [feature_names[i] for i in top_pos_idx],
        "top_fake_words": [feature_names[i] for i in top_neg_idx],
    }


# -----------------------------
# Flask-ready prediction
# -----------------------------
def load_assets(
    model_path: str = "model.pkl",
    vectorizer_path: str = "vectorizer.pkl",
    important_words_path: str = "important_words.json",
):
    """Load model/vectorizer/important words once at app startup."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    important_words = {}
    if os.path.exists(important_words_path):
        with open(important_words_path, "r", encoding="utf-8") as f:
            important_words = json.load(f)
    return model, vectorizer, important_words


def predict_news(
    text: str,
    model,
    vectorizer,
    important_words: Dict[str, List[str]] | None = None,
) -> Tuple[str, float, Dict[str, List[str]]]:
    """
    Predict label and confidence for a single news article.
    Returns (label, confidence, important_words).
    """
    if not text or not isinstance(text, str):
        return "FAKE", 0.0, important_words or {}

    clean = clean_text(text)
    X = vectorizer.transform([clean])
    proba = model.predict_proba(X)[0]
    pred = int(proba[1] >= 0.5)
    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(proba[pred])
    return label, confidence, important_words or {}


if __name__ == "__main__":
    # Example training run (offline). Update paths for your environment.
    ensure_nltk_resources()
    train_and_save(
        fake_csv_path="D:/Codezen_New/Data/Fake.csv",
        real_csv_path="D:/Codezen_New/Data/True.csv",
        model_path="model.pkl",
        vectorizer_path="vectorizer.pkl",
        important_words_path="important_words.json",
        top_n=20,
    )
