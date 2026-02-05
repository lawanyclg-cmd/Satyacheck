from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from flask import Flask, render_template, request
import joblib
from markupsafe import Markup, escape
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Flask app setup
app = Flask(__name__)

# -----------------------------
# Model asset loading (startup)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROQ_MODEL = "llama-3.3-70b-versatile"

def _resolve_path(filename: str) -> str:
    """Look for model files in Flask root first, then Flask/model."""
    root_path = os.path.join(BASE_DIR, filename)
    model_path = os.path.join(BASE_DIR, "model", filename)
    if os.path.exists(root_path):
        return root_path
    return model_path

MODEL = joblib.load(_resolve_path("model.pkl"))
VECTORIZER = joblib.load(_resolve_path("vectorizer.pkl"))

with open(_resolve_path("important_words.json"), "r", encoding="utf-8") as f:
    IMPORTANT_WORDS: Dict[str, List[str]] = json.load(f)

_STEMMER = PorterStemmer()
try:
    _STOP_WORDS = set(stopwords.words("english"))
except Exception:
    _STOP_WORDS = set()


def clean_text(text: str) -> str:
    """Lightweight cleaner aligned with training preprocessing."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    cleaned: List[str] = []
    for word in words:
        if word not in _STOP_WORDS and len(word) > 2:
            cleaned.append(_STEMMER.stem(word))
    return " ".join(cleaned)


# -----------------------------
# Prediction helper
# -----------------------------
def predict_news(text: str) -> Tuple[str, float, Dict[str, List[str]]]:
    """
    Predict label and confidence for a single news article.
    Returns (label, confidence, important_words).
    """
    if MODEL is None or VECTORIZER is None:
        return "FAKE", 0.0, IMPORTANT_WORDS

    if not text or not isinstance(text, str):
        return "FAKE", 0.0, IMPORTANT_WORDS

    # Clean, vectorize, and predict
    cleaned = clean_text(text)
    vec = VECTORIZER.transform([cleaned])
    print(f"DEBUG: vec.shape = {vec.shape}")  # should be (1, N)

    prediction = MODEL.predict(vec)[0]
    confidence = float(MODEL.predict_proba(vec).max())
    label = "REAL" if prediction == 1 else "FAKE"
    return label, confidence, IMPORTANT_WORDS


def extract_text_from_url(url: str) -> str:
    """
    Fetch an article URL, extract visible paragraph text, and return combined text.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        combined = " ".join([p for p in paragraphs if p])
        return combined
    except Exception:
        return ""


def _looks_like_url(text: str) -> bool:
    if not text:
        return False
    if text.startswith("http://") or text.startswith("https://"):
        return True
    return text.startswith("www.")


def generate_explanation(
    label: str,
    confidence: float,
    real_words: List[str],
    fake_words: List[str],
    news_excerpt: str,
) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "Explanation unavailable (missing GROQ_API_KEY)."

    prompt = (
        "You are an expert fact-checker.\n\n"
        f"A machine learning model has classified a news article as {label} "
        f"with {confidence}% confidence.\n\n"
        f"Words indicating REAL news: {', '.join(real_words)}\n"
        f"Words indicating FAKE news: {', '.join(fake_words)}\n\n"
        "Here is a portion of the article:\n"
        f"\"\"\"{news_excerpt}\"\"\"\n\n"
        f"Explain in simple human language WHY this article is likely {label}.\n"
        "Do not mention machine learning or keywords.\n"
        "Write like a journalist explaining the reasoning to a reader.\n"
        "Point out tone, wording style, credibility, and patterns.\n"
        "Be concise: 2-3 sentences max, under 70 words, only the most important points."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        explanation = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return explanation.strip() or "Explanation unavailable."
    except requests.exceptions.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        body = getattr(exc.response, "text", "")
        body_preview = body[:400].replace("\n", " ").strip() if body else ""
        if status is not None:
            print(f"Groq API error: HTTP {status} - {body_preview}")
        else:
            print(f"Groq API connection error: {type(exc).__name__}: {exc}")
        return "Explanation unavailable (could not reach Groq)."
    except Exception as exc:
        print(f"Groq API unexpected error: {type(exc).__name__}: {exc}")
        return "Explanation unavailable (could not reach Groq)."


def generate_claim_evidence_table(news_text: str) -> List[Dict[str, str]]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return []

    excerpt = (news_text or "")[:1200]
    prompt = (
        "You are an expert fact-checker.\n\n"
        "From the article text below, identify up to 4 major factual claims made by the author.\n\n"
        "For each claim, determine whether the article itself provides supporting evidence such as data, "
        "sources, references, expert quotes, or verifiable facts.\n\n"
        "Return the output strictly in this JSON format:\n\n"
        "[\n"
        "  {\n"
        "    \"claim\": \"text of the claim\",\n"
        "    \"evidence\": \"Yes\" or \"No\",\n"
        "    \"reason\": \"short explanation why evidence is or is not present\"\n"
        "  }\n"
        "]\n\n"
        "Article text:\n"
        f"\"\"\"{excerpt}\"\"\""
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        return []
    except requests.exceptions.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        body = getattr(exc.response, "text", "")
        body_preview = body[:400].replace("\n", " ").strip() if body else ""
        if status is not None:
            print(f"Groq Claim/Evidence error: HTTP {status} - {body_preview}")
        else:
            print(f"Groq Claim/Evidence connection error: {type(exc).__name__}: {exc}")
        return []
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Groq Claim/Evidence parse error: {type(exc).__name__}: {exc}")
        return []
    except Exception as exc:
        print(f"Groq Claim/Evidence unexpected error: {type(exc).__name__}: {exc}")
        return []


def generate_annotations(news_text: str) -> List[Dict[str, str]]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return []

    excerpt = (news_text or "")[:1500]
    prompt = (
        "You are a professional fact-checker.\n\n"
        "From the article text below, extract up to 6 exact sentences that demonstrate any of the following:\n\n"
        "- Emotional or sensational wording\n"
        "- Conspiracy or controversial framing\n"
        "- Claims made without evidence\n"
        "- Persuasive or manipulative tone\n\n"
        "Return ONLY valid JSON in this format:\n\n"
        "[\n"
        "  {\n"
        "    \"sentence\": \"exact sentence from the article\",\n"
        "    \"issue\": \"Emotional wording / Conspiracy framing / No evidence / Manipulative tone\",\n"
        "    \"explanation\": \"short explanation of why this sentence is problematic\"\n"
        "  }\n"
        "]\n\n"
        "Article text:\n"
        f"\"\"\"{excerpt}\"\"\""
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        return []
    except requests.exceptions.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        body = getattr(exc.response, "text", "")
        body_preview = body[:400].replace("\n", " ").strip() if body else ""
        if status is not None:
            print(f"Groq Annotations error: HTTP {status} - {body_preview}")
        else:
            print(f"Groq Annotations connection error: {type(exc).__name__}: {exc}")
        return []
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Groq Annotations parse error: {type(exc).__name__}: {exc}")
        return []
    except Exception as exc:
        print(f"Groq Annotations unexpected error: {type(exc).__name__}: {exc}")
        return []


def annotate_article(text: str, annotations: List[Dict[str, str]]) -> Markup:
    if not text:
        return Markup("")

    annotated = escape(text)
    for item in annotations:
        sentence = str(item.get("sentence", "") or "").strip()
        explanation = str(item.get("explanation", "") or "").strip()
        if not sentence:
            continue
        safe_sentence = escape(sentence)
        safe_note = escape(explanation) if explanation else escape("No explanation provided.")
        replacement = (
            f'<span class="annotated" data-note="{safe_note}">{safe_sentence}</span>'
        )
        annotated = annotated.replace(safe_sentence, Markup(replacement))
    return Markup(annotated)


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        text = request.form.get("news", "").strip()

        if _looks_like_url(text):
            url = text if text.startswith("http") else f"https://{text}"
            extracted = extract_text_from_url(url)
            if not extracted:
                error = "We could not extract readable text from that URL. Please try another link."
            else:
                label, confidence, words = predict_news(extracted)
                real_words = words.get("top_real_words", words.get("real_words", []))
                fake_words = words.get("top_fake_words", words.get("fake_words", []))
                explanation = generate_explanation(
                    label=label,
                    confidence=round(confidence * 100, 2),
                    real_words=real_words,
                    fake_words=fake_words,
                    news_excerpt=extracted[:500],
                )
                claim_table = generate_claim_evidence_table(extracted)
                result = {
                    "label": label,
                    "confidence": round(confidence * 100, 2),
                    "important_words": words,
                    "explanation": explanation,
                    "claim_table": claim_table,
                    "article_text": extracted,
                    "input_text": text,
                }
        elif text:
            label, confidence, words = predict_news(text)
            real_words = words.get("top_real_words", words.get("real_words", []))
            fake_words = words.get("top_fake_words", words.get("fake_words", []))
            explanation = generate_explanation(
                label=label,
                confidence=round(confidence * 100, 2),
                real_words=real_words,
                fake_words=fake_words,
                news_excerpt=text[:500],
            )
            claim_table = generate_claim_evidence_table(text)
            result = {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "important_words": words,
                "explanation": explanation,
                "claim_table": claim_table,
                "article_text": text,
                "input_text": text,
            }
        else:
            error = "Please paste news text or an article URL."

    return render_template("index.html", result=result, error=error)


@app.route("/annotated", methods=["POST"])
def annotated():
    article_text = request.form.get("article_text", "").strip()
    if not article_text:
        return render_template(
            "annotated.html",
            annotated_text=Markup(""),
            error="No article text found to annotate. Please analyze an article first.",
        )

    annotations = generate_annotations(article_text)
    annotated_text = annotate_article(article_text, annotations)
    return render_template(
        "annotated.html",
        annotated_text=annotated_text,
        error=None,
    )


if __name__ == "__main__":
    app.run()

