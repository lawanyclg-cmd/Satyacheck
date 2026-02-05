🛰️ SatyaCheck — Explainable Fake News Detection

SatyaCheck is an explainable fake news detection web application built using Flask, Machine Learning, and LLM-powered explanations.

Unlike typical fake news detectors that only output REAL or FAKE, SatyaCheck focuses on transparency and human-level explainability — showing users why an article may be misleading.

✨ Features

Paste news text or a news article URL

Automatic article text extraction using BeautifulSoup

ML model (TF-IDF + Logistic Regression) for prediction

Confidence score for each prediction

Important word indicators (real vs fake signals)

AI-generated human-friendly explanation using Groq (Llama 3)

Claim vs Evidence table generated from the article

Annotated reasoning for better understanding

Clean UI with Light/Dark mode

Fully server-rendered Flask app (no JavaScript)

⚙️ Tech Stack

Python

Flask

Scikit-learn

BeautifulSoup

Groq API (Llama 3)

HTML + CSS (no JS)

🌍 **Live Demo:** https://satyacheck.onrender.com
