# 🧠 Multilingual Sacred Text NLP Analyzer

> An advanced, production-ready NLP Web Application for analyzing and comparing sacred texts across Gurbani, Quran, and Bible — powered by a lightweight SQLite database and cutting-edge Deep Learning models.

---

## 📌 Project Overview

This project is a full-stack Machine Learning Web Application built with **Streamlit**. It performs deep linguistic and semantic analysis on multilingual sacred texts including **Punjabi (Gurmukhi), Arabic, English, Hindi, and Urdu**.

At its core, the system:
- Auto-detects the input language
- Translates it to English (with override for authentic Gurbani translations)
- Analyzes Sentiment, Emotion, Themes, and Author/Raag metadata
- Finds semantically similar verses across three major scriptures

All scripture data is stored in a lightweight **SQLite database (`sacred_text.db`)** containing over **97,000 verses** across 5 tables, optimized for cloud deployment.

---

## ✨ Key Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Gurbani Metadata Engine** | 4-Tier TF-IDF lookup against 60,555 verses for 100% authentic Author, Raag & Ang (page) |
| 2 | **Authentic Translation Override** | Replaces Google Translate with the official theological English from the database |
| 3 | **Cross-Scripture Similarity** | Finds top-3 semantically similar verses across Gurbani, Quran & Bible |
| 4 | **Sentiment Analysis** | 3-way classification (Positive / Neutral / Negative) with confidence score |
| 5 | **Emotion Detection** | 7-way emotion classification (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral) |
| 6 | **Sacred Theme Extraction** | Domain-specific theological theme detection (e.g., Divine Identity, Devotion) |
| 7 | **Session History** | Every analysis is logged to the `history` table, filtered per user session |
| 8 | **Batch Analysis** | Upload a CSV file to run the full pipeline on multiple texts at once |
| 9 | **Export Results** | Download any analysis result as a formatted CSV file |
| 10 | **Clear Server Cache** | Force Streamlit to reload models and embeddings for immediate updates |

---

## 🗄️ SQLite Database: `sacred_text.db`

| Table | Rows | Description |
|-------|------|-------------|
| `gurbani` | 60,555 | Full Siri Guru Granth Sahib — Gurmukhi, English, Transliteration, Raag, Writer, Ang |
| `quran` | 6,235 | Holy Quran — Arabic original, Surah & Ayah numbers, Sahih International English translation |
| `bible` | 31,103 | Bible (BBE Translation) — Book, Chapter, Verse, English text |
| `balanced_corpus` | 18,705 | Auto-balanced sample (6,235 per scripture) for similarity search |
| `history` | Dynamic | Per-session user analysis log with timestamp, sentiment, emotion |

---

## 🔍 Gurbani Metadata — 4-Tier Lookup Strategy

```
Input Text
    │
    ├─► Tier 1: Exact Gurmukhi Substring Match      → 100% Confidence
    ├─► Tier 2: Char N-gram TF-IDF (Gurmukhi)       → Handles partial / variant text
    ├─► Tier 3: Exact English Substring Match        → 100% Confidence
    └─► Tier 4: Word N-gram TF-IDF (English)        → Handles translated/paraphrased input
```

- If a match is found → Author, Raag, and Ang are shown directly from the database
- If Raag or Author is `Unknown` → that field is **hidden** from the UI automatically

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Web App** | Streamlit |
| **Database** | SQLite3 |
| **Deep Learning** | HuggingFace Transformers (RoBERTa, DistilRoBERTa) |
| **Semantic Search** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Classical ML** | Scikit-Learn (TF-IDF, Cosine Similarity, SVM, Logistic Regression) |
| **NLP** | SpaCy (`en_core_web_sm`) |
| **Translation** | Deep-Translator (Google Translate API) |
| **Language Detection** | LangDetect |
| **Data** | Pandas, NumPy |

---

## 📂 Project Structure

```
NLP_Project/
│
├── app.py                              # Streamlit Frontend — UI, cards, tabs, history
├── main_pipeline.py                    # Core NLP Engine — orchestrates all analysis
│
├── database/                           # Database layer
│   ├── db_connection.py                # SQLite connector
│   ├── sacred_text.db                  # Core SQLite Database
│   ├── balance_datasets.py             # Script to build balanced_corpus table
│   └── mysql_to_sqlite.py              # Script used for migrating from MySQL to SQLite
│
├── nlp_engine/                         # NLP processing layer (Python package)
│   ├── gurbani_lookup.py               # 4-Tier Gurbani metadata search (SQLite)
│   ├── similarity.py                   # Cross-scripture semantic similarity (SQLite)
│   └── classifiers.py                  # ML Author & Raag classifiers
│
├── cache/                              # ML model & embedding cache (Pre-computed for cloud)
│   ├── embeddings_cache/               # Sentence embedding .npy files (Bible, Quran, Gurbani)
│   └── saved_models/                   # Trained classifier .pkl files
│
├── Dataset/                            # Raw CSV scripture source files
│   ├── Arabic-Original.csv             # Quran (pipe-delimited: surah|ayah|arabic)
│   └── t_bbe.csv                       # Bible BBE translation
│
├── batch_files/                        # Launcher scripts
│   ├── run_streamlit.bat               # One-click app launcher (navigates to root)
│   └── run_pipeline.bat                # Terminal pipeline runner
│
└── README.md                           # This file
```

---

## ▶️ How to Run

### Prerequisites
- Python 3.10+
- The project is fully standalone and requires no external database servers due to SQLite integration.

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2 — Launch the App
Double-click **`batch_files\run_streamlit.bat`** (Windows)

Or run manually from the terminal:
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
transformers
sentence-transformers
spacy
torch
langdetect
deep-translator
joblib
```
