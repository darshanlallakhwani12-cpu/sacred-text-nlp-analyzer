"""
gurbani_lookup.py  —  Authentic Raag & Guru lookup for any Gurbani input.
==========================================================================

4-tier strategy (tried in order):

  Tier 1  Exact Gurmukhi substring match          → 100% confidence
  Tier 2  TF-IDF on Gurmukhi char n-grams         → handles partial / different dandas
  Tier 3  Exact English substring match            → 100% confidence
  Tier 4  TF-IDF on English word n-grams           → handles Google-Translate paraphrasing

All data comes straight from the MySQL 'gurbani' table — no ML guessing, no balanced_corpus.
"""

import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalise whitespace and remove dandas / visraam marks."""
    text = re.sub(r'[।॥\|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource: load dataset + build two TF-IDF indexes (Gurmukhi + English)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_indexes():
    from database.db_connection import get_sqlalchemy_engine
    engine = get_sqlalchemy_engine()
    if not engine:
        return None, None, None, None, None

    try:
        df = pd.read_sql('SELECT * FROM gurbani', con=engine)
    except Exception as e:
        print(f"[GurbaniLookup] DB Error: {e}")
        return None, None, None, None, None

    df['gurmukhi']        = df['gurmukhi'].fillna('').apply(_clean)
    df['english']         = df['english'].fillna('')
    df['transliteration'] = df['transliteration'].fillna('')
    df['writer']          = df['writer'].fillna('Unknown')
    df['raag']            = df['raag'].fillna('Unknown')
    df['ang']             = df['ang'].fillna('') if 'ang' in df.columns else ''

    # ── Gurmukhi TF-IDF (character n-grams: 2-4) ────────────────────────────
    print(f"[GurbaniLookup] Building Gurmukhi n-gram index over {len(df)} verses …")
    gur_vec = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        min_df=1, sublinear_tf=True
    )
    gur_mat = gur_vec.fit_transform(df['gurmukhi'].tolist())

    # ── English TF-IDF (word + bigrams) ─────────────────────────────────────
    print("[GurbaniLookup] Building English word index …")
    eng_vec = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        min_df=1, sublinear_tf=True, strip_accents='unicode'
    )
    eng_mat = eng_vec.fit_transform(df['english'].tolist())

    print("[GurbaniLookup] Both indexes ready.")
    return df, gur_vec, gur_mat, eng_vec, eng_mat


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_gurbani_metadata(original_text: str, translated_text: str) -> dict | None:
    """
    Return  {'author': str, 'raag': str, 'confidence': float, 'matched_text': str}
    or None if no confident match is found.
    """
    result = _load_indexes()
    if result[0] is None:
        return None
    df, gur_vec, gur_mat, eng_vec, eng_mat = result

    orig  = _clean(original_text)
    trans = translated_text.strip()

    # ── Tier 1: Exact Gurmukhi substring match ───────────────────────────────
    chunks = [c.strip() for c in re.split(r'[\n]', orig) if len(c.strip()) > 8]
    if not chunks:
        chunks = [orig] if len(orig) > 8 else []

    for chunk in chunks:
        mask = df['gurmukhi'].str.contains(re.escape(chunk), regex=True, na=False)
        hits = df[mask]
        if not hits.empty:
            row = hits.iloc[0]
            return _result(row, 100.0)

    # ── Tier 2: TF-IDF Gurmukhi char n-gram similarity ──────────────────────
    if len(orig) > 8:
        try:
            q_vec  = gur_vec.transform([orig])
            scores = cosine_similarity(q_vec, gur_mat)[0]
            best_i = int(np.argmax(scores))
            best_s = float(scores[best_i])
            if best_s >= 0.30:          # decent character overlap
                conf = _boost(best_s, low=80, high=99)
                return _result(df.iloc[best_i], conf)
        except Exception:
            pass

    # ── Tier 3: Exact English substring match ───────────────────────────────
    t_chunks = [c.strip() for c in re.split(r'[\n.]', trans) if len(c.strip()) > 12]
    for chunk in t_chunks:
        mask = df['english'].str.contains(re.escape(chunk), case=False, regex=True, na=False)
        hits = df[mask]
        if not hits.empty:
            row = hits.iloc[0]
            return _result(row, 100.0)

    # ── Tier 4: TF-IDF English word similarity ──────────────────────────────
    query = trans if len(trans) > 8 else orig
    if len(query) < 8:
        return None

    try:
        q_vec  = eng_vec.transform([query])
        scores = cosine_similarity(q_vec, eng_mat)[0]
        best_i = int(np.argmax(scores))
        best_s = float(scores[best_i])
        if best_s < 0.08:               # not enough overlap — don't guess
            return None
        conf = _boost(best_s, low=75, high=97)
        return _result(df.iloc[best_i], conf)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _result(row, confidence: float) -> dict:
    return {
        'author':           str(row['writer']),
        'raag':             str(row['raag']),
        'confidence':       round(confidence, 1),
        'matched_english':  str(row['english']),
        'matched_gurmukhi': str(row['gurmukhi']),
        'ang':              str(row['ang']) if 'ang' in row.index else ''
    }


def _boost(raw: float, low: float, high: float) -> float:
    """Map a raw score in [0,1] to the [low, high] display range."""
    clamped = min(max(raw, 0.0), 1.0)
    return low + clamped * (high - low)
