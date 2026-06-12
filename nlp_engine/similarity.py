import pandas as pd
import numpy as np
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'cache', 'embeddings_cache')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ============================================================
# Lazy-load everything via st.cache_resource
# ============================================================

@st.cache_resource
def load_full_gurbani():
    """Load the full 60k+ gurbani dataset for exact authentication."""
    from database.db_connection import get_sqlalchemy_engine
    engine = get_sqlalchemy_engine()
    if not engine:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql('SELECT * FROM gurbani', con=engine)
        df['gurmukhi'] = df['gurmukhi'].fillna('')
        df['english'] = df['english'].fillna('')
        return df
    except Exception as e:
        print(f"[Similarity] DB Error loading gurbani: {e}")
        return pd.DataFrame()

def exact_gurbani_match(original_text, translated_text):
    """Find the exact verse in the full corpus for 100% authentic metadata."""
    df = load_full_gurbani()
    if df.empty: return None
    
    import re
    # Split original text by delimiters (danda, newline, period)
    chunks = re.split(r'[॥।\n.]', original_text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 5]
    
    for orig in chunks:
        # Try matching original text (Gurmukhi or Transliteration)
        matches = df[df['gurmukhi'].str.contains(orig, regex=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            return {'author': str(row.get('writer', 'Unknown')), 'raag': str(row.get('raag', 'Unknown')), 'text': str(row.get('english', ''))}
            
    # Split English text by delimiters
    t_chunks = re.split(r'[\n.]', translated_text)
    t_chunks = [c.strip() for c in t_chunks if len(c.strip()) > 5]
    
    for trans in t_chunks:
        # Try matching English translation
        matches = df[df['english'].str.contains(trans, case=False, regex=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            return {'author': str(row.get('writer', 'Unknown')), 'raag': str(row.get('raag', 'Unknown')), 'text': str(row.get('english', ''))}
            
    return None

@st.cache_resource
def load_sentence_transformer():
    print("[Similarity] Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')   # 80MB — fast!
    print("[Similarity] Model loaded.")
    return model


@st.cache_resource
def load_all_data():
    """Load balanced corpus and compute/load embeddings. Cached by Streamlit."""
    st_model = load_sentence_transformer()

    from database.db_connection import get_sqlalchemy_engine
    engine = get_sqlalchemy_engine()
    
    try:
        df = pd.read_sql('SELECT * FROM balanced_corpus', con=engine)
    except Exception as e:
        print(f"[Similarity] WARNING: Error loading balanced_corpus from DB: {e}")
        return [], [], [], np.array([]), np.array([]), np.array([])

    gurbani, quran, bible = [], [], []

    for _, row in df.iterrows():
        record = {
            'text': str(row['text']).strip(),
            'scripture': str(row.get('source', '')).strip() if 'source' in df.columns else str(row.get('scripture', '')).strip(),
            'author': str(row.get('author_or_raag', 'Unknown')) if 'author_or_raag' in df.columns else str(row.get('author', 'Unknown')),
            'raag': 'Unknown',
            'ang': ''
        }
        if record['scripture'].lower() == 'gurbani':
            gurbani.append(record)
        elif record['scripture'].lower() == 'quran':
            quran.append(record)
        elif record['scripture'].lower() == 'bible':
            bible.append(record)

    def get_embeddings(records, cache_name):
        cache_path = os.path.join(EMBEDDINGS_DIR, f'{cache_name}_balanced.npy')

        if os.path.exists(cache_path):
            print(f"[Similarity] Loading cached {cache_name} embeddings...")
            return np.load(cache_path, allow_pickle=True)

        texts = [r['text'] for r in records]
        print(f"[Similarity] Computing embeddings for {len(texts)} {cache_name} verses...")
        embeddings = st_model.encode(texts, show_progress_bar=True, batch_size=64)
        np.save(cache_path, embeddings)
        print(f"[Similarity] {cache_name} embeddings cached.")
        return embeddings

    gurbani_emb = get_embeddings(gurbani, 'gurbani')
    quran_emb   = get_embeddings(quran,   'quran')
    bible_emb   = get_embeddings(bible,   'bible')

    print(f"[Similarity] Ready! Gurbani: {len(gurbani)}, Quran: {len(quran)}, Bible: {len(bible)}\n")
    return gurbani, quran, bible, gurbani_emb, quran_emb, bible_emb


def find_similar(text, top_k=3):
    """Find top-k most similar verses from each scripture."""
    st_model = load_sentence_transformer()
    gurbani_records, quran_records, bible_records, \
        gurbani_embeddings, quran_embeddings, bible_embeddings = load_all_data()

    query_embedding = st_model.encode([text])
    results = {}

    for name, records, embeddings in [
        ('Gurbani', gurbani_records, gurbani_embeddings),
        ('Quran',   quran_records,   quran_embeddings),
        ('Bible',   bible_records,   bible_embeddings)
    ]:
        if len(records) == 0 or embeddings.size == 0:
            results[name] = []
            continue

        scores = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = scores.argsort()[-top_k:][::-1]

        matches = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            # Apply mathematical boost to raw cosine score for better UI perception
            # A raw score of 0.58 will map to ~83%
            boosted_score = (raw_score ** 0.35) * 100 if raw_score > 0 else 0.0
            
            matches.append({
                'text':      records[idx]['text'],
                'scripture': name,
                'score':     round(min(boosted_score, 99.9), 1),
                'author':    records[idx]['author'],
                'raag':      records[idx].get('raag', 'Unknown'),
                'ang':       records[idx].get('ang', '')
            })
        results[name] = matches

    return results
