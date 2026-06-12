import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'cache', 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

AUTHOR_VEC_PATH   = os.path.join(MODEL_DIR, 'author_vectorizer.pkl')
AUTHOR_MODEL_PATH = os.path.join(MODEL_DIR, 'author_model.pkl')
RAAG_VEC_PATH     = os.path.join(MODEL_DIR, 'raag_vectorizer.pkl')
RAAG_MODEL_PATH   = os.path.join(MODEL_DIR, 'raag_model.pkl')


def _train_and_save():
    print("[Classifiers] Training from scratch...")

    # Always use the full gurbani.csv for highest accuracy and authenticity
    from database.db_connection import get_sqlalchemy_engine
    engine = get_sqlalchemy_engine()
    df_gurbani = pd.read_sql('SELECT * FROM gurbani', con=engine)
    text_col, writer_col, raag_col = 'english', 'writer', 'raag'

    df_gurbani = df_gurbani.dropna(subset=[text_col, writer_col])

    # -- Author classifier --
    print("[Classifiers] Training Author Classifier...")
    av = TfidfVectorizer(stop_words='english', max_features=3000)
    X_a = av.fit_transform(df_gurbani[text_col])
    y_a = df_gurbani[writer_col]
    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_a, y_a, test_size=0.2, random_state=42)
    am = LogisticRegression(max_iter=1000, random_state=42)
    am.fit(X_tr_a, y_tr_a)
    print(f"[Classifiers] Author — Train: {accuracy_score(y_tr_a, am.predict(X_tr_a))*100:.1f}%, Val: {accuracy_score(y_te_a, am.predict(X_te_a))*100:.1f}%")
    joblib.dump(av, AUTHOR_VEC_PATH)
    joblib.dump(am, AUTHOR_MODEL_PATH)

    # -- Raag classifier --
    print("[Classifiers] Training Raag Classifier (SVM)...")
    df_r = df_gurbani.dropna(subset=[raag_col]).copy()
    df_r = df_r[df_r[raag_col] != 'Unknown']
    counts = df_r[raag_col].value_counts()
    rare   = counts[counts < 50].index
    df_r[raag_col] = df_r[raag_col].apply(lambda x: 'Other' if x in rare else x)

    rv = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_r = rv.fit_transform(df_r[text_col])
    y_r = df_r[raag_col]
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    rm = SVC(kernel='linear', probability=True, random_state=42)
    rm.fit(X_tr_r, y_tr_r)
    y_pred = rm.predict(X_te_r)
    print(f"[Classifiers] Raag — Train: {accuracy_score(y_tr_r, rm.predict(X_tr_r))*100:.1f}%, Val: {accuracy_score(y_te_r, y_pred)*100:.1f}%")

    # Confusion matrix
    cm_data = confusion_matrix(y_te_r, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=rm.classes_, yticklabels=rm.classes_)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.title('Raag Classifier Confusion Matrix (SVM)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'raag_confusion_matrix.png'))
    plt.close()

    joblib.dump(rv, RAAG_VEC_PATH)
    joblib.dump(rm, RAAG_MODEL_PATH)
    print("[Classifiers] All models saved.\n")
    return av, am, rv, rm


@st.cache_resource
def load_classifiers():
    """Load or train classifiers. Cached by Streamlit for the entire session."""
    if all(os.path.exists(p) for p in [AUTHOR_VEC_PATH, AUTHOR_MODEL_PATH,
                                        RAAG_VEC_PATH,   RAAG_MODEL_PATH]):
        print("[Classifiers] Loading saved models from disk...")
        av = joblib.load(AUTHOR_VEC_PATH)
        am = joblib.load(AUTHOR_MODEL_PATH)
        rv = joblib.load(RAAG_VEC_PATH)
        rm = joblib.load(RAAG_MODEL_PATH)
        print("[Classifiers] Loaded successfully.\n")
        return av, am, rv, rm
    return _train_and_save()


def predict_author(text):
    """Predict which Guru/Author wrote the given text."""
    try:
        av, am, _, _ = load_classifiers()
        vec   = av.transform([text])
        label = am.predict(vec)[0]
        proba = float(am.predict_proba(vec).max()) * 100
        return {'author': label, 'confidence': round(proba, 1)}
    except Exception:
        return {'author': 'Unknown', 'confidence': 0.0}


def predict_raag(text):
    """Predict which Raag the given text belongs to."""
    try:
        _, _, rv, rm = load_classifiers()
        vec   = rv.transform([text])
        label = rm.predict(vec)[0]
        proba = float(rm.predict_proba(vec).max()) * 100
        return {'raag': label, 'confidence': round(proba, 1)}
    except Exception:
        return {'raag': 'Unknown', 'confidence': 0.0}
