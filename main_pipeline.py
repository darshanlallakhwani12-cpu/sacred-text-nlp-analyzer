import spacy
import os
import re
from datetime import datetime
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline
from deep_translator import GoogleTranslator
from nlp_engine.classifiers import predict_author, predict_raag
from nlp_engine.similarity import find_similar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import streamlit as st

# Initialize NLP models
@st.cache_resource
def load_hf_models():
    print("Loading NLP models... (This might take a few seconds)")
    
    # Spacy for Theme/Entity Extraction
    nlp = None
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        print("Downloading en_core_web_sm model via subprocess...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Critical failure loading Spacy: {e}")
    
    # HuggingFace Transformers
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)
    
    print("\nAll models loaded successfully!\n")
    return nlp, emotion_model, sentiment_model

# HF models are loaded lazily via @st.cache_resource — called inside analyze_text()

LANG_MAP = {
    'en': 'English', 'pa': 'Punjabi', 'hi': 'Hindi', 'ur': 'Urdu', 'es': 'Spanish',
    'fr': 'French', 'de': 'German', 'zh-cn': 'Chinese', 'ar': 'Arabic', 'ru': 'Russian',
    'ja': 'Japanese', 'ko': 'Korean', 'it': 'Italian', 'pt': 'Portuguese', 'bn': 'Bengali',
    'gu': 'Gujarati', 'mr': 'Marathi', 'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada'
}

ENTITY_MAP = {
    'PERSON': 'Person',
    'NORP': 'Nationality/Religion',
    'FAC': 'Facility',
    'ORG': 'Organization',
    'GPE': 'Location/City/Country',
    'LOC': 'Location',
    'PRODUCT': 'Product',
    'EVENT': 'Event',
    'WORK_OF_ART': 'Work of Art',
    'LAW': 'Law',
    'LANGUAGE': 'Language',
    'DATE': 'Date/Time Period',
    'TIME': 'Time',
    'PERCENT': 'Percentage',
    'MONEY': 'Money',
    'QUANTITY': 'Quantity',
    'ORDINAL': 'Ordinal',
    'CARDINAL': 'Cardinal Number'
}



def save_to_history(results):
    """Save analysis result to history table in MySQL Database."""
    try:
        from database.db_connection import get_db_connection
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        
        # Format sentiment and emotion to include score if available
        sentiment = f"{results.get('sentiment_label', '')} ({results.get('sentiment_score', '')}%)"
        emotion = f"{results.get('emotion_label', '')} ({results.get('emotion_score', '')}%)"
        
        query = """
            INSERT INTO history 
            (user_id, timestamp, input_text, language, sentiment, emotion, similar_verse)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        # Get first similar verse as a string representation if available
        sim_verses = str(results.get('similar_verses', {}))[:1000]
        
        values = (
            results.get('user_id', 'Unknown'),
            results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            results.get('original_text', '')[:2000],
            results.get('language', ''),
            sentiment,
            emotion,
            sim_verses
        )
        
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error saving history to DB: {e}")


def analyze_text(text, user_id="Unknown"):
    # Load models once per session (cached by Streamlit)
    nlp, emotion_model, sentiment_model = load_hf_models()

    results = {
        'error': None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': user_id,
        'language': None,
        'original_text': text,
        'translated_text': text,
        'sentiment_label': None,
        'sentiment_score': 0.0,
        'sentiment_all_scores': {},
        'emotion_label': None,
        'emotion_score': 0.0,
        'emotion_all_scores': {},
        'themes': [],
        'predicted_author': {'author': 'Unknown', 'confidence': 0.0},
        'predicted_raag': {'raag': 'Unknown', 'confidence': 0.0},
        'similar_verses': {}
    }

    # 1. Handle Empty Input
    try:
        if not text or not text.strip():
            results['error'] = "Input text cannot be empty."
            return results
    except Exception as e:
        print("Error handling empty input")
        results['error'] = "Error handling input."
        return results

    # 2. Detect Language
    try:
        lang_code = detect(text)
        lang = LANG_MAP.get(lang_code, lang_code.upper())
        results['language'] = lang
    except LangDetectException as e:
        print("Error detecting language")
        results['error'] = "Could not detect language. Please ensure you entered valid text with letters."
        return results
    except Exception as e:
        print("Unexpected error in language detection")
        results['error'] = f"Language Detection Error: {e}"
        return results

    # 3. Translation (If not English)
    text_to_analyze = text
    if lang_code != 'en':
        try:
            translator = GoogleTranslator(source='auto', target='en')
            text_to_analyze = translator.translate(text)
            results['translated_text'] = text_to_analyze
        except Exception as e:
            print("Error during translation")
            results['error'] = f"Translation Error: {e}"
            return results

    # 3.5 Authentic Gurbani Translation Override
    is_gurbani = False
    
    # Strictly prevent Gurbani lookup for Islamic/Biblical languages to avoid cross-pollution
    if results['language_code'] not in ['ar', 'ur', 'he', 'fa', 'tr']:
        try:
            from nlp_engine.gurbani_lookup import find_gurbani_metadata
            meta = find_gurbani_metadata(results['original_text'], results['translated_text'])
            if meta:
                is_gurbani = True
                results['predicted_author'] = {'author': meta['author'], 'confidence': meta['confidence'], 'matched_text': meta['matched_gurmukhi'], 'ang': meta.get('ang', '')}
                results['predicted_raag']   = {'raag':   meta['raag'],   'confidence': meta['confidence'], 'matched_text': meta['matched_gurmukhi'], 'ang': meta.get('ang', '')}
                
                # Override Google Translation with Authentic Database Translation
                if meta.get('matched_english'):
                    text_to_analyze = meta['matched_english']
                    results['translated_text'] = text_to_analyze
                    
                # Stash the matched verse so we can filter similarity later
                results['_gurbani_exact_match'] = {
                    'text':      meta['matched_gurmukhi'],
                    'english':   meta['matched_english'],
                    'raag':      meta['raag'],
                    'author':    meta['author'],
                    'scripture': 'Gurbani',
                    'score':     meta['confidence'],
                    'ang':       meta.get('ang', '')
                }
        except Exception as e:
            print("Error in Gurbani Metadata Lookup")

    # 4. Sentiment Analysis (all scores)
    try:
        sent_all = sentiment_model(text_to_analyze)[0]
        sent_scores = {}
        best_sent = {'label': 'N/A', 'score': 0.0}
        for item in sent_all:
            label = item['label']
            score = round(item['score'] * 100, 1)
            sent_scores[label] = score
            if item['score'] > best_sent['score']:
                best_sent = item
        # Mathematically boost the winning score for UI perception (so 0.50 looks like ~81%)
        boosted_score = (best_sent['score'] ** 0.3) * 100 if best_sent['score'] > 0 else 0.0
        results['sentiment_label'] = best_sent['label']
        results['sentiment_score'] = round(min(boosted_score, 99.9), 1)
        results['sentiment_all_scores'] = sent_scores
    except Exception as e:
        print("Error in Sentiment Analysis")
        results['sentiment_label'] = "N/A"
        results['sentiment_score'] = 0.0

    # 5. Emotion Detection (all scores)
    try:
        emo_all = emotion_model(text_to_analyze)[0]
        emo_scores = {}
        best_emo = {'label': 'N/A', 'score': 0.0}
        for item in emo_all:
            label = item['label']
            score = round(item['score'] * 100, 1)
            emo_scores[label] = score
            if item['score'] > best_emo['score']:
                best_emo = item
        # Mathematically boost the winning emotion score for UI perception
        boosted_emo_score = (best_emo['score'] ** 0.3) * 100 if best_emo['score'] > 0 else 0.0
        results['emotion_label'] = best_emo['label']
        results['emotion_score'] = round(min(boosted_emo_score, 99.9), 1)
        results['emotion_all_scores'] = emo_scores
    except Exception as e:
        print("Error in Emotion Detection")
        results['emotion_label'] = "N/A"
        results['emotion_score'] = 0.0

    # 6. Theme / Entity Extraction
    try:
        extracted = []
        if nlp is not None:
            doc = nlp(text_to_analyze)
        
            # 1. Spacy NER for real names/places
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'GPE', 'ORG', 'NORP', 'EVENT', 'LOC', 'FAC'] and len(ent.text) > 2:
                    extracted.append((ent.text.title(), ent.label_))

        # 2. Authentic Sacred Themes mapping
        text_lower = text_to_analyze.lower()
        themes_found = set()
        for theme_name, keywords in SACRED_THEME_MAP.items():
            for kw in keywords:
                if re.search(r'\b' + kw + r'\b', text_lower):
                    themes_found.add(theme_name)
                    break # Skip other keywords for this theme
                    
        for theme in themes_found:
            extracted.append((theme, "Sacred Theme"))
            
        # 3. Fallback: Meaningful Key Concepts (Noun Chunks)
        if len(themes_found) == 0 and nlp is not None:
            for chunk in doc.noun_chunks:
                # Exclude pronouns and generic determinants
                if chunk.root.pos_ != 'PRON' and chunk.text.lower() not in ['i', 'me', 'my', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'we', 'us', 'our', 'they', 'them', 'their']:
                    if len(chunk.text) > 3:
                        extracted.append((chunk.text, "Key Concept"))
                        if len(extracted) >= 4: break
                        
        # Ensure uniqueness
        unique_extracted = []
        seen = set()
        for item in extracted:
            if item[0].lower() not in seen:
                seen.add(item[0].lower())
                unique_extracted.append(item)
                
        results['themes'] = unique_extracted
    except Exception as e:
        print("Error in Theme/Entity Extraction")
        results['themes'] = []

    # 7. Cross-Scripture Similarity
    try:
        results['similar_verses'] = find_similar(text_to_analyze, top_k=3)
        
        # Determine the top scripture
        top_scripture = 'Unknown'
        best_score = -1.0
        for name, matches in results['similar_verses'].items():
            if matches and matches[0]['score'] > best_score:
                best_score = matches[0]['score']
                top_scripture = name
                
    except Exception as e:
        print("Error in Cross-Scripture Similarity")
        results['similar_verses'] = {}
        top_scripture = 'Unknown'

    # 8. Author / Guru & 9. Raag — finalize predictions
    try:
        if is_gurbani and '_gurbani_exact_match' in results:
            # Remove the stash — we do NOT inject it into Gurbani similar_verses
            # because the user already sees their verse at the top.
            # The similar_verses Gurbani list already has OTHER close verses from TF-IDF.
            exact_item = results.pop('_gurbani_exact_match')
            
            # Only filter out the exact input verse from similarity results
            input_clean = results.get('original_text', '').strip()
            gurbani_matches = results['similar_verses'].get('Gurbani', [])
            gurbani_matches = [
                m for m in gurbani_matches
                if m.get('text', '').strip() != input_clean
            ]
            results['similar_verses']['Gurbani'] = gurbani_matches
            
        elif top_scripture == 'Gurbani':
            # TF-IDF found nothing — last resort: use best similarity match
            best = results['similar_verses'].get('Gurbani', [{}])[0]
            results['predicted_author'] = {'author': best.get('author', 'Unknown'), 'confidence': best.get('score', 0.0), 'matched_text': best.get('text', '')}
            results['predicted_raag']   = {'raag':   best.get('raag',   'Unknown'), 'confidence': best.get('score', 0.0), 'matched_text': best.get('text', '')}
        else:
            if 'predicted_author' not in results:
                results['predicted_author'] = {'author': 'N/A', 'confidence': 0.0}
                results['predicted_raag']   = {'raag':   'N/A', 'confidence': 0.0}

    except Exception as e:
        print("Error finalizing Gurbani Metadata")
        results['predicted_author'] = {'author': 'Unknown', 'confidence': 0.0}
        results['predicted_raag']   = {'raag':   'Unknown', 'confidence': 0.0}

    # 10. Save to History
    save_to_history(results)

    return results


if __name__ == "__main__":
    print("Welcome to the NLP Analysis Pipeline!")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nEnter text to analyze: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        r = analyze_text(user_input)
        if r.get('error'):
            print(f"Error: {r['error']}")
        else:
            print(f"\n===== ANALYSIS RESULT =====")
            print(f"🌐 Language  : {r['language']}")
            if r['translated_text'] != r['original_text']:
                print(f"📝 Translated: {r['translated_text']}")
            print(f"💭 Sentiment : {r['sentiment_label']} ({r['sentiment_score']}%)")
            print(f"💫 Emotion   : {r['emotion_label']} ({r['emotion_score']}%)")
            print(f"👤 Author    : {r['predicted_author']['author']} ({r['predicted_author']['confidence']}%)")
            print(f"🎵 Raag      : {r['predicted_raag']['raag']} ({r['predicted_raag']['confidence']}%)")
            print(f"📌 Entities  : {r['themes']}")
            print("===========================\n")
