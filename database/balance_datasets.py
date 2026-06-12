import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_connection import get_sqlalchemy_engine

def balance_datasets():
    print("Loading datasets from MySQL Database...")
    engine = get_sqlalchemy_engine()
    if not engine:
        print("Database connection failed. Exiting.")
        return

    # Load Gurbani
    print("Loading Gurbani...")
    try:
        df_g = pd.read_sql('SELECT * FROM gurbani', con=engine)
        df_gurbani = pd.DataFrame({
            'text': df_g['english'].fillna(''),
            'source': 'Gurbani',
            'original_ref': 'Ang ' + df_g['ang'].astype(str),
            'author_or_raag': df_g['writer'].fillna('Unknown')
        })
    except Exception as e:
        print(f"Error loading Gurbani from DB: {e}")
        df_gurbani = pd.DataFrame()

    # Load Quran
    print("Loading Quran...")
    try:
        df_q = pd.read_sql('SELECT * FROM quran', con=engine)
        df_quran = pd.DataFrame({
            'text': df_q['english'].fillna(df_q.get('arabic', '')),
            'source': 'Quran',
            'original_ref': 'Surah ' + df_q['surah_no'].astype(str) + ', Ayah ' + df_q['ayah_no'].astype(str),
            'author_or_raag': 'Unknown'
        })
    except Exception as e:
        print(f"Error loading Quran from DB: {e}")
        df_quran = pd.DataFrame()

    # Load Bible
    print("Loading Bible...")
    try:
        df_b = pd.read_sql('SELECT * FROM bible', con=engine)
        df_bible = pd.DataFrame({
            'text': df_b['text'].fillna(''),
            'source': 'Bible',
            'original_ref': 'Book ' + df_b['book'].astype(str) + ', Ch ' + df_b['chapter'].astype(str) + ', V ' + df_b['verse'].astype(str),
            'author_or_raag': 'Unknown'
        })
    except Exception as e:
        print(f"Error loading Bible from DB: {e}")
        df_bible = pd.DataFrame()

    print(f"Gurbani rows: {len(df_gurbani)}")
    print(f"Quran rows:   {len(df_quran)}")
    print(f"Bible rows:   {len(df_bible)}")

    if len(df_gurbani) == 0 or len(df_quran) == 0 or len(df_bible) == 0:
        print("One of the datasets is empty or missing. Please ensure all 3 tables are populated via MySQL Workbench first.")
        return

    min_size = min(len(df_gurbani), len(df_quran), len(df_bible))
    print(f"Smallest dataset: {min_size} — balancing to this size...")

    df_g_bal = df_gurbani.sample(n=min_size, random_state=42)
    df_q_bal = df_quran.sample(n=min_size, random_state=42)
    df_b_bal = df_bible.sample(n=min_size, random_state=42)

    df_balanced = pd.concat([df_g_bal, df_q_bal, df_b_bal]).reset_index(drop=True)

    print("Saving balanced corpus to MySQL database...")
    df_balanced.to_sql('balanced_corpus', con=engine, if_exists='replace', index=False)
    
    print(f"\nDone! balanced_corpus table created with {len(df_balanced)} total rows ({min_size} per scripture).")

if __name__ == "__main__":
    balance_datasets()
