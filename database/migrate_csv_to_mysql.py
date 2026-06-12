"""
migrate_csv_to_mysql.py
========================
One-time script to import your existing CSV data into MySQL.
Run this AFTER running setup_database.sql in MySQL Workbench.

Usage:
    python migrate_csv_to_mysql.py
"""
import pandas as pd
import os
import sys

# Add project root to path so we can import from 'database' package
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_connection import get_sqlalchemy_engine

def migrate():
    engine = get_sqlalchemy_engine()
    if not engine:
        print("ERROR: Could not connect to database. Check config.py credentials.")
        return

    print("=" * 60)
    print("  Sacred Text DB Migration: CSV -> MySQL")
    print("=" * 60)

    # ── 1. Gurbani ────────────────────────────────────────────────
    gurbani_path = os.path.join(BASE_DIR, 'gurbani.csv')
    if os.path.exists(gurbani_path):
        print("\n[1/3] Importing gurbani.csv -> gurbani table...")
        df = pd.read_csv(gurbani_path, encoding='utf-8')
        # Rename writer -> writer, keep all columns matching the schema
        # Reorder to match table: ang, gurmukhi, english, transliteration, raag, writer
        cols_needed = ['gurmukhi', 'english', 'transliteration', 'raag', 'writer', 'ang']
        df = df[[c for c in cols_needed if c in df.columns]]
        df.to_sql('gurbani', con=engine, if_exists='replace', index=False, chunksize=1000)
        print(f"    Done! {len(df):,} rows imported into `gurbani` table.")
    else:
        print("[1/3] SKIPPED: gurbani.csv not found.")

    # ── 2. Quran ──────────────────────────────────────────────────
    quran_path = os.path.join(BASE_DIR, 'Dataset', 'Arabic-Original.csv')
    if os.path.exists(quran_path):
        print("\n[2/3] Importing Arabic-Original.csv -> quran table...")
        # Parse pipe-delimited file: surah|ayah|arabic
        lines = []
        with open(quran_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|', 2)
                    if len(parts) >= 3:
                        try:
                            lines.append({
                                'surah_no': int(parts[0].strip()),
                                'ayah_no':  int(parts[1].strip()),
                                'arabic':   parts[2].strip(),
                                'english':  ''  # No English in this file; fill via Kaggle dataset or leave blank
                            })
                        except ValueError:
                            pass

        # Check for optional Kaggle English Quran
        kaggle_path = os.path.join(BASE_DIR, 'Dataset', 'Quran_Dataset.csv')
        if os.path.exists(kaggle_path):
            print("    Found Kaggle Quran dataset. Merging English translations...")
            df_k = pd.read_csv(kaggle_path)
            eng_col = next((c for c in ['Yusuf Ali', 'EnglishText', 'english'] if c in df_k.columns), None)
            if eng_col:
                surah_col = next((c for c in ['Surah', 'surah'] if c in df_k.columns), None)
                ayah_col  = next((c for c in ['Ayat', 'ayah'] if c in df_k.columns), None)
                if surah_col and ayah_col:
                    eng_lookup = {(row[surah_col], row[ayah_col]): str(row[eng_col]) for _, row in df_k.iterrows()}
                    for item in lines:
                        item['english'] = eng_lookup.get((item['surah_no'], item['ayah_no']), '')

        df_q = pd.DataFrame(lines)
        df_q.to_sql('quran', con=engine, if_exists='replace', index=False, chunksize=500)
        print(f"    Done! {len(df_q):,} rows imported into `quran` table.")
    else:
        print("[2/3] SKIPPED: Dataset/Arabic-Original.csv not found.")

    # ── 3. Bible ──────────────────────────────────────────────────
    bible_path = os.path.join(BASE_DIR, 'Dataset', 't_bbe.csv')
    if os.path.exists(bible_path):
        print("\n[3/3] Importing t_bbe.csv -> bible table...")
        df_b = pd.read_csv(bible_path, encoding='utf-8')
        # The BBE Bible CSV has columns: id, b, c, v, t
        df_bible = pd.DataFrame({
            'book':    df_b['b'].astype(str),
            'chapter': df_b['c'],
            'verse':   df_b['v'],
            'text':    df_b['t'].fillna('')
        })
        df_bible.to_sql('bible', con=engine, if_exists='replace', index=False, chunksize=1000)
        print(f"    Done! {len(df_bible):,} rows imported into `bible` table.")
    else:
        print("[3/3] SKIPPED: Dataset/t_bbe.csv not found.")

    print("\n" + "=" * 60)
    print("  Migration complete! Now run balance_datasets.py to build")
    print("  the balanced_corpus table in the database.")
    print("=" * 60)

if __name__ == "__main__":
    migrate()
