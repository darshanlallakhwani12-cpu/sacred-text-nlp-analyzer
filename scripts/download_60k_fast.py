import requests
import pandas as pd
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

print("======================================================")
print("  Gurbani Full 60,000+ Verses FAST Downloader (v4) ")
print("         With Ang Numbers & Incremental Save        ")
print("======================================================")
sys.stdout.flush()

API_URL = "https://api.gurbaninow.com/v2/ang/{ang}"
CSV_FILE = "gurbani_with_ang.csv"

def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj: return obj[key]
        for v in obj.values():
            res = find_key(v, key)
            if res is not None: return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_key(item, key)
            if res is not None: return res
    return None

def fetch_ang(ang):
    retries = 3
    while retries > 0:
        try:
            res = requests.get(API_URL.format(ang=ang), timeout=15)
            if res.status_code == 200:
                data = res.json()
                verses = []
                if 'page' in data:
                    for item in data['page']:
                        try:
                            line = item.get('line', {})
                            gurmukhi = line.get('gurmukhi', {}).get('unicode', '')

                            eng_trans = ''
                            if 'translation' in line and 'english' in line['translation']:
                                eng_trans = line['translation']['english'].get('default', '')

                            eng_lit = ''
                            if 'transliteration' in line and 'english' in line['transliteration']:
                                eng_lit = line['transliteration']['english'].get('text', '')

                            author_obj = find_key(item, 'author') or find_key(item, 'writer') or {}
                            writer = author_obj.get('english', 'Unknown') if isinstance(author_obj, dict) else str(author_obj)

                            raag_obj = find_key(item, 'raag') or find_key(item, 'melody') or {}
                            raag = raag_obj.get('english', 'Unknown') if isinstance(raag_obj, dict) else str(raag_obj)

                            ang_num = line.get('ang', ang)

                            if gurmukhi.strip():
                                verses.append({
                                    'gurmukhi':       gurmukhi,
                                    'english':        eng_trans,
                                    'transliteration': eng_lit,
                                    'raag':           raag,
                                    'writer':         writer,
                                    'ang':            ang_num
                                })
                        except Exception:
                            pass
                return verses
            else:
                time.sleep(2)
                retries -= 1
        except Exception:
            time.sleep(2)
            retries -= 1
    return []

print("Starting download... saving incrementally every 100 Angs.")
sys.stdout.flush()

all_verses = []
t0 = time.time()

# Process in chunks of 100 Angs
for chunk_start in range(1, 1431, 100):
    chunk_end = min(chunk_start + 99, 1430)
    chunk_verses = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(fetch_ang, range(chunk_start, chunk_end + 1)))
        
    for res in results:
        if res:
            chunk_verses.extend(res)
            all_verses.extend(res)
            
    # Incremental save
    df_chunk = pd.DataFrame(chunk_verses)
    mode = 'w' if chunk_start == 1 else 'a'
    header = True if chunk_start == 1 else False
    df_chunk.to_csv(CSV_FILE, mode=mode, header=header, index=False, encoding='utf-8')
    
    print(f"Downloaded Angs {chunk_start}-{chunk_end}. Total verses so far: {len(all_verses)}")
    sys.stdout.flush()

elapsed = time.time() - t0
print(f"\n✅ Successfully downloaded {len(all_verses)} verses in {elapsed:.1f} seconds!")
print(f"File saved as: {CSV_FILE}")
sys.stdout.flush()
