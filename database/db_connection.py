import mysql.connector
from sqlalchemy import create_engine, text
import urllib.parse
from database.config import DB_CONFIG

SETUP_SQL = """
CREATE TABLE IF NOT EXISTS gurbani (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ang INT,
    gurmukhi LONGTEXT,
    english LONGTEXT,
    transliteration LONGTEXT,
    raag VARCHAR(200),
    writer VARCHAR(200)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS quran (
    id INT AUTO_INCREMENT PRIMARY KEY,
    surah_no INT,
    ayah_no INT,
    arabic LONGTEXT,
    english LONGTEXT
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS bible (
    id INT AUTO_INCREMENT PRIMARY KEY,
    book VARCHAR(100),
    chapter INT,
    verse INT,
    text LONGTEXT
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS balanced_corpus (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source VARCHAR(50),
    text LONGTEXT,
    original_ref VARCHAR(200),
    author_or_raag VARCHAR(200)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(50),
    timestamp DATETIME,
    input_text LONGTEXT,
    language VARCHAR(50),
    sentiment VARCHAR(50),
    emotion VARCHAR(50),
    similar_verse LONGTEXT
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""

def _ensure_database_exists():
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cursor.execute(f"USE {DB_CONFIG['database']};")
        for statement in SETUP_SQL.strip().split(';'):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as err:
        print(f"[DB] Setup Error: {err}")
        return False

def get_db_connection():
    """Returns a raw mysql-connector-python connection object."""
    _ensure_database_exists()
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL Database: {err}")
        return None

def get_sqlalchemy_engine():
    """Returns a SQLAlchemy engine object for pandas read_sql / to_sql."""
    _ensure_database_exists()
    try:
        password = urllib.parse.quote_plus(DB_CONFIG['password'])
        db_url = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{password}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_url)
        return engine
    except Exception as err:
        print(f"Error creating SQLAlchemy Engine: {err}")
        return None

if __name__ == "__main__":
    print("Testing MySQL connection...")
    conn = get_db_connection()
    if conn and conn.is_connected():
        print("[OK] Connected via mysql-connector-python!")
        conn.close()
    else:
        print("[FAIL] Could not connect via mysql-connector-python.")

    engine = get_sqlalchemy_engine()
    if engine:
        try:
            with engine.connect() as connection:
                connection.execute(text('SELECT 1'))
                print("[OK] Connected via SQLAlchemy Engine!")
        except Exception as e:
            print(f"[FAIL] SQLAlchemy Engine error: {e}")
