import sqlite3
from sqlalchemy import create_engine
from database.config import DB_CONFIG, SQLITE_DB_PATH

def get_db_connection():
    """Returns a raw sqlite3 connection object."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        return conn
    except sqlite3.Error as err:
        print(f"Error connecting to SQLite Database: {err}")
        return None

def get_sqlalchemy_engine():
    """Returns a SQLAlchemy engine object for pandas read_sql / to_sql."""
    try:
        engine = create_engine(DB_CONFIG['database_url'])
        return engine
    except Exception as err:
        print(f"Error creating SQLAlchemy Engine: {err}")
        return None

if __name__ == "__main__":
    print("Testing SQLite connection...")
    conn = get_db_connection()
    if conn:
        print("[OK] Connected via sqlite3!")
        conn.close()
    else:
        print("[FAIL] Could not connect via sqlite3.")

    engine = get_sqlalchemy_engine()
    if engine:
        try:
            with engine.connect() as connection:
                print("[OK] Connected via SQLAlchemy Engine!")
        except Exception as e:
            print(f"[FAIL] SQLAlchemy Engine error: {e}")
