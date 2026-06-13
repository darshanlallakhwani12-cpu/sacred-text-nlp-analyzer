import os
import pandas as pd
from sqlalchemy import create_engine
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.db_connection import get_sqlalchemy_engine

def migrate_to_sqlite():
    mysql_engine = get_sqlalchemy_engine()
    if not mysql_engine:
        print("Failed to connect to MySQL.")
        return

    sqlite_db_path = os.path.join(BASE_DIR, 'database', 'sacred_text.db')
    sqlite_engine = create_engine(f"sqlite:///{sqlite_db_path}")

    tables_to_migrate = ['gurbani', 'quran', 'bible', 'balanced_corpus', 'history']

    print(f"Migrating data from MySQL to SQLite ({sqlite_db_path})...")

    for table in tables_to_migrate:
        try:
            print(f"Exporting table '{table}' from MySQL...")
            df = pd.read_sql(f"SELECT * FROM {table}", con=mysql_engine)
            print(f"Importing {len(df)} rows into SQLite '{table}'...")
            df.to_sql(table, con=sqlite_engine, if_exists='replace', index=False)
            print(f"Table '{table}' migrated successfully.\n")
        except Exception as e:
            print(f"Error migrating table '{table}': {e}\n")

    print("Migration to SQLite Complete! You can now switch the app to use SQLite.")

if __name__ == "__main__":
    migrate_to_sqlite()
