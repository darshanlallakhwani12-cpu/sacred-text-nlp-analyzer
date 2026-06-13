# config.py
# Centralized configuration for the NLP Analyzer Project
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_DB_PATH = os.path.join(BASE_DIR, 'database', 'sacred_text.db')

# Database Configuration (SQLite)
DB_CONFIG = {
    'database_url': f"sqlite:///{SQLITE_DB_PATH}"
}
