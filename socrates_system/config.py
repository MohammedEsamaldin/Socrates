"""
Configuration settings for the Socrates Agent System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Knowledge Graph settings
KG_DATABASE_PATH = DATA_DIR / "knowledge_graph.db"
SESSION_KG_PATH = DATA_DIR / "session_kg.json"

# External API settings
WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
SEARCH_API_URL = "https://api.duckduckgo.com/"

# Remote AGLA API settings (if set, Socrates will call the remote service instead of local AGLA models)
AGLA_API_URL = os.getenv('AGLA_API_URL', '').strip()  # e.g., "https://<modal-app>.modal.run"
AGLA_API_VERIFY_PATH = os.getenv('AGLA_API_VERIFY_PATH', '/verify')
try:
    AGLA_API_TIMEOUT = int(os.getenv('AGLA_API_TIMEOUT', '120'))
except Exception:
    AGLA_API_TIMEOUT = 120

# API Keys (set these as environment variables or update directly)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # For OpenAI API if needed
GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')  # For Google Fact Check API
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')  # For Claude API if needed

# Alternative: Set keys directly (NOT recommended for production)
# OPENAI_API_KEY = "your-openai-api-key-here"
# GOOGLE_FACT_CHECK_API_KEY = "your-google-fact-check-api-key-here"

# Multimodal model settings
VISION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
# Stronger semantic encoder for better similarity and canonicalization
NLP_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# SOTA spaCy NER (falls back in code if not installed)
ENTITY_MODEL_NAME = "en_core_web_trf"

# Socrates Agent settings
MAX_SOCRATIC_QUESTIONS = 5
CONFIDENCE_THRESHOLD = 0.7
CONTRADICTION_THRESHOLD = 0.3
SIMILARITY_THRESHOLD = 0.4
CATEGORIZATION_CONFIDENCE_THRESHOLD = 0.75

# Flask settings
UPLOAD_FOLDER = DATA_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Logging configuration
LOG_LEVEL = "INFO"  # File logs level
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Console log level (controls how much appears in the terminal). Keep important only by default.
CONSOLE_LOG_LEVEL = "WARNING"
