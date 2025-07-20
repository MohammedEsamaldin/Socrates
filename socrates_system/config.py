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

# Multimodal model settings
VISION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
NLP_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENTITY_MODEL_NAME = "en_core_web_sm"

# Socrates Agent settings
MAX_SOCRATIC_QUESTIONS = 5
CONFIDENCE_THRESHOLD = 0.7
CONTRADICTION_THRESHOLD = 0.3

# Flask settings
UPLOAD_FOLDER = DATA_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
