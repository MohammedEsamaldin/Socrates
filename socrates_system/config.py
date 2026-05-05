"""
Configuration settings for the Socrates Agent System.

All environment-variable reads live exclusively in this module.
Every other module receives config values via get_app_config() or
constructor arguments — never by calling os.getenv() directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Filesystem layout (not env-sourced)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for _dir in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    _dir.mkdir(exist_ok=True)

KG_DATABASE_PATH = DATA_DIR / "knowledge_graph.db"
SESSION_KG_PATH = DATA_DIR / "session_kg.json"

WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
SEARCH_API_URL = "https://api.duckduckgo.com/"

# Multimodal model names (hardware-dependent; not from env)
VISION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
NLP_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
ENTITY_MODEL_NAME = "en_core_web_trf"

# Hardcoded thresholds (algorithmic, not operator-tuneable via env)
MAX_SOCRATIC_QUESTIONS = 5
CONFIDENCE_THRESHOLD = 0.7
CONTRADICTION_THRESHOLD = 0.3
SIMILARITY_THRESHOLD = 0.4
CATEGORIZATION_CONFIDENCE_THRESHOLD = 0.75

# Flask settings
UPLOAD_FOLDER = DATA_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ---------------------------------------------------------------------------
# Private env-reading helpers (only called inside get_app_config)
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_opt_float(key: str) -> Optional[float]:
    v = os.getenv(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _env_opt_int(key: str) -> Optional[int]:
    v = os.getenv(key)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _env_opt_bool(key: str) -> Optional[bool]:
    v = os.getenv(key)
    if v is None:
        return None
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# AppConfig — typed, frozen snapshot of all env-sourced configuration.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """All environment-sourced runtime configuration in one place.

    Fields are grouped by concern. Each field documents its env variable.
    Constructed once by get_app_config() and cached for the process lifetime.

    Tests that need different values should call get_app_config.cache_clear()
    before mutating environment variables, then call get_app_config() again.
    """

    # LLM backend
    llm_provider: str           # SOC_LLM_PROVIDER
    llm_model: Optional[str]    # SOC_LLM_MODEL
    ollama_base_url: str        # OLLAMA_BASE_URL / OLLAMA_URL
    openai_api_key: str         # OPENAI_API_KEY
    openai_base_url: str        # OPENAI_BASE_URL / OPENAI_API_BASE
    anthropic_api_key: str      # ANTHROPIC_API_KEY
    anthropic_base_url: str     # ANTHROPIC_BASE_URL

    # Per-module LLM overrides
    factuality_llm_provider: Optional[str]  # FACTUALITY_LLM_PROVIDER
    factuality_llm_model: Optional[str]     # FACTUALITY_LLM_MODEL
    self_llm_provider: Optional[str]        # SELF_LLM_PROVIDER
    self_llm_model: Optional[str]           # SELF_LLM_MODEL

    # Remote AGLA API
    agla_api_url: str           # AGLA_API_URL
    agla_api_verify_path: str   # AGLA_API_VERIFY_PATH
    agla_api_timeout: int       # AGLA_API_TIMEOUT

    # Pipeline behavior
    session_id: Optional[str]               # SOC_SESSION_ID
    conflict_resolution_mode: str           # SOC_CONFLICT_MODE
    router_mode: str                        # SOC_ROUTER_MODE
    factuality_enabled: bool                # FACTUALITY_ENABLED
    factuality_context_mode: str            # FACTUALITY_CONTEXT_MODE
    factuality_context_max_items: int       # FACTUALITY_CONTEXT_MAX_ITEMS
    factuality_max_retries: int             # FACTUALITY_MAX_RETRIES
    factuality_timeout: float               # FACTUALITY_TIMEOUT
    factuality_backoff: float               # FACTUALITY_BACKOFF
    clarification_enabled: bool             # CLARIFICATION_ENABLED
    clarification_dev_mode: bool            # CLARIFICATION_DEV_MODE
    post_factuality_clarification: bool     # SOC_POST_FACTUALITY_CLAR
    question_gen_enabled: bool              # QUESTION_GEN_ENABLED
    qg_min_confidence: Optional[float]      # QG_MIN_CONFIDENCE_THRESHOLD
    qg_max_complexity: Optional[float]      # QG_MAX_COMPLEXITY_RATIO
    qg_enable_fallback: Optional[bool]      # QG_ENABLE_FALLBACK
    qg_prioritize_visual: Optional[bool]    # QG_PRIORITIZE_VISUAL
    qg_questions_per_category: Optional[int]  # QG_QUESTIONS_PER_CATEGORY

    # Behavioral flags
    allow_polarity_flip: bool   # SOC_ALLOW_POLARITY_FLIP
    max_evidence_log: int       # SOC_MAX_EVIDENCE_LOG
    kg_max_items: int           # SOC_KG_MAX_ITEMS
    show_kg: bool               # SOC_SHOW_KG
    no_color: bool              # NO_COLOR
    log_level: str              # SOC_LOG_LEVEL
    console_log_level: str      # CONSOLE_LOG_LEVEL

    # External API keys
    google_fact_check_api_key: str  # GOOGLE_FACT_CHECK_API_KEY
    google_factcheck_api_key: str   # GOOGLE_FACTCHECK_API_KEY (alias used by some clients)
    tavily_api_key: str             # TAVILY_API_KEY
    openai_model: str               # OPENAI_MODEL

    # LLaVA-specific toggles
    llava_no_4bit: bool         # SOC_LLAVA_NO_4BIT
    llava_slow_tokenizer: bool  # SOC_LLAVA_SLOW_TOKENIZER
    llava_orig_use_cli: bool    # SOC_LLAVA_ORIG_USE_CLI
    llava_conv_template: str    # SOC_LLAVA_CONV_TEMPLATE
    llava_timeout_sec: int      # SOC_LLAVA_TIMEOUT_SEC

    # Clarification module knobs
    clarification_refine_questions: bool            # REFINE_QUESTIONS_WITH_LLM
    clarification_correct_claim: bool               # CORRECT_CLAIM_WITH_LLM
    clarification_require_user_rewrite: bool        # REQUIRE_USER_REWRITE
    clarification_selective_token_replacement: bool # SELECTIVE_TOKEN_REPLACEMENT
    clarification_max_char_diff_ratio: float        # SELECTIVE_MAX_CHAR_DIFF_RATIO
    clarification_max_token_change_ratio: float     # SELECTIVE_MAX_TOKEN_CHANGE_RATIO
    clarification_allow_insertions: bool            # SELECTIVE_ALLOW_INSERTIONS
    clarification_min_tokens: int                   # SELECTIVE_MIN_TOKENS


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Read all environment configuration once and return a frozen AppConfig.

    Subsequent calls return the cached instance. To reload (e.g. in tests),
    call get_app_config.cache_clear() before re-calling this function.
    """
    return AppConfig(
        # LLM backend
        llm_provider=os.getenv("SOC_LLM_PROVIDER", "ollama").lower(),
        llm_model=os.getenv("SOC_LLM_MODEL") or None,
        ollama_base_url=(
            os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL") or "http://localhost:11434"
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=(
            os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        ),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
        # Per-module LLM overrides
        factuality_llm_provider=os.getenv("FACTUALITY_LLM_PROVIDER") or None,
        factuality_llm_model=os.getenv("FACTUALITY_LLM_MODEL") or None,
        self_llm_provider=os.getenv("SELF_LLM_PROVIDER") or None,
        self_llm_model=os.getenv("SELF_LLM_MODEL") or None,
        # Remote AGLA API
        agla_api_url=os.getenv("AGLA_API_URL", "").strip(),
        agla_api_verify_path=os.getenv("AGLA_API_VERIFY_PATH", "/verify"),
        agla_api_timeout=_env_int("AGLA_API_TIMEOUT", 120),
        # Pipeline behavior
        session_id=os.getenv("SOC_SESSION_ID") or None,
        conflict_resolution_mode=os.getenv("SOC_CONFLICT_MODE", "auto"),
        router_mode=os.getenv("SOC_ROUTER_MODE", "llm"),
        factuality_enabled=_env_bool("FACTUALITY_ENABLED", True),
        factuality_context_mode=os.getenv("FACTUALITY_CONTEXT_MODE", "SOCRATIC_QUESTIONS"),
        factuality_context_max_items=_env_int("FACTUALITY_CONTEXT_MAX_ITEMS", 6),
        factuality_max_retries=_env_int("FACTUALITY_MAX_RETRIES", 2),
        factuality_timeout=_env_float("FACTUALITY_TIMEOUT", 6.0),
        factuality_backoff=_env_float("FACTUALITY_BACKOFF", 0.5),
        clarification_enabled=_env_bool("CLARIFICATION_ENABLED", True),
        clarification_dev_mode=_env_bool("CLARIFICATION_DEV_MODE", False),
        post_factuality_clarification=_env_bool("SOC_POST_FACTUALITY_CLAR", True),
        question_gen_enabled=_env_bool("QUESTION_GEN_ENABLED", True),
        qg_min_confidence=_env_opt_float("QG_MIN_CONFIDENCE_THRESHOLD"),
        qg_max_complexity=_env_opt_float("QG_MAX_COMPLEXITY_RATIO"),
        qg_enable_fallback=_env_opt_bool("QG_ENABLE_FALLBACK"),
        qg_prioritize_visual=_env_opt_bool("QG_PRIORITIZE_VISUAL"),
        qg_questions_per_category=_env_opt_int("QG_QUESTIONS_PER_CATEGORY"),
        # Behavioral flags
        allow_polarity_flip=_env_bool("SOC_ALLOW_POLARITY_FLIP", False),
        max_evidence_log=_env_int("SOC_MAX_EVIDENCE_LOG", 2),
        kg_max_items=_env_int("SOC_KG_MAX_ITEMS", 10),
        show_kg=_env_bool("SOC_SHOW_KG", False),
        no_color=bool(os.getenv("NO_COLOR")),
        log_level=os.getenv("SOC_LOG_LEVEL", "INFO").upper(),
        console_log_level=os.getenv("CONSOLE_LOG_LEVEL", "WARNING").upper(),
        # External API keys
        google_fact_check_api_key=os.getenv("GOOGLE_FACT_CHECK_API_KEY", ""),
        google_factcheck_api_key=os.getenv("GOOGLE_FACTCHECK_API_KEY", ""),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        # LLaVA toggles
        llava_no_4bit=_env_bool("SOC_LLAVA_NO_4BIT", False),
        llava_slow_tokenizer=_env_bool("SOC_LLAVA_SLOW_TOKENIZER", False),
        llava_orig_use_cli=_env_bool("SOC_LLAVA_ORIG_USE_CLI", False),
        llava_conv_template=os.getenv("SOC_LLAVA_CONV_TEMPLATE", "llava_v1"),
        llava_timeout_sec=_env_int("SOC_LLAVA_TIMEOUT_SEC", 180),
        # Clarification knobs
        clarification_refine_questions=_env_bool("REFINE_QUESTIONS_WITH_LLM", True),
        clarification_correct_claim=_env_bool("CORRECT_CLAIM_WITH_LLM", True),
        clarification_require_user_rewrite=_env_bool("REQUIRE_USER_REWRITE", False),
        clarification_selective_token_replacement=_env_bool("SELECTIVE_TOKEN_REPLACEMENT", True),
        clarification_max_char_diff_ratio=_env_float("SELECTIVE_MAX_CHAR_DIFF_RATIO", 0.4),
        clarification_max_token_change_ratio=_env_float("SELECTIVE_MAX_TOKEN_CHANGE_RATIO", 0.5),
        clarification_allow_insertions=_env_bool("SELECTIVE_ALLOW_INSERTIONS", False),
        clarification_min_tokens=_env_int("SELECTIVE_MIN_TOKENS", 3),
    )


# ---------------------------------------------------------------------------
# Backward-compatible module-level names — delegate to AppConfig.
# Code that imports these names directly continues to work unchanged.
# ---------------------------------------------------------------------------
_c = get_app_config()
AGLA_API_URL = _c.agla_api_url
AGLA_API_VERIFY_PATH = _c.agla_api_verify_path
AGLA_API_TIMEOUT = _c.agla_api_timeout
OPENAI_API_KEY = _c.openai_api_key
GOOGLE_FACT_CHECK_API_KEY = _c.google_fact_check_api_key
ANTHROPIC_API_KEY = _c.anthropic_api_key
LOG_LEVEL = _c.log_level
CONSOLE_LOG_LEVEL = _c.console_log_level
