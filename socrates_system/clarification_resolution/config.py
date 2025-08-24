from __future__ import annotations

import os

# Clarification module feature flag
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

CLARIFICATION_ENABLED: bool = _env_bool("CLARIFICATION_ENABLED", True)

# Development mode enables manual overrides and verbose logging
DEV_MODE_DEFAULT: bool = _env_bool("CLARIFICATION_DEV_MODE", True)

# Dialogue and timing
MAX_QUESTIONS_PER_SESSION: int = 1
RESPONSE_TIMEOUT_SECONDS: int = 120

# Confidence thresholds
MIN_RESOLUTION_CONFIDENCE: float = 0.6
HIGH_CONFIDENCE_THRESHOLD: float = 0.8
LOW_CONFIDENCE_THRESHOLD: float = 0.4

# Routing defaults per issue type
DEFAULT_NEXT_ACTION = {
    "VISUAL_CONFLICT": "REVERIFY_PIPELINE",
    "KNOWLEDGE_CONTRADICTION": "REVERIFY_PIPELINE",
    "AMBIGUITY": "REVERIFY_PIPELINE",
    "EXTERNAL_FACTUAL_CONFLICT": "REVERIFY_PIPELINE",
}

# LLM prompt knobs
REFINE_QUESTIONS_WITH_LLM: bool = _env_bool("REFINE_QUESTIONS_WITH_LLM", True)
CORRECT_CLAIM_WITH_LLM: bool = _env_bool("CORRECT_CLAIM_WITH_LLM", True)
REQUIRE_USER_REWRITE: bool = _env_bool("REQUIRE_USER_REWRITE", False)

# Safety and formatting
MAX_CORRECTED_CLAIM_TOKENS: int = 100

# Selective token replacement controls
SELECTIVE_TOKEN_REPLACEMENT: bool = _env_bool("SELECTIVE_TOKEN_REPLACEMENT", True)
# How much the corrected claim can differ in characters before we reject wholesale replacement
SELECTIVE_MAX_CHAR_DIFF_RATIO: float = _env_float("SELECTIVE_MAX_CHAR_DIFF_RATIO", 0.4)
# Fraction of tokens in the original claim that may change (replace/delete/insert) before we reject
SELECTIVE_MAX_TOKEN_CHANGE_RATIO: float = _env_float("SELECTIVE_MAX_TOKEN_CHANGE_RATIO", 0.5)
# Allow insertions/deletions beyond direct replacements
SELECTIVE_ALLOW_INSERTIONS: bool = _env_bool("SELECTIVE_ALLOW_INSERTIONS", False)
# Ignore selective replacement for very short claims (token count below this)
SELECTIVE_MIN_TOKENS: int = _env_int("SELECTIVE_MIN_TOKENS", 3)

