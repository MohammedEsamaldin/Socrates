"""
Configuration constants for the clarification resolution module.

All values are sourced from the central AppConfig (which reads environment
variables once at startup). Modules inside this package reference these
names directly (e.g. ``clar_cfg.REFINE_QUESTIONS_WITH_LLM``).
"""
from __future__ import annotations

from socrates_system.config import get_app_config as _get_cfg

def _cfg():
    return _get_cfg()

# Clarification module feature flag
CLARIFICATION_ENABLED: bool = _cfg().clarification_enabled

# Development mode enables manual overrides and verbose logging
DEV_MODE_DEFAULT: bool = _cfg().clarification_dev_mode

# Dialogue and timing (hardcoded; not operator-tuneable via env)
MAX_QUESTIONS_PER_SESSION: int = 1
RESPONSE_TIMEOUT_SECONDS: int = 120

# Confidence thresholds (hardcoded algorithmic constants)
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
REFINE_QUESTIONS_WITH_LLM: bool = _cfg().clarification_refine_questions
CORRECT_CLAIM_WITH_LLM: bool = _cfg().clarification_correct_claim
REQUIRE_USER_REWRITE: bool = _cfg().clarification_require_user_rewrite

# Safety and formatting (hardcoded)
MAX_CORRECTED_CLAIM_TOKENS: int = 100

# Selective token replacement controls
SELECTIVE_TOKEN_REPLACEMENT: bool = _cfg().clarification_selective_token_replacement
SELECTIVE_MAX_CHAR_DIFF_RATIO: float = _cfg().clarification_max_char_diff_ratio
SELECTIVE_MAX_TOKEN_CHANGE_RATIO: float = _cfg().clarification_max_token_change_ratio
SELECTIVE_ALLOW_INSERTIONS: bool = _cfg().clarification_allow_insertions
SELECTIVE_MIN_TOKENS: int = _cfg().clarification_min_tokens
