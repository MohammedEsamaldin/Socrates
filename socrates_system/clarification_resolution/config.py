from __future__ import annotations

# Clarification module feature flag
CLARIFICATION_ENABLED: bool = True

# Development mode enables manual overrides and verbose logging
DEV_MODE_DEFAULT: bool = True

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
REFINE_QUESTIONS_WITH_LLM: bool = True
CORRECT_CLAIM_WITH_LLM: bool = False
REQUIRE_USER_REWRITE: bool = True

# Safety and formatting
MAX_CORRECTED_CLAIM_TOKENS: int = 100

