"""
Data models for the clarification resolution module.

Defines enumerations and dataclasses that form the public API contract
between :class:`ClarificationResolutionModule` and its callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    # Prefer the project's shared ClaimCategoryType if available
    from socrates_system.modules.shared_structures import ClaimCategoryType
except Exception:
    # Fallback lightweight enum to avoid hard dependency at import time
    class ClaimCategoryType(Enum):
        """Claim verification category used when the shared_structures import is unavailable."""

        VISUAL_GROUNDING_REQUIRED = "VISUAL_GROUNDING_REQUIRED"
        EXTERNAL_KNOWLEDGE_REQUIRED = "EXTERNAL_KNOWLEDGE_REQUIRED"
        SELF_CONSISTENCY_REQUIRED = "SELF_CONSISTENCY_REQUIRED"
        AMBIGUOUS_RESOLUTION_REQUIRED = "AMBIGUOUS_RESOLUTION_REQUIRED"
        SUBJECTIVE_OPINION = "SUBJECTIVE_OPINION"
        PROCEDURAL_DESCRIPTIVE = "PROCEDURAL_DESCRIPTIVE"


class IssueType(Enum):
    """The four clarification issues handled by the module."""
    VISUAL_CONFLICT = "VISUAL_CONFLICT"
    KNOWLEDGE_CONTRADICTION = "KNOWLEDGE_CONTRADICTION"
    AMBIGUITY = "AMBIGUITY"
    EXTERNAL_FACTUAL_CONFLICT = "EXTERNAL_FACTUAL_CONFLICT"


class ResolutionAction(Enum):
    """Where the corrected claim should go next."""
    REVERIFY_PIPELINE = "REVERIFY_PIPELINE"
    DIRECT_TO_KG = "DIRECT_TO_KG"
    REJECT_CLAIM = "REJECT_CLAIM"
    NO_ACTION = "NO_ACTION"


@dataclass
class FactCheckResult:
    """Normalized fact-check result format consumed by the module.

    This should be mapped from your pipeline's factuality output.
    """
    verdict: str  # e.g., PASS/FAIL/UNCERTAIN or TRUE/FALSE/INSUFFICIENT_EVIDENCE
    confidence: float = 0.0
    reasoning: Optional[str] = None  # LLM opinion / explanation
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # [{summary, url, source, ...}]
    sources: List[str] = field(default_factory=list)


@dataclass
class ClarificationContext:
    """Input payload for the ClarificationResolutionModule."""
    claim_text: str
    category: ClaimCategoryType
    fact_check: FactCheckResult
    failed_check_type: str  # e.g., CROSS_MODAL, EXTERNAL_SOURCE, KNOWLEDGE_GRAPH, EXPERT_VERIFICATION
    # Derived or provided issue type guiding question generation
    issue_type: IssueType = IssueType.AMBIGUITY
    claim_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocraticQuestion:
    """A single Socratic question presented to the user during clarification.

    Attributes:
        id: Short UUID-based identifier for response mapping.
        text: The question text shown to the user.
        qtype: One of ``"open-ended"``, ``"binary"``, or ``"selection"``.
        choices: List of choice strings for selection-type questions.
        expects: Hint describing the desired answer form
            (e.g. ``"rewrite_precise_claim"``).
        metadata: Optional free-form metadata (e.g. justification, evidence used).
    """

    id: str
    text: str
    qtype: str = "open-ended"  # one of: binary, selection, open-ended
    choices: Optional[List[str]] = None
    expects: Optional[str] = None  # hint for desired info (e.g., "disambiguate term X")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClarificationResult:
    """Output from the ClarificationResolutionModule."""
    original_claim: str
    corrected_claim: Optional[str]
    questions: List[SocraticQuestion]
    responses: Dict[str, Any]
    resolution_confidence: float
    next_action: ResolutionAction
    reasoning: str = ""
    issue_type: IssueType = IssueType.AMBIGUITY
    rerun_verification: bool = False
