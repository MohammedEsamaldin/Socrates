"""
Clarification resolution package.

Provides the :class:`ClarificationResolutionModule` and its supporting data
models for generating, collecting, and processing Socratic clarification
dialogues when a claim verification fails or is uncertain.
"""
from .data_models import (
    IssueType,
    ResolutionAction,
    FactCheckResult,
    ClarificationContext,
    SocraticQuestion,
    ClarificationResult,
)
from .core import ClarificationResolutionModule

__all__ = [
    "IssueType",
    "ResolutionAction",
    "FactCheckResult",
    "ClarificationContext",
    "SocraticQuestion",
    "ClarificationResult",
    "ClarificationResolutionModule",
]
