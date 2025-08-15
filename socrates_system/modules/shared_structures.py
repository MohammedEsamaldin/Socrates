"""
Shared data structures for the Socrates Agent modules.

This file centralizes all the dataclasses and Enums used across the claim
processing pipeline to prevent circular import errors.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum, auto

@dataclass
class ExtractedEntity:
    """Represents an entity extracted from text"""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractedRelationship:
    """Represents a relationship between two entities."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0

class VerificationMethod(Enum):
    """Enum for different verification methods."""
    EXTERNAL_SOURCE = auto()
    KNOWLEDGE_GRAPH = auto()
    CROSS_MODAL = auto()
    EXPERT_VERIFICATION = auto()
    CALCULATION = auto()
    DEFINITIONAL = auto()
    UNVERIFIABLE = auto()

@dataclass
class VerificationRoute:
    """Represents the routing decision for a claim."""
    method: VerificationMethod
    confidence: float
    justification: str
    estimated_cost: float
    estimated_latency: float

class ClaimCategoryType(Enum):
    """Enum for MLLM Hallucination Detection claim categories.
    
    Categories are designed for detecting and mitigating factual hallucinations
    in Multimodal Large Language Models through targeted verification approaches.
    """
    VISUAL_GROUNDING_REQUIRED = ("Claims that make assertions about visual elements, objects, scenes, or spatial relationships "
                               "that can be directly verified against the source image through visual analysis.")
    
    EXTERNAL_KNOWLEDGE_REQUIRED = ("Claims that require verification against external world knowledge, factual databases, "
                                 "or common sense that cannot be determined from the image alone.")
    
    SELF_CONSISTENCY_REQUIRED = ("Claims about entities or concepts that should be checked against previously established "
                               "and verified knowledge within the system's knowledge graph.")
    
    AMBIGUOUS_RESOLUTION_REQUIRED = ("Claims that lack sufficient clarity, specificity, or context to be properly "
                                   "categorized or verified without additional clarification.")
    
    SUBJECTIVE_OPINION = ("Claims expressing personal opinions, preferences, aesthetic judgments, or subjective "
                        "interpretations that cannot be factually verified.")
    
    PROCEDURAL_DESCRIPTIVE = ("Claims describing processes, methods, or step-by-step procedures that are "
                            "context-dependent and not easily fact-checkable.")

@dataclass
class ClaimCategory:
    """Represents a category assigned to a claim."""
    name: ClaimCategoryType
    confidence: float
    justification: str

@dataclass
class ExtractedClaim:
    """Represents an extracted claim with all its metadata."""
    text: str
    start_char: int
    end_char: int
    confidence: float
    source_text: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    categories: List[ClaimCategory] = field(default_factory=list)
    verification_route: Optional[VerificationRoute] = None
    context_window: Optional[str] = None
    ambiguity_reason: Optional[str] = None # e.g., "PRONOUN_REFERENCE", "VAGUE_DEMONSTRATIVE"
    # External factuality outcome (populated when routed to EXTERNAL_SOURCE)
    factuality_status: Optional[str] = None  # PASS | FAIL | UNCERTAIN | ERROR
    factuality_confidence: float = 0.0
    factuality_verdict: Optional[bool] = None  # True for PASS, False for FAIL, None for UNCERTAIN/ERROR
    factuality_evidence: List[str] = field(default_factory=list)
    factuality_sources: List[str] = field(default_factory=list)
    factuality_reasoning: Optional[str] = None
