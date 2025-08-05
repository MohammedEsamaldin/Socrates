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
    """Enum for different types of claims."""
    FACTUAL = "A general statement of fact that can be verified as true or false and does not fall into a more specific category like QUANTITATIVE or TEMPORAL. For example: 'The sky is blue.' or 'Water freezes at 0 degrees Celsius.'",
    QUANTITATIVE = "Involves specific numbers, statistics, measurements, quantities, percentages, currency, or dimensions. The claim's primary assertion relies on this numerical data. For example: 'The population is over 1 million' or 'The project cost $20,000.'",
    TEMPORAL = "Tied to specific times, dates, periods, or the sequence of events. This includes historical events, durations, and chronological assertions. For example: 'The event occurred in 1999' or 'The meeting lasted for two hours.'",
    RELATIONAL = "Describes a relationship between two or more entities or concepts, such as 'is part of,' 'is the capital of,' 'is the founder of,' or 'is a member of'. For example: 'Paris is the capital of France.'",
    CAUSAL = "Asserts a cause-and-effect relationship between events or conditions. Look for keywords like 'causes,' 'leads to,' 'because,' 'due to,' 'as a result of,' 'therefore,' or 'consequently.'",
    COMPARATIVE = "Makes a direct or indirect comparison between two or more entities. This can be explicit ('more than,' 'less than') or implicit ('taller,' 'bigger,' 'better'). For example: 'New York is bigger than London.'",
    DEFINITIONAL = "Provides a definition, classification, or explanation of a term or concept. Look for phrases like 'is defined as,' 'means,' 'refers to,' or 'is a type of.' For example: 'A democracy is a system of government...'",
    SUBJECTIVE_OPINION = "Expresses a personal belief, feeling, preference, or value judgment that cannot be objectively verified. Look for phrases like 'I think,' 'in my opinion,' or words like 'beautiful,' 'best,' or 'important.'",
    CROSS_MODAL = "Relates to or references a piece of information that is not in the text, such as a visual element, an image, or a video that needs to be analyzed to verify the claim. For example: 'The graph shows a steady increase' or 'In the picture, the building is red.'",
    HYPOTHETICAL_PREDICTIVE = "Speculates about future events or hypothetical situations. Claims that cannot be verified at the present time because they have not happened yet. Look for keywords like 'will be,' 'might happen,' 'could,' or 'if... then...'",
    SELF_REFERENTIAL = "Refers to the current conversation, the user, or the agent's own state. These claims are usually not meant for external fact-checking. For example: 'You just asked me about...' or 'I am an AI assistant.'",
    AMBIGUOUS_UNCLEAR = "The claim is too vague, poorly defined, or lacks sufficient context to be categorized or verified by an automated system. This is a crucial fallback category for claims that require human clarification or are unprocessable."

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
