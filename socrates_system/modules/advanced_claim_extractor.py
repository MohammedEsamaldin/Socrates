"""
Advanced Claim Extractor - Zero-shot Faithful Factual Error Correction inspired
Implements sophisticated claim extraction with relationships and attributes using local LLM
"""
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_manager import get_llm_manager, LLMTaskType
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ClaimType(Enum):
    """Types of claims for categorization"""
    FACTUAL = "factual"           # Verifiable factual statements
    RELATIONAL = "relational"     # Relationships between entities
    TEMPORAL = "temporal"         # Time-based claims
    QUANTITATIVE = "quantitative" # Numerical/statistical claims
    CAUSAL = "causal"            # Cause-effect relationships
    COMPARATIVE = "comparative"   # Comparative statements
    DEFINITIONAL = "definitional" # Definitions and classifications

class VerifiabilityLevel(Enum):
    """Levels of claim verifiability"""
    HIGH = "high"         # Easily verifiable with external sources
    MEDIUM = "medium"     # Requires some interpretation or context
    LOW = "low"          # Difficult to verify objectively
    SUBJECTIVE = "subjective"  # Opinion-based, not factual

@dataclass
class Entity:
    """Represents an entity extracted from text"""
    text: str
    entity_type: str
    confidence: float
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source: str
    relation: str
    target: str
    confidence: float
    context: str
    temporal_info: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdvancedClaim:
    """Enhanced claim structure with comprehensive metadata"""
    text: str
    claim_type: ClaimType
    verifiability: VerifiabilityLevel
    confidence: float
    entities: List[Entity]
    relationships: List[Relationship]
    context: str
    evidence_requirements: List[str]
    verification_questions: List[str]
    temporal_context: Optional[str] = None
    spatial_context: Optional[str] = None
    source_span: Tuple[int, int] = (0, 0)
    dependencies: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    extraction_timestamp: datetime = field(default_factory=datetime.now)

class AdvancedClaimExtractor:
    """
    Advanced Claim Extractor implementing Zero-shot Faithful Factual Error Correction methodology
    Uses local LLM for sophisticated claim extraction with relationships and attributes
    """
    
    def __init__(self):
        """Initialize the advanced claim extractor"""
        self.llm_manager = get_llm_manager()
        self.extraction_cache = {}
        
        # Initialize extraction patterns and rules
        self._init_extraction_patterns()
        
        logger.info("AdvancedClaimExtractor initialized with LLM integration")
    
    def _init_extraction_patterns(self):
        """Initialize patterns for claim identification and classification"""
        self.claim_patterns = {
            ClaimType.FACTUAL: [
                r'\b(is|are|was|were|has|have|had)\b.*\b(fact|true|false|correct|incorrect)\b',
                r'\b(according to|research shows|studies indicate|data reveals)\b',
                r'\b(percent|percentage|number|amount|quantity)\b.*\b(of|in|from)\b'
            ],
            ClaimType.TEMPORAL: [
                r'\b(in|on|at|during|before|after|since|until)\b.*\b(\d{4}|\d{1,2}/\d{1,2})\b',
                r'\b(yesterday|today|tomorrow|recently|currently|previously)\b',
                r'\b(first|last|next|previous)\b.*\b(time|year|month|day)\b'
            ],
            ClaimType.CAUSAL: [
                r'\b(because|due to|caused by|results in|leads to|triggers)\b',
                r'\b(if.*then|when.*then|as a result|consequently)\b',
                r'\b(correlation|causation|effect|impact|influence)\b'
            ],
            ClaimType.COMPARATIVE: [
                r'\b(more|less|better|worse|higher|lower|greater|smaller)\b.*\bthan\b',
                r'\b(compared to|in comparison|versus|vs\.?)\b',
                r'\b(most|least|best|worst|highest|lowest)\b'
            ]
        }
        
        self.entity_patterns = {
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'ORGANIZATION': r'\b[A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|LLC|University|Institute)\b',
            'LOCATION': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|Province))\b',
            'DATE': r'\b\d{1,2}/\d{1,2}/\d{4}|\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'NUMBER': r'\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand))?\b'
        }
    
    async def extract_claims(self, text: str, context: Dict[str, Any] = None) -> List[AdvancedClaim]:
        """
        Extract advanced claims from text using Zero-shot methodology
        
        Args:
            text: Input text to extract claims from
            context: Additional context for extraction
            
        Returns:
            List of AdvancedClaim objects with comprehensive metadata
        """
        logger.info(f"Extracting claims from text: {text[:100]}...")
        
        try:
            # Stage 1: Initial claim identification using LLM
            initial_claims = await self._identify_initial_claims(text, context)
            
            # Stage 2: Entity and relationship extraction
            enhanced_claims = await self._extract_entities_and_relationships(initial_claims, text)
            
            # Stage 3: Claim classification and verification planning
            classified_claims = await self._classify_and_plan_verification(enhanced_claims, text)
            
            # Stage 4: Generate verification questions (Zero-shot methodology)
            final_claims = await self._generate_verification_questions(classified_claims)
            
            # Stage 5: Post-processing and validation
            validated_claims = self._validate_and_deduplicate(final_claims)
            
            logger.info(f"Extracted {len(validated_claims)} advanced claims")
            return validated_claims
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []
    
    async def _identify_initial_claims(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Stage 1: Identify initial claims using LLM"""
        logger.debug("Stage 1: Initial claim identification")
        
        # Use LLM for sophisticated claim extraction
        response = await self.llm_manager.extract_claims(text, context)
        
        if response.error:
            logger.error(f"LLM claim extraction failed: {response.error}")
            return []
        
        # Parse structured output
        if response.structured_output and 'claims' in response.structured_output:
            return response.structured_output['claims']
        
        # Fallback: parse from content
        return self._parse_claims_from_content(response.content, text)
    
    def _parse_claims_from_content(self, content: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse claims from LLM content when structured output fails"""
        claims = []
        
        # Try to extract JSON-like structures
        json_pattern = r'\{[^{}]*"text"[^{}]*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                claim_data = json.loads(match)
                claims.append(claim_data)
            except json.JSONDecodeError:
                continue
        
        # Fallback: extract sentences as potential claims
        if not claims:
            sentences = re.split(r'[.!?]+', original_text)
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Filter out very short sentences
                    claims.append({
                        'text': sentence,
                        'confidence': 0.6,
                        'entities': [],
                        'relationships': [],
                        'context': f"Sentence {i+1}"
                    })
        
        return claims
    
    async def _extract_entities_and_relationships(self, initial_claims: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Stage 2: Extract entities and relationships for each claim"""
        logger.debug("Stage 2: Entity and relationship extraction")
        
        enhanced_claims = []
        
        for claim_data in initial_claims:
            claim_text = claim_data.get('text', '')
            
            # Extract entities using pattern matching and LLM
            entities = await self._extract_entities(claim_text)
            
            # Extract relationships using LLM
            if entities:
                entity_names = [e['text'] for e in entities]
                relationships_response = await self.llm_manager.extract_relationships(
                    claim_text, entity_names
                )
                
                relationships = []
                if relationships_response.structured_output and 'relationships' in relationships_response.structured_output:
                    relationships = relationships_response.structured_output['relationships']
            else:
                relationships = []
            
            # Enhance claim data
            enhanced_claim = {
                **claim_data,
                'entities': entities,
                'relationships': relationships,
                'enhanced': True
            }
            
            enhanced_claims.append(enhanced_claim)
        
        return enhanced_claims
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using patterns and LLM"""
        entities = []
        
        # Pattern-based entity extraction
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    'text': match.group(),
                    'entity_type': entity_type,
                    'confidence': 0.8,
                    'span': (match.start(), match.end()),
                    'attributes': {}
                }
                entities.append(entity)
        
        # Remove duplicates and overlaps
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and overlapping entities"""
        # Sort by span start position
        entities.sort(key=lambda x: x.get('span', (0, 0))[0])
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            entity_span = entity.get('span', (0, 0))
            
            for existing in deduplicated:
                existing_span = existing.get('span', (0, 0))
                if (entity_span[0] < existing_span[1] and entity_span[1] > existing_span[0]):
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
    
    async def _classify_and_plan_verification(self, enhanced_claims: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Stage 3: Classify claims and plan verification approach"""
        logger.debug("Stage 3: Claim classification and verification planning")
        
        classified_claims = []
        
        for claim_data in enhanced_claims:
            claim_text = claim_data.get('text', '')
            
            # Classify claim type
            claim_type = self._classify_claim_type(claim_text)
            
            # Determine verifiability level
            verifiability = self._assess_verifiability(claim_text, claim_data.get('entities', []))
            
            # Plan evidence requirements
            evidence_requirements = self._plan_evidence_requirements(claim_text, claim_type, verifiability)
            
            # Add classification data
            classified_claim = {
                **claim_data,
                'claim_type': claim_type.value,
                'verifiability': verifiability.value,
                'evidence_requirements': evidence_requirements,
                'classified': True
            }
            
            classified_claims.append(classified_claim)
        
        return classified_claims
    
    def _classify_claim_type(self, claim_text: str) -> ClaimType:
        """Classify the type of claim based on patterns"""
        claim_lower = claim_text.lower()
        
        # Check patterns for each claim type
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                if re.search(pattern, claim_lower):
                    return claim_type
        
        # Default to factual if no specific pattern matches
        return ClaimType.FACTUAL
    
    def _assess_verifiability(self, claim_text: str, entities: List[Dict[str, Any]]) -> VerifiabilityLevel:
        """Assess how easily the claim can be verified"""
        claim_lower = claim_text.lower()
        
        # Subjective indicators
        subjective_indicators = ['believe', 'think', 'feel', 'opinion', 'prefer', 'like', 'dislike']
        if any(indicator in claim_lower for indicator in subjective_indicators):
            return VerifiabilityLevel.SUBJECTIVE
        
        # High verifiability indicators
        high_verifiability = ['research', 'study', 'data', 'statistics', 'published', 'official']
        if any(indicator in claim_lower for indicator in high_verifiability):
            return VerifiabilityLevel.HIGH
        
        # Check entity types for verifiability
        entity_types = [e.get('entity_type', '') for e in entities]
        if 'DATE' in entity_types or 'NUMBER' in entity_types:
            return VerifiabilityLevel.HIGH
        
        if 'PERSON' in entity_types or 'ORGANIZATION' in entity_types:
            return VerifiabilityLevel.MEDIUM
        
        return VerifiabilityLevel.MEDIUM
    
    def _plan_evidence_requirements(self, claim_text: str, claim_type: ClaimType, verifiability: VerifiabilityLevel) -> List[str]:
        """Plan what evidence would be needed to verify the claim"""
        requirements = []
        
        if claim_type == ClaimType.FACTUAL:
            requirements.extend(['authoritative sources', 'official documentation'])
        elif claim_type == ClaimType.TEMPORAL:
            requirements.extend(['historical records', 'timeline verification'])
        elif claim_type == ClaimType.QUANTITATIVE:
            requirements.extend(['statistical data', 'numerical verification'])
        elif claim_type == ClaimType.CAUSAL:
            requirements.extend(['causal analysis', 'correlation studies'])
        
        if verifiability == VerifiabilityLevel.HIGH:
            requirements.append('direct factual verification')
        elif verifiability == VerifiabilityLevel.MEDIUM:
            requirements.append('contextual interpretation')
        
        return requirements
    
    async def _generate_verification_questions(self, classified_claims: List[Dict[str, Any]]) -> List[AdvancedClaim]:
        """Stage 4: Generate verification questions using Zero-shot methodology"""
        logger.debug("Stage 4: Verification question generation")
        
        final_claims = []
        
        for claim_data in classified_claims:
            claim_text = claim_data.get('text', '')
            
            # Generate Socratic questions for verification
            questions_response = await self.llm_manager.generate_socratic_questions(
                claim_text, 
                context={
                    'claim_type': claim_data.get('claim_type'),
                    'verifiability': claim_data.get('verifiability'),
                    'entities': claim_data.get('entities', []),
                    'evidence_requirements': claim_data.get('evidence_requirements', [])
                }
            )
            
            verification_questions = []
            if questions_response.structured_output and 'questions' in questions_response.structured_output:
                verification_questions = [q.get('question', '') for q in questions_response.structured_output['questions']]
            
            # Create AdvancedClaim object
            advanced_claim = AdvancedClaim(
                text=claim_text,
                claim_type=ClaimType(claim_data.get('claim_type', 'factual')),
                verifiability=VerifiabilityLevel(claim_data.get('verifiability', 'medium')),
                confidence=claim_data.get('confidence', 0.7),
                entities=[Entity(**e) if isinstance(e, dict) else e for e in claim_data.get('entities', [])],
                relationships=[Relationship(**r) if isinstance(r, dict) else r for r in claim_data.get('relationships', [])],
                context=claim_data.get('context', ''),
                evidence_requirements=claim_data.get('evidence_requirements', []),
                verification_questions=verification_questions,
                attributes=claim_data.get('attributes', {})
            )
            
            final_claims.append(advanced_claim)
        
        return final_claims
    
    def _validate_and_deduplicate(self, claims: List[AdvancedClaim]) -> List[AdvancedClaim]:
        """Stage 5: Validate and deduplicate claims"""
        logger.debug("Stage 5: Validation and deduplication")
        
        # Remove duplicates based on text similarity
        unique_claims = []
        seen_texts = set()
        
        for claim in claims:
            # Normalize text for comparison
            normalized_text = re.sub(r'\s+', ' ', claim.text.lower().strip())
            
            if normalized_text not in seen_texts:
                unique_claims.append(claim)
                seen_texts.add(normalized_text)
        
        # Sort by confidence and verifiability
        unique_claims.sort(key=lambda x: (x.confidence, x.verifiability.value), reverse=True)
        
        return unique_claims
    
    def get_extraction_summary(self, claims: List[AdvancedClaim]) -> Dict[str, Any]:
        """Get comprehensive summary of extracted claims"""
        if not claims:
            return {"total_claims": 0}
        
        claim_types = {}
        verifiability_levels = {}
        total_entities = 0
        total_relationships = 0
        avg_confidence = 0
        
        for claim in claims:
            # Count claim types
            claim_types[claim.claim_type.value] = claim_types.get(claim.claim_type.value, 0) + 1
            
            # Count verifiability levels
            verifiability_levels[claim.verifiability.value] = verifiability_levels.get(claim.verifiability.value, 0) + 1
            
            # Count entities and relationships
            total_entities += len(claim.entities)
            total_relationships += len(claim.relationships)
            avg_confidence += claim.confidence
        
        return {
            "total_claims": len(claims),
            "claim_types": claim_types,
            "verifiability_levels": verifiability_levels,
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "average_confidence": avg_confidence / len(claims),
            "highest_confidence": max(claim.confidence for claim in claims),
            "lowest_confidence": min(claim.confidence for claim in claims),
            "extraction_methodology": "Zero-shot Faithful Factual Error Correction inspired"
        }
    
    async def extract_claims_with_evidence_planning(self, text: str, evidence_sources: List[str] = None) -> Tuple[List[AdvancedClaim], Dict[str, Any]]:
        """
        Extract claims and create evidence verification plan
        Following Zero-shot methodology: formulate questions, plan evidence lookup
        """
        claims = await self.extract_claims(text)
        
        # Create evidence verification plan
        verification_plan = {
            "claims_to_verify": len([c for c in claims if c.verifiability in [VerifiabilityLevel.HIGH, VerifiabilityLevel.MEDIUM]]),
            "evidence_sources_needed": [],
            "verification_questions": [],
            "verification_strategy": {}
        }
        
        for claim in claims:
            if claim.verifiability != VerifiabilityLevel.SUBJECTIVE:
                verification_plan["verification_questions"].extend(claim.verification_questions)
                verification_plan["evidence_sources_needed"].extend(claim.evidence_requirements)
                
                verification_plan["verification_strategy"][claim.text] = {
                    "questions": claim.verification_questions,
                    "evidence_needed": claim.evidence_requirements,
                    "verification_approach": self._get_verification_approach(claim)
                }
        
        # Remove duplicates
        verification_plan["evidence_sources_needed"] = list(set(verification_plan["evidence_sources_needed"]))
        verification_plan["verification_questions"] = list(set(verification_plan["verification_questions"]))
        
        return claims, verification_plan
    
    def _get_verification_approach(self, claim: AdvancedClaim) -> str:
        """Determine the best verification approach for a claim"""
        if claim.claim_type == ClaimType.QUANTITATIVE:
            return "statistical_verification"
        elif claim.claim_type == ClaimType.TEMPORAL:
            return "historical_verification"
        elif claim.claim_type == ClaimType.CAUSAL:
            return "causal_analysis"
        elif claim.verifiability == VerifiabilityLevel.HIGH:
            return "direct_factual_check"
        else:
            return "contextual_verification"
