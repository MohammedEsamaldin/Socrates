"""
Claim Extraction Module - Advanced NLP-based claim extraction
Uses spaCy and transformer models for sophisticated claim identification
"""
import spacy
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass

from ..utils.logger import setup_logger
from ..config import ENTITY_MODEL_NAME, NLP_MODEL_NAME

logger = setup_logger(__name__)

@dataclass
class ExtractedClaim:
    """Represents an extracted claim with metadata"""
    text: str
    entities: List[Dict[str, Any]]
    claim_type: str
    confidence: float
    sentence_position: int
    dependencies: List[str]

class ClaimExtractor:
    """
    Advanced claim extraction using NLP techniques
    Identifies factual claims, relationships, and attributes from user input
    """
    
    def __init__(self):
        logger.info("Initializing Claim Extractor...")
        
        try:
            # Load spaCy model for entity recognition
            self.nlp = spacy.load(ENTITY_MODEL_NAME)
            
            # Load sentence transformer for semantic analysis
            self.sentence_model = SentenceTransformer(NLP_MODEL_NAME)
            
            # Claim patterns for different types of factual statements
            self.claim_patterns = {
                'attribute': [
                    r'(.+) is (.+)',
                    r'(.+) has (.+)',
                    r'(.+) contains (.+)',
                    r'(.+) measures (.+)',
                    r'(.+) weighs (.+)',
                    r'(.+) costs (.+)'
                ],
                'relationship': [
                    r'(.+) is located in (.+)',
                    r'(.+) belongs to (.+)',
                    r'(.+) is part of (.+)',
                    r'(.+) is connected to (.+)',
                    r'(.+) is related to (.+)',
                    r'(.+) causes (.+)'
                ],
                'temporal': [
                    r'(.+) happened in (.+)',
                    r'(.+) was built in (.+)',
                    r'(.+) occurred on (.+)',
                    r'(.+) started in (.+)',
                    r'(.+) ended in (.+)'
                ],
                'comparative': [
                    r'(.+) is bigger than (.+)',
                    r'(.+) is smaller than (.+)',
                    r'(.+) is better than (.+)',
                    r'(.+) is faster than (.+)',
                    r'(.+) is more (.+) than (.+)'
                ]
            }
            
            logger.info("Claim Extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Claim Extractor: {str(e)}")
            raise
    
    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract factual claims from input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted claims with metadata
        """
        logger.info(f"Extracting claims from text: {text[:100]}...")
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Split into sentences
            sentences = [sent.text.strip() for sent in doc.sents]
            
            claims = []
            
            for i, sentence in enumerate(sentences):
                # Extract claims from each sentence
                sentence_claims = self._extract_claims_from_sentence(sentence, i, doc)
                claims.extend(sentence_claims)
            
            # Filter and rank claims by confidence
            claims = self._filter_and_rank_claims(claims)
            
            logger.info(f"Extracted {len(claims)} claims")
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {str(e)}")
            return []
    
    def _extract_claims_from_sentence(self, sentence: str, position: int, doc) -> List[ExtractedClaim]:
        """Extract claims from a single sentence"""
        claims = []
        
        # Extract entities from sentence
        sentence_doc = self.nlp(sentence)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in sentence_doc.ents
        ]
        
        # Check against claim patterns
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    # Calculate confidence based on entity presence and pattern match
                    confidence = self._calculate_claim_confidence(sentence, entities, match)
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        claim = ExtractedClaim(
                            text=sentence,
                            entities=entities,
                            claim_type=claim_type,
                            confidence=confidence,
                            sentence_position=position,
                            dependencies=self._extract_dependencies(sentence_doc)
                        )
                        claims.append(claim)
                        break  # Only one pattern match per sentence
        
        # If no pattern matches, check if sentence contains factual content
        if not claims and self._is_factual_sentence(sentence, entities):
            claim = ExtractedClaim(
                text=sentence,
                entities=entities,
                claim_type='general',
                confidence=0.5,
                sentence_position=position,
                dependencies=self._extract_dependencies(sentence_doc)
            )
            claims.append(claim)
        
        return claims
    
    def _calculate_claim_confidence(self, sentence: str, entities: List[Dict], match) -> float:
        """Calculate confidence score for a claim"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on named entities
        if entities:
            confidence += 0.2 * min(len(entities), 3)  # Max boost of 0.6
        
        # Boost confidence for specific entity types
        entity_types = [ent['label'] for ent in entities]
        important_types = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'QUANTITY']
        for ent_type in important_types:
            if ent_type in entity_types:
                confidence += 0.1
        
        # Reduce confidence for questions or uncertain language
        uncertain_words = ['maybe', 'perhaps', 'might', 'could', 'possibly']
        if any(word in sentence.lower() for word in uncertain_words):
            confidence -= 0.2
        
        if sentence.strip().endswith('?'):
            confidence -= 0.3
        
        return min(confidence, 1.0)
    
    def _is_factual_sentence(self, sentence: str, entities: List[Dict]) -> bool:
        """Determine if a sentence contains factual content"""
        # Check for factual indicators
        factual_indicators = [
            'is', 'are', 'was', 'were', 'has', 'have', 'contains',
            'located', 'built', 'created', 'founded', 'established'
        ]
        
        # Must have entities and factual indicators
        return (len(entities) > 0 and 
                any(indicator in sentence.lower() for indicator in factual_indicators) and
                not sentence.strip().endswith('?'))
    
    def _extract_dependencies(self, doc) -> List[str]:
        """Extract syntactic dependencies from sentence"""
        dependencies = []
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr']:
                dependencies.append(f"{token.text}:{token.dep_}")
        return dependencies
    
    def _filter_and_rank_claims(self, claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
        """Filter and rank claims by confidence and importance"""
        # Remove duplicates
        unique_claims = []
        seen_texts = set()
        
        for claim in claims:
            if claim.text not in seen_texts:
                unique_claims.append(claim)
                seen_texts.add(claim.text)
        
        # Sort by confidence and entity count
        unique_claims.sort(key=lambda x: (x.confidence, len(x.entities)), reverse=True)
        
        return unique_claims
    
    def get_claim_summary(self, claims: List[ExtractedClaim]) -> Dict[str, Any]:
        """Get summary statistics of extracted claims"""
        if not claims:
            return {"total_claims": 0}
        
        claim_types = {}
        total_entities = 0
        avg_confidence = 0
        
        for claim in claims:
            claim_types[claim.claim_type] = claim_types.get(claim.claim_type, 0) + 1
            total_entities += len(claim.entities)
            avg_confidence += claim.confidence
        
        return {
            "total_claims": len(claims),
            "claim_types": claim_types,
            "total_entities": total_entities,
            "average_confidence": avg_confidence / len(claims),
            "highest_confidence": max(claim.confidence for claim in claims),
            "lowest_confidence": min(claim.confidence for claim in claims)
        }
