"""
Ambiguity Checker - Detects and handles ambiguous claims
Identifies unclear or ambiguous statements that need clarification
"""
import logging
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class AmbiguityResult:
    """Result of ambiguity check"""
    needs_clarification: bool
    ambiguous_terms: List[str]
    clarification_questions: List[str]
    ambiguity_score: float
    reasoning: str

class AmbiguityChecker:
    """
    Ambiguity checker for identifying unclear claims
    Detects various types of ambiguity that require clarification
    """
    
    def __init__(self):
        logger.info("Initializing Ambiguity Checker...")
        
        # Ambiguous terms and patterns
        self.ambiguous_indicators = {
            'vague_quantifiers': ['some', 'many', 'few', 'several', 'most', 'often', 'sometimes'],
            'subjective_terms': ['beautiful', 'good', 'bad', 'better', 'worse', 'nice', 'ugly'],
            'relative_terms': ['big', 'small', 'large', 'tiny', 'fast', 'slow', 'high', 'low'],
            'temporal_vague': ['recently', 'soon', 'later', 'earlier', 'long ago', 'nowadays'],
            'pronouns': ['it', 'this', 'that', 'they', 'them', 'he', 'she'],
            'modal_uncertainty': ['might', 'could', 'may', 'possibly', 'probably', 'perhaps']
        }
        
        # Context-dependent terms
        self.context_dependent = ['here', 'there', 'now', 'then', 'today', 'yesterday']
        
        logger.info("Ambiguity Checker initialized")
    
    def check_ambiguity(self, claim: str, context: str = "") -> Dict[str, Any]:
        """
        Check if claim contains ambiguous elements
        
        Args:
            claim: The claim to check for ambiguity
            context: Additional context (original input)
            
        Returns:
            Dictionary containing ambiguity analysis results
        """
        logger.info(f"Checking ambiguity for: {claim[:50]}...")
        
        try:
            # Analyze different types of ambiguity
            ambiguous_terms = self._identify_ambiguous_terms(claim)
            context_issues = self._check_context_dependency(claim, context)
            structural_ambiguity = self._check_structural_ambiguity(claim)
            
            # Calculate ambiguity score
            ambiguity_score = self._calculate_ambiguity_score(
                ambiguous_terms, context_issues, structural_ambiguity
            )
            
            # Determine if clarification is needed
            needs_clarification = ambiguity_score > 0.3
            
            # Generate clarification questions
            clarification_questions = self._generate_clarification_questions(
                claim, ambiguous_terms, context_issues, structural_ambiguity
            )
            
            # Generate reasoning
            reasoning = self._generate_ambiguity_reasoning(
                ambiguous_terms, context_issues, structural_ambiguity, ambiguity_score
            )
            
            result = {
                "needs_clarification": needs_clarification,
                "ambiguous_terms": ambiguous_terms,
                "clarification_questions": clarification_questions,
                "ambiguity_score": ambiguity_score,
                "reasoning": reasoning,
                "context_issues": context_issues,
                "structural_issues": structural_ambiguity
            }
            
            logger.info(f"Ambiguity check completed: {'needs clarification' if needs_clarification else 'clear'}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ambiguity check: {str(e)}")
            return {
                "needs_clarification": False,
                "ambiguous_terms": [],
                "clarification_questions": [],
                "ambiguity_score": 0.0,
                "reasoning": "Error in ambiguity analysis"
            }
    
    def _identify_ambiguous_terms(self, claim: str) -> List[str]:
        """Identify ambiguous terms in the claim"""
        ambiguous_terms = []
        claim_lower = claim.lower()
        
        for category, terms in self.ambiguous_indicators.items():
            for term in terms:
                if f" {term} " in f" {claim_lower} " or claim_lower.startswith(f"{term} ") or claim_lower.endswith(f" {term}"):
                    ambiguous_terms.append(term)
        
        # Check for context-dependent terms
        for term in self.context_dependent:
            if f" {term} " in f" {claim_lower} ":
                ambiguous_terms.append(term)
        
        return list(set(ambiguous_terms))  # Remove duplicates
    
    def _check_context_dependency(self, claim: str, context: str) -> List[str]:
        """Check for context-dependent issues"""
        issues = []
        
        # Check for pronouns without clear antecedents
        pronouns = re.findall(r'\b(it|this|that|they|them|he|she)\b', claim.lower())
        if pronouns and not context:
            issues.append("Contains pronouns without clear context")
        
        # Check for demonstratives
        demonstratives = re.findall(r'\b(this|that|these|those)\b', claim.lower())
        if demonstratives:
            issues.append("Contains demonstrative pronouns that may need clarification")
        
        # Check for location/time references
        location_time = re.findall(r'\b(here|there|now|then|today|yesterday)\b', claim.lower())
        if location_time:
            issues.append("Contains location or time references that depend on context")
        
        return issues
    
    def _check_structural_ambiguity(self, claim: str) -> List[str]:
        """Check for structural ambiguity in the sentence"""
        issues = []
        
        # Check for multiple possible interpretations
        if " or " in claim.lower():
            issues.append("Contains 'or' which may create ambiguity")
        
        # Check for complex nested structures
        if claim.count('(') != claim.count(')'):
            issues.append("Unbalanced parentheses may cause confusion")
        
        # Check for multiple subjects or predicates
        if claim.count(',') > 2:
            issues.append("Complex sentence structure may be ambiguous")
        
        # Check for negation ambiguity
        negation_words = ['not', 'no', 'never', 'none']
        negation_count = sum(1 for word in negation_words if word in claim.lower().split())
        if negation_count > 1:
            issues.append("Multiple negations may create ambiguity")
        
        return issues
    
    def _calculate_ambiguity_score(self, ambiguous_terms: List[str], 
                                 context_issues: List[str], 
                                 structural_issues: List[str]) -> float:
        """Calculate overall ambiguity score"""
        # it claculate the ambiuity scores based on the number of terms and issues found
        score = 0.0
        
        # Score for ambiguous terms
        score += len(ambiguous_terms) * 0.1
        
        # Score for context issues
        score += len(context_issues) * 0.2
        
        # Score for structural issues
        score += len(structural_issues) * 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_clarification_questions(self, claim: str, ambiguous_terms: List[str],
                                        context_issues: List[str], 
                                        structural_issues: List[str]) -> List[str]:
        """Generate specific clarification questions"""
        questions = []
        
        # Questions for ambiguous terms
        for term in ambiguous_terms[:3]:  # Limit to top 3
            if term in ['some', 'many', 'few', 'several']:
                questions.append(f"Could you specify approximately how many when you say '{term}'?")
            elif term in ['big', 'small', 'large', 'tiny']:
                questions.append(f"Could you provide more specific measurements or comparisons for '{term}'?")
            elif term in ['recently', 'soon', 'later']:
                questions.append(f"Could you specify a timeframe for '{term}'?")
            elif term in ['good', 'bad', 'better', 'worse']:
                questions.append(f"What criteria are you using to evaluate '{term}'?")
            elif term in ['it', 'this', 'that', 'they']:
                questions.append(f"What specifically does '{term}' refer to in your statement?")
        
        # Questions for context issues
        if "Contains pronouns without clear context" in context_issues:
            questions.append("Could you clarify what the pronouns in your statement refer to?")
        
        if "Contains location or time references" in context_issues:
            questions.append("Could you specify the exact location or time you're referring to?")
        
        # Questions for structural issues
        if "Contains 'or' which may create ambiguity" in structural_issues:
            questions.append("Are you referring to one specific option, or could multiple options be true?")
        
        # General clarification if no specific questions
        if not questions:
            questions.append("Could you provide more specific details about your claim?")
        
        return questions[:3]  # Limit to 3 questions
    
    def _generate_ambiguity_reasoning(self, ambiguous_terms: List[str],
                                    context_issues: List[str],
                                    structural_issues: List[str],
                                    ambiguity_score: float) -> str:
        """Generate reasoning for ambiguity assessment"""
        reasoning_parts = []
        
        if ambiguous_terms:
            reasoning_parts.append(f"Found {len(ambiguous_terms)} ambiguous terms: {', '.join(ambiguous_terms[:3])}")
        
        if context_issues:
            reasoning_parts.append(f"Identified {len(context_issues)} context-dependent issues")
        
        if structural_issues:
            reasoning_parts.append(f"Detected {len(structural_issues)} structural ambiguities")
        
        if ambiguity_score > 0.5:
            reasoning_parts.append("High ambiguity score suggests significant clarification needed")
        elif ambiguity_score > 0.3:
            reasoning_parts.append("Moderate ambiguity detected, some clarification would be helpful")
        else:
            reasoning_parts.append("Low ambiguity detected, claim is relatively clear")
        
        return ". ".join(reasoning_parts) if reasoning_parts else "No significant ambiguity detected"
