"""
Question Generation Module - Sophisticated Socratic Question Generation
Implements advanced question generation strategies for different verification contexts
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class QuestionType(Enum):
    VERIFICATION = "verification"
    CLARIFICATION = "clarification"
    DEEPER_ANALYSIS = "deeper_analysis"
    CONSISTENCY = "consistency"
    AMBIGUITY = "ambiguity"
    CROSS_ALIGNMENT = "cross_alignment"

@dataclass
class SocraticInquiry:
    """Represents a sophisticated Socratic inquiry"""
    question: str
    reasoning: str
    expected_answer_type: str
    confidence: float
    context: Dict[str, Any]
    follow_up_questions: List[str]

class QuestionGenerator:
    """
    Advanced Socratic Question Generator
    Creates contextually appropriate questions for different verification scenarios
    """
    
    def __init__(self):
        logger.info("Initializing Question Generator...")
        
        # Question templates for different contexts
        self.question_templates = {
            QuestionType.VERIFICATION: {
                'attribute': [
                    "What evidence supports the claim that {subject} {predicate}?",
                    "How can we independently verify that {subject} {predicate}?",
                    "What authoritative sources confirm that {subject} {predicate}?",
                    "Under what conditions would {subject} {predicate} be true?",
                    "What would contradict the assertion that {subject} {predicate}?"
                ],
                'relationship': [
                    "What establishes the relationship between {entity1} and {entity2}?",
                    "How is the connection between {entity1} and {entity2} documented?",
                    "What evidence links {entity1} to {entity2}?",
                    "Through what mechanism are {entity1} and {entity2} related?",
                    "What would sever the relationship between {entity1} and {entity2}?"
                ],
                'temporal': [
                    "What historical records support this temporal claim?",
                    "How is this date or time period verified?",
                    "What contemporary sources document this timing?",
                    "What sequence of events led to this temporal assertion?",
                    "How does this timing align with other known events?"
                ],
                'general': [
                    "What foundational evidence supports this claim?",
                    "How might we test the validity of this assertion?",
                    "What assumptions underlie this statement?",
                    "What would need to be true for this claim to hold?",
                    "How does this claim relate to established knowledge?"
                ]
            },
            QuestionType.CLARIFICATION: [
                "Could you elaborate on what you mean by '{ambiguous_term}'?",
                "When you say '{claim}', are you referring to {interpretation1} or {interpretation2}?",
                "What specific context are you considering for this claim?",
                "Could you provide more details about '{unclear_aspect}'?",
                "What evidence would you consider most relevant to this claim?"
            ],
            QuestionType.DEEPER_ANALYSIS: [
                "What underlying assumptions does this claim rest upon?",
                "How might cultural or contextual factors influence this assertion?",
                "What are the implications if this claim were false?",
                "What related claims would also need to be true?",
                "How does this claim fit within the broader domain of knowledge?"
            ],
            QuestionType.CONSISTENCY: [
                "How does this claim align with your previous statements about {topic}?",
                "What reconciles this assertion with the earlier claim that {previous_claim}?",
                "How do we resolve the apparent contradiction between this and {conflicting_claim}?",
                "What additional context might explain these seemingly inconsistent claims?",
                "Which of these conflicting assertions should take precedence and why?"
            ],
            QuestionType.AMBIGUITY: [
                "What specific interpretation of '{ambiguous_term}' are you using?",
                "Could this statement be understood in multiple ways?",
                "What additional information would make this claim more precise?",
                "Are there implicit assumptions in this statement that should be made explicit?",
                "What scope or limitations apply to this claim?"
            ],
            QuestionType.CROSS_ALIGNMENT: [
                "How does what you've described align with what we observe in the image?",
                "What visual elements support or contradict your textual claim?",
                "How might the visual context change the interpretation of your statement?",
                "What details in the image are most relevant to verifying this claim?",
                "How do we reconcile differences between the visual and textual information?"
            ]
        }
        
        # Reasoning templates for different question types
        self.reasoning_templates = {
            QuestionType.VERIFICATION: "This question seeks to establish independent evidence for the claim, following the Socratic principle of examining the foundations of knowledge.",
            QuestionType.CLARIFICATION: "This inquiry aims to resolve ambiguity and ensure precise understanding, as clarity is essential for meaningful verification.",
            QuestionType.DEEPER_ANALYSIS: "This question probes the underlying structure and implications of the claim, revealing hidden assumptions and connections.",
            QuestionType.CONSISTENCY: "This inquiry examines internal consistency within the knowledge framework, identifying potential contradictions that require resolution.",
            QuestionType.AMBIGUITY: "This question addresses linguistic or conceptual ambiguity that could lead to misinterpretation or false verification.",
            QuestionType.CROSS_ALIGNMENT: "This inquiry examines the consistency between different modalities of information, ensuring multimodal coherence."
        }
        
        logger.info("Question Generator initialized successfully")
    
    def generate_socratic_inquiry(self, claim: str, question_type: str, 
                                context: Optional[Dict[str, Any]] = None) -> SocraticInquiry:
        """
        Generate a sophisticated Socratic inquiry for a given claim and context
        
        Args:
            claim: The claim to generate questions for
            question_type: Type of question needed
            context: Additional context for question generation
            
        Returns:
            SocraticInquiry object with question and metadata
        """
        logger.info(f"Generating {question_type} question for claim: {claim[:50]}...")
        
        try:
            q_type = QuestionType(question_type)
            context = context or {}
            
            # Parse claim for key components
            claim_components = self._parse_claim_components(claim)
            
            # Generate primary question
            primary_question = self._generate_primary_question(claim, q_type, claim_components, context)
            
            # Generate follow-up questions
            follow_ups = self._generate_follow_up_questions(claim, q_type, claim_components, context)
            
            # Determine expected answer type
            expected_answer_type = self._determine_expected_answer_type(claim, q_type)
            
            # Calculate confidence
            confidence = self._calculate_question_confidence(claim, q_type, claim_components)
            
            # Get reasoning
            reasoning = self._generate_reasoning(q_type, claim, context)
            
            inquiry = SocraticInquiry(
                question=primary_question,
                reasoning=reasoning,
                expected_answer_type=expected_answer_type,
                confidence=confidence,
                context=context,
                follow_up_questions=follow_ups
            )
            
            logger.info(f"Generated Socratic inquiry: {primary_question[:50]}...")
            return inquiry
            
        except Exception as e:
            logger.error(f"Error generating Socratic inquiry: {str(e)}")
            # Return fallback question
            return SocraticInquiry(
                question=f"What evidence supports the claim: '{claim}'?",
                reasoning="Fallback question for basic verification",
                expected_answer_type="evidence",
                confidence=0.5,
                context=context or {},
                follow_up_questions=[]
            )
    
    def _parse_claim_components(self, claim: str) -> Dict[str, Any]:
        """Parse claim into components for question generation"""
        components = {
            'full_claim': claim,
            'subject': None,
            'predicate': None,
            'entities': [],
            'claim_type': 'general'
        }
        
        # Simple parsing for common patterns
        claim_lower = claim.lower().strip()
        
        # Attribute pattern: X is Y
        if ' is ' in claim_lower:
            parts = claim.split(' is ', 1)
            if len(parts) == 2:
                components['subject'] = parts[0].strip()
                components['predicate'] = parts[1].strip()
                components['claim_type'] = 'attribute'
        
        # Relationship pattern: X is in Y, X belongs to Y, etc.
        relationship_indicators = [' is in ', ' belongs to ', ' is part of ', ' is located in ']
        for indicator in relationship_indicators:
            if indicator in claim_lower:
                parts = claim.split(indicator, 1)
                if len(parts) == 2:
                    components['entity1'] = parts[0].strip()
                    components['entity2'] = parts[1].strip()
                    components['claim_type'] = 'relationship'
                break
        
        # Extract potential entities (simple heuristic)
        words = claim.split()
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 1]
        components['entities'] = capitalized_words
        
        return components
    
    def _generate_primary_question(self, claim: str, q_type: QuestionType, 
                                 components: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate the primary Socratic question"""
        
        if q_type == QuestionType.VERIFICATION:
            claim_type = components.get('claim_type', 'general')
            templates = self.question_templates[q_type].get(claim_type, 
                                                          self.question_templates[q_type]['general'])
            template = random.choice(templates)
            
            # Fill in template with claim components
            if claim_type == 'attribute' and components.get('subject') and components.get('predicate'):
                return template.format(
                    subject=components['subject'],
                    predicate=components['predicate']
                )
            elif claim_type == 'relationship' and components.get('entity1') and components.get('entity2'):
                return template.format(
                    entity1=components['entity1'],
                    entity2=components['entity2']
                )
            else:
                return template.replace('{subject}', claim).replace('{predicate}', '').strip()
        
        else:
            templates = self.question_templates[q_type]
            template = random.choice(templates)
            
            # Fill in context-specific information
            if '{claim}' in template:
                template = template.replace('{claim}', claim)
            if '{ambiguous_term}' in template and context.get('ambiguous_terms'):
                template = template.replace('{ambiguous_term}', context['ambiguous_terms'][0])
            if '{topic}' in template and context.get('topic'):
                template = template.replace('{topic}', context['topic'])
            if '{previous_claim}' in template and context.get('previous_claim'):
                template = template.replace('{previous_claim}', context['previous_claim'])
            if '{conflicting_claim}' in template and context.get('conflicting_claim'):
                template = template.replace('{conflicting_claim}', context['conflicting_claim'])
            
            return template
    
    def _generate_follow_up_questions(self, claim: str, q_type: QuestionType, 
                                    components: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions for deeper inquiry"""
        follow_ups = []
        
        if q_type == QuestionType.VERIFICATION:
            follow_ups = [
                f"What would constitute sufficient evidence for '{claim}'?",
                f"What alternative explanations might account for the observations related to '{claim}'?",
                f"How might we distinguish between correlation and causation in '{claim}'?"
            ]
        
        elif q_type == QuestionType.CLARIFICATION:
            follow_ups = [
                "What specific examples would illustrate your point?",
                "How would you define the key terms in your statement?",
                "What context is most relevant to understanding this claim?"
            ]
        
        elif q_type == QuestionType.CONSISTENCY:
            follow_ups = [
                "What principles guide the resolution of such contradictions?",
                "How do we weigh conflicting pieces of evidence?",
                "What additional information might reconcile these differences?"
            ]
        
        return follow_ups[:2]  # Limit to 2 follow-ups
    
    def _determine_expected_answer_type(self, claim: str, q_type: QuestionType) -> str:
        """Determine what type of answer is expected"""
        answer_types = {
            QuestionType.VERIFICATION: "evidence_and_reasoning",
            QuestionType.CLARIFICATION: "clarification_and_context",
            QuestionType.DEEPER_ANALYSIS: "analysis_and_implications",
            QuestionType.CONSISTENCY: "reconciliation_strategy",
            QuestionType.AMBIGUITY: "disambiguation",
            QuestionType.CROSS_ALIGNMENT: "multimodal_consistency_check"
        }
        return answer_types.get(q_type, "general_response")
    
    def _calculate_question_confidence(self, claim: str, q_type: QuestionType, 
                                     components: Dict[str, Any]) -> float:
        """Calculate confidence in the generated question"""
        confidence = 0.7  # Base confidence
        
        # Boost confidence for well-structured claims
        if components.get('subject') and components.get('predicate'):
            confidence += 0.1
        
        # Boost confidence for claims with entities
        if components.get('entities'):
            confidence += 0.1
        
        # Boost confidence for specific question types
        if q_type in [QuestionType.VERIFICATION, QuestionType.CLARIFICATION]:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, q_type: QuestionType, claim: str, context: Dict[str, Any]) -> str:
        """Generate reasoning for why this question is being asked"""
        base_reasoning = self.reasoning_templates[q_type]
        
        # Add context-specific reasoning
        if context.get('contradictions'):
            base_reasoning += f" Specifically, this addresses contradictions found in: {context['contradictions'][:100]}..."
        elif context.get('ambiguous_terms'):
            base_reasoning += f" This focuses on clarifying ambiguous terms: {', '.join(context['ambiguous_terms'][:3])}."
        
        return base_reasoning
    
    def generate_question_sequence(self, claim: str, verification_context: Dict[str, Any]) -> List[SocraticInquiry]:
        """Generate a sequence of related Socratic questions for comprehensive verification"""
        logger.info(f"Generating question sequence for claim: {claim[:50]}...")
        
        sequence = []
        
        # Start with verification
        verification_q = self.generate_socratic_inquiry(claim, "verification")
        sequence.append(verification_q)
        
        # Add clarification if needed
        if verification_context.get('needs_clarification'):
            clarification_q = self.generate_socratic_inquiry(claim, "clarification", verification_context)
            sequence.append(clarification_q)
        
        # Add deeper analysis
        deeper_q = self.generate_socratic_inquiry(claim, "deeper_analysis", verification_context)
        sequence.append(deeper_q)
        
        # Add consistency check if there are previous claims
        if verification_context.get('previous_claims'):
            consistency_q = self.generate_socratic_inquiry(claim, "consistency", verification_context)
            sequence.append(consistency_q)
        
        logger.info(f"Generated sequence of {len(sequence)} questions")
        return sequence
