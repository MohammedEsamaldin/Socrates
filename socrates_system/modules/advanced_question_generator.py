"""
Advanced Socratic Question Generator - LLM-powered sophisticated questioning
Implements advanced Socratic methodology using open-source LLMs for deep inquiry
"""
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_manager import get_llm_manager, LLMTaskType
from modules.advanced_claim_extractor import AdvancedClaim, ClaimType, VerifiabilityLevel
from utils.logger import setup_logger

logger = setup_logger(__name__)

class QuestionType(Enum):
    """Types of Socratic questions for different inquiry purposes"""
    EVIDENCE_SEEKING = "evidence_seeking"         # Questions seeking supporting evidence
    ASSUMPTION_CHALLENGING = "assumption_challenging"  # Questions challenging assumptions
    PERSPECTIVE_SHIFTING = "perspective_shifting"     # Questions offering alternative viewpoints
    IMPLICATION_EXPLORING = "implication_exploring"   # Questions exploring consequences
    CLARIFICATION_SEEKING = "clarification_seeking"   # Questions seeking clarification
    CONTRADICTION_REVEALING = "contradiction_revealing"  # Questions revealing contradictions
    SOURCE_VERIFICATION = "source_verification"       # Questions about source reliability
    LOGICAL_CONSISTENCY = "logical_consistency"       # Questions testing logical consistency
    CONTEXTUAL_ANALYSIS = "contextual_analysis"       # Questions about context and scope
    CAUSAL_REASONING = "causal_reasoning"            # Questions about cause-effect relationships

class InquiryDepth(Enum):
    """Depth levels for Socratic inquiry"""
    SURFACE = "surface"           # Basic factual questions
    ANALYTICAL = "analytical"     # Questions requiring analysis
    EVALUATIVE = "evaluative"     # Questions requiring evaluation and judgment
    SYNTHETIC = "synthetic"       # Questions requiring synthesis of multiple concepts
    METACOGNITIVE = "metacognitive"  # Questions about thinking processes

@dataclass
class SocraticQuestion:
    """Enhanced Socratic question with comprehensive metadata"""
    question: str
    question_type: QuestionType
    inquiry_depth: InquiryDepth
    reasoning: str
    expected_answer_type: str
    confidence: float
    context: Dict[str, Any]
    follow_up_questions: List[str] = field(default_factory=list)
    verification_strategy: str = ""
    cognitive_load: str = "medium"  # low, medium, high
    prerequisite_knowledge: List[str] = field(default_factory=list)
    potential_biases: List[str] = field(default_factory=list)
    alternative_framings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SocraticInquiryChain:
    """Chain of related Socratic questions forming a complete inquiry"""
    primary_claim: str
    inquiry_goal: str
    questions: List[SocraticQuestion]
    logical_flow: List[str]
    expected_outcomes: List[str]
    confidence: float
    complexity_level: str
    estimated_duration: str
    prerequisites: List[str] = field(default_factory=list)

class AdvancedQuestionGenerator:
    """
    Advanced Socratic Question Generator using sophisticated LLM-powered methodology
    Generates contextually aware, strategically designed questions for claim verification
    """
    
    def __init__(self):
        """Initialize the advanced question generator"""
        self.llm_manager = get_llm_manager()
        self.question_cache = {}
        self.inquiry_patterns = {}
        
        # Initialize questioning strategies
        self._init_questioning_strategies()
        
        logger.info("AdvancedQuestionGenerator initialized with LLM integration")
    
    def _init_questioning_strategies(self):
        """Initialize sophisticated questioning strategies for different scenarios"""
        self.questioning_strategies = {
            ClaimType.FACTUAL: {
                "primary_questions": [QuestionType.EVIDENCE_SEEKING, QuestionType.SOURCE_VERIFICATION],
                "secondary_questions": [QuestionType.CONTRADICTION_REVEALING, QuestionType.CONTEXTUAL_ANALYSIS],
                "depth_progression": [InquiryDepth.SURFACE, InquiryDepth.ANALYTICAL, InquiryDepth.EVALUATIVE]
            },
            ClaimType.CAUSAL: {
                "primary_questions": [QuestionType.CAUSAL_REASONING, QuestionType.ASSUMPTION_CHALLENGING],
                "secondary_questions": [QuestionType.IMPLICATION_EXPLORING, QuestionType.PERSPECTIVE_SHIFTING],
                "depth_progression": [InquiryDepth.ANALYTICAL, InquiryDepth.EVALUATIVE, InquiryDepth.SYNTHETIC]
            },
            ClaimType.TEMPORAL: {
                "primary_questions": [QuestionType.EVIDENCE_SEEKING, QuestionType.CONTEXTUAL_ANALYSIS],
                "secondary_questions": [QuestionType.SOURCE_VERIFICATION, QuestionType.LOGICAL_CONSISTENCY],
                "depth_progression": [InquiryDepth.SURFACE, InquiryDepth.ANALYTICAL]
            },
            ClaimType.QUANTITATIVE: {
                "primary_questions": [QuestionType.SOURCE_VERIFICATION, QuestionType.EVIDENCE_SEEKING],
                "secondary_questions": [QuestionType.ASSUMPTION_CHALLENGING, QuestionType.CONTEXTUAL_ANALYSIS],
                "depth_progression": [InquiryDepth.SURFACE, InquiryDepth.ANALYTICAL, InquiryDepth.EVALUATIVE]
            },
            ClaimType.COMPARATIVE: {
                "primary_questions": [QuestionType.ASSUMPTION_CHALLENGING, QuestionType.PERSPECTIVE_SHIFTING],
                "secondary_questions": [QuestionType.EVIDENCE_SEEKING, QuestionType.CONTEXTUAL_ANALYSIS],
                "depth_progression": [InquiryDepth.ANALYTICAL, InquiryDepth.EVALUATIVE, InquiryDepth.SYNTHETIC]
            }
        }
        
        # Question templates for different types and depths
        self.question_templates = {
            QuestionType.EVIDENCE_SEEKING: {
                InquiryDepth.SURFACE: [
                    "What evidence supports the claim that {claim}?",
                    "Where can we find reliable information about {key_concept}?",
                    "What sources have documented {specific_fact}?"
                ],
                InquiryDepth.ANALYTICAL: [
                    "How reliable is the evidence for {claim}, and what factors might affect its credibility?",
                    "What types of evidence would be most convincing for {claim_type} claims like this?",
                    "How does the available evidence compare to what we would expect if {claim} were true?"
                ],
                InquiryDepth.EVALUATIVE: [
                    "Given the quality and quantity of available evidence, how confident should we be in {claim}?",
                    "What evidence would be needed to definitively prove or disprove {claim}?",
                    "How do we weigh conflicting pieces of evidence regarding {claim}?"
                ]
            },
            QuestionType.ASSUMPTION_CHALLENGING: {
                InquiryDepth.ANALYTICAL: [
                    "What assumptions underlie the claim that {claim}?",
                    "What if we assumed the opposite of {key_assumption} - how would that change our conclusion?",
                    "Are there unstated premises in the argument for {claim}?"
                ],
                InquiryDepth.EVALUATIVE: [
                    "How justified are the assumptions behind {claim}?",
                    "Which assumptions are most critical to the validity of {claim}?",
                    "What happens to {claim} if we remove or modify key assumptions?"
                ],
                InquiryDepth.SYNTHETIC: [
                    "How do the assumptions behind {claim} relate to broader theoretical frameworks?",
                    "What alternative assumption sets could lead to different conclusions about {topic}?",
                    "How might cultural or contextual assumptions influence our interpretation of {claim}?"
                ]
            },
            QuestionType.CAUSAL_REASONING: {
                InquiryDepth.ANALYTICAL: [
                    "What mechanism could explain how {cause} leads to {effect}?",
                    "Are there alternative explanations for the relationship between {cause} and {effect}?",
                    "What conditions must be present for {cause} to produce {effect}?"
                ],
                InquiryDepth.EVALUATIVE: [
                    "How strong is the causal relationship between {cause} and {effect}?",
                    "What evidence distinguishes causation from correlation in this case?",
                    "How do we rule out confounding factors in the {cause}-{effect} relationship?"
                ],
                InquiryDepth.SYNTHETIC: [
                    "How does the {cause}-{effect} relationship fit into larger causal networks?",
                    "What are the broader implications if {cause} truly causes {effect}?",
                    "How might this causal relationship interact with other causal mechanisms?"
                ]
            }
        }
    
    async def generate_socratic_inquiry(self, claim: AdvancedClaim, inquiry_goal: str = "verification", 
                                      context: Dict[str, Any] = None) -> SocraticInquiryChain:
        """
        Generate a comprehensive Socratic inquiry chain for a claim
        
        Args:
            claim: The AdvancedClaim to investigate
            inquiry_goal: Purpose of the inquiry (verification, exploration, clarification)
            context: Additional context for question generation
            
        Returns:
            SocraticInquiryChain with strategically designed questions
        """
        logger.info(f"Generating Socratic inquiry for claim: {claim.text[:50]}...")
        
        try:
            # Stage 1: Analyze claim and determine questioning strategy
            strategy = self._determine_questioning_strategy(claim, inquiry_goal)
            
            # Stage 2: Generate primary questions using LLM
            primary_questions = await self._generate_primary_questions(claim, strategy, context)
            
            # Stage 3: Generate follow-up and deeper questions
            secondary_questions = await self._generate_secondary_questions(claim, primary_questions, strategy)
            
            # Stage 4: Create logical flow and inquiry chain
            inquiry_chain = await self._create_inquiry_chain(claim, primary_questions + secondary_questions, inquiry_goal)
            
            logger.info(f"Generated inquiry chain with {len(inquiry_chain.questions)} questions")
            return inquiry_chain
            
        except Exception as e:
            logger.error(f"Socratic inquiry generation failed: {e}")
            return self._create_fallback_inquiry(claim, inquiry_goal)
    
    def _determine_questioning_strategy(self, claim: AdvancedClaim, inquiry_goal: str) -> Dict[str, Any]:
        """Determine the optimal questioning strategy based on claim characteristics"""
        strategy = self.questioning_strategies.get(claim.claim_type, self.questioning_strategies[ClaimType.FACTUAL])
        
        # Adjust strategy based on verifiability
        if claim.verifiability == VerifiabilityLevel.HIGH:
            strategy["focus"] = "evidence_verification"
            strategy["complexity"] = "medium"
        elif claim.verifiability == VerifiabilityLevel.MEDIUM:
            strategy["focus"] = "contextual_analysis"
            strategy["complexity"] = "high"
        elif claim.verifiability == VerifiabilityLevel.LOW:
            strategy["focus"] = "assumption_challenging"
            strategy["complexity"] = "high"
        else:  # SUBJECTIVE
            strategy["focus"] = "perspective_exploration"
            strategy["complexity"] = "low"
        
        # Adjust for inquiry goal
        if inquiry_goal == "exploration":
            strategy["depth_progression"].append(InquiryDepth.SYNTHETIC)
        elif inquiry_goal == "clarification":
            strategy["primary_questions"] = [QuestionType.CLARIFICATION_SEEKING, QuestionType.CONTEXTUAL_ANALYSIS]
        
        return strategy
    
    async def _generate_primary_questions(self, claim: AdvancedClaim, strategy: Dict[str, Any], 
                                        context: Dict[str, Any] = None) -> List[SocraticQuestion]:
        """Generate primary questions using LLM with strategic prompting"""
        primary_questions = []
        
        # Prepare context for LLM
        llm_context = {
            "claim": claim.text,
            "claim_type": claim.claim_type.value,
            "verifiability": claim.verifiability.value,
            "entities": [e.text for e in claim.entities],
            "relationships": [f"{r.source} {r.relation} {r.target}" for r in claim.relationships],
            "strategy_focus": strategy.get("focus", "verification"),
            "question_types": [qt.value for qt in strategy["primary_questions"]],
            "depth_levels": [d.value for d in strategy["depth_progression"]],
            **(context or {})
        }
        
        # Generate questions using LLM
        response = await self.llm_manager.generate_socratic_questions(claim.text, llm_context)
        
        if response.structured_output and 'questions' in response.structured_output:
            for q_data in response.structured_output['questions']:
                question = SocraticQuestion(
                    question=q_data.get('question', ''),
                    question_type=QuestionType(q_data.get('focus_area', 'evidence_seeking')),
                    inquiry_depth=InquiryDepth(q_data.get('depth', 'analytical')),
                    reasoning=q_data.get('reasoning', ''),
                    expected_answer_type=q_data.get('expected_answer_type', 'factual'),
                    confidence=q_data.get('confidence', 0.8),
                    context=llm_context,
                    verification_strategy=strategy.get("focus", "verification")
                )
                primary_questions.append(question)
        
        # Fallback: generate template-based questions if LLM fails
        if not primary_questions:
            primary_questions = self._generate_template_questions(claim, strategy)
        
        return primary_questions
    
    def _generate_template_questions(self, claim: AdvancedClaim, strategy: Dict[str, Any]) -> List[SocraticQuestion]:
        """Generate questions using templates as fallback"""
        questions = []
        
        for question_type in strategy["primary_questions"]:
            for depth in strategy["depth_progression"][:2]:  # Limit to first 2 depths for templates
                templates = self.question_templates.get(question_type, {}).get(depth, [])
                
                for template in templates[:1]:  # One question per type/depth
                    try:
                        question_text = template.format(
                            claim=claim.text,
                            key_concept=claim.entities[0].text if claim.entities else "this concept",
                            specific_fact=claim.text,
                            claim_type=claim.claim_type.value,
                            key_assumption="the main assumption",
                            cause=claim.relationships[0].source if claim.relationships else "the cause",
                            effect=claim.relationships[0].target if claim.relationships else "the effect",
                            topic=claim.text
                        )
                        
                        question = SocraticQuestion(
                            question=question_text,
                            question_type=question_type,
                            inquiry_depth=depth,
                            reasoning=f"Template-generated {question_type.value} question at {depth.value} level",
                            expected_answer_type="explanatory",
                            confidence=0.7,
                            context={"template_based": True}
                        )
                        questions.append(question)
                        
                    except (KeyError, IndexError):
                        continue
        
        return questions
    
    async def _generate_secondary_questions(self, claim: AdvancedClaim, primary_questions: List[SocraticQuestion], 
                                          strategy: Dict[str, Any]) -> List[SocraticQuestion]:
        """Generate follow-up and deeper questions based on primary questions"""
        secondary_questions = []
        
        for primary_q in primary_questions:
            # Generate follow-up questions using LLM
            follow_up_context = {
                "primary_question": primary_q.question,
                "question_type": primary_q.question_type.value,
                "claim": claim.text,
                "reasoning": primary_q.reasoning
            }
            
            response = await self.llm_manager.generate_reasoning(
                f"What follow-up questions would deepen the inquiry started by: {primary_q.question}",
                [claim.text],
                follow_up_context
            )
            
            # Parse follow-up questions from response
            follow_ups = self._extract_questions_from_response(response.content)
            
            for follow_up in follow_ups[:2]:  # Limit to 2 follow-ups per primary question
                secondary_question = SocraticQuestion(
                    question=follow_up,
                    question_type=primary_q.question_type,
                    inquiry_depth=InquiryDepth.EVALUATIVE,  # Follow-ups are typically deeper
                    reasoning=f"Follow-up to: {primary_q.question}",
                    expected_answer_type="analytical",
                    confidence=0.75,
                    context=follow_up_context
                )
                secondary_questions.append(secondary_question)
        
        return secondary_questions
    
    def _extract_questions_from_response(self, response_content: str) -> List[str]:
        """Extract questions from LLM response content"""
        import re
        
        # Look for question patterns
        question_patterns = [
            r'(?:^|\n)\s*\d+\.\s*([^?\n]+\?)',  # Numbered questions
            r'(?:^|\n)\s*[-•]\s*([^?\n]+\?)',   # Bulleted questions
            r'([A-Z][^?\n]+\?)',                # Any sentence ending with ?
        ]
        
        questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, response_content, re.MULTILINE)
            questions.extend(matches)
        
        # Clean and deduplicate
        cleaned_questions = []
        for q in questions:
            q = q.strip()
            if len(q) > 10 and q not in cleaned_questions:
                cleaned_questions.append(q)
        
        return cleaned_questions[:5]  # Limit to 5 questions
    
    async def _create_inquiry_chain(self, claim: AdvancedClaim, questions: List[SocraticQuestion], 
                                  inquiry_goal: str) -> SocraticInquiryChain:
        """Create a logical inquiry chain from generated questions"""
        
        # Sort questions by depth and type for logical flow
        sorted_questions = sorted(questions, key=lambda q: (q.inquiry_depth.value, q.question_type.value))
        
        # Create logical flow description
        logical_flow = []
        for i, question in enumerate(sorted_questions):
            flow_step = f"Step {i+1}: {question.question_type.value} at {question.inquiry_depth.value} level"
            logical_flow.append(flow_step)
        
        # Determine expected outcomes
        expected_outcomes = [
            "Enhanced understanding of claim validity",
            "Identification of evidence gaps",
            "Recognition of underlying assumptions",
            "Assessment of logical consistency"
        ]
        
        if claim.verifiability == VerifiabilityLevel.HIGH:
            expected_outcomes.append("Clear verification or refutation")
        else:
            expected_outcomes.append("Nuanced understanding of complexity")
        
        # Calculate overall confidence
        avg_confidence = sum(q.confidence for q in sorted_questions) / len(sorted_questions) if sorted_questions else 0.5
        
        return SocraticInquiryChain(
            primary_claim=claim.text,
            inquiry_goal=inquiry_goal,
            questions=sorted_questions,
            logical_flow=logical_flow,
            expected_outcomes=expected_outcomes,
            confidence=avg_confidence,
            complexity_level=self._assess_complexity(sorted_questions),
            estimated_duration=self._estimate_duration(sorted_questions),
            prerequisites=claim.evidence_requirements
        )
    
    def _assess_complexity(self, questions: List[SocraticQuestion]) -> str:
        """Assess the complexity level of the inquiry chain"""
        depth_scores = {
            InquiryDepth.SURFACE: 1,
            InquiryDepth.ANALYTICAL: 2,
            InquiryDepth.EVALUATIVE: 3,
            InquiryDepth.SYNTHETIC: 4,
            InquiryDepth.METACOGNITIVE: 5
        }
        
        if not questions:
            return "low"
        
        avg_depth = sum(depth_scores[q.inquiry_depth] for q in questions) / len(questions)
        
        if avg_depth <= 1.5:
            return "low"
        elif avg_depth <= 2.5:
            return "medium"
        elif avg_depth <= 3.5:
            return "high"
        else:
            return "very_high"
    
    def _estimate_duration(self, questions: List[SocraticQuestion]) -> str:
        """Estimate time needed for the inquiry"""
        base_time = len(questions) * 2  # 2 minutes per question
        
        complexity_multiplier = {
            "low": 1.0,
            "medium": 1.5,
            "high": 2.0,
            "very_high": 3.0
        }
        
        complexity = self._assess_complexity(questions)
        total_minutes = base_time * complexity_multiplier.get(complexity, 1.5)
        
        if total_minutes <= 10:
            return "5-10 minutes"
        elif total_minutes <= 20:
            return "10-20 minutes"
        elif total_minutes <= 30:
            return "20-30 minutes"
        else:
            return "30+ minutes"
    
    def _create_fallback_inquiry(self, claim: AdvancedClaim, inquiry_goal: str) -> SocraticInquiryChain:
        """Create a basic inquiry chain as fallback"""
        basic_questions = [
            SocraticQuestion(
                question=f"What evidence supports the claim that {claim.text}?",
                question_type=QuestionType.EVIDENCE_SEEKING,
                inquiry_depth=InquiryDepth.SURFACE,
                reasoning="Basic evidence-seeking question",
                expected_answer_type="factual",
                confidence=0.6,
                context={"fallback": True}
            ),
            SocraticQuestion(
                question=f"What assumptions underlie this claim?",
                question_type=QuestionType.ASSUMPTION_CHALLENGING,
                inquiry_depth=InquiryDepth.ANALYTICAL,
                reasoning="Basic assumption-challenging question",
                expected_answer_type="analytical",
                confidence=0.6,
                context={"fallback": True}
            )
        ]
        
        return SocraticInquiryChain(
            primary_claim=claim.text,
            inquiry_goal=inquiry_goal,
            questions=basic_questions,
            logical_flow=["Basic evidence inquiry", "Assumption analysis"],
            expected_outcomes=["Basic understanding of evidence and assumptions"],
            confidence=0.6,
            complexity_level="low",
            estimated_duration="5-10 minutes"
        )
    
    async def generate_adaptive_questions(self, claim: AdvancedClaim, previous_answers: List[str], 
                                        context: Dict[str, Any] = None) -> List[SocraticQuestion]:
        """
        Generate adaptive questions based on previous answers
        Implements dynamic questioning that responds to user responses
        """
        logger.info("Generating adaptive questions based on previous answers")
        
        # Analyze previous answers to determine next questions
        analysis_context = {
            "claim": claim.text,
            "previous_answers": previous_answers,
            "claim_context": context or {}
        }
        
        response = await self.llm_manager.generate_reasoning(
            "Based on the previous answers, what questions would best continue this Socratic inquiry?",
            previous_answers,
            analysis_context
        )
        
        # Extract and structure adaptive questions
        adaptive_questions = []
        extracted_questions = self._extract_questions_from_response(response.content)
        
        for i, question_text in enumerate(extracted_questions):
            adaptive_question = SocraticQuestion(
                question=question_text,
                question_type=QuestionType.CLARIFICATION_SEEKING,  # Default for adaptive questions
                inquiry_depth=InquiryDepth.ANALYTICAL,
                reasoning=f"Adaptive question based on previous responses: {response.reasoning or 'LLM analysis'}",
                expected_answer_type="clarifying",
                confidence=0.8,
                context=analysis_context,
                verification_strategy="adaptive_inquiry"
            )
            adaptive_questions.append(adaptive_question)
        
        return adaptive_questions
    
    def get_inquiry_summary(self, inquiry_chain: SocraticInquiryChain) -> Dict[str, Any]:
        """Get comprehensive summary of Socratic inquiry chain"""
        question_types = {}
        depth_levels = {}
        
        for question in inquiry_chain.questions:
            question_types[question.question_type.value] = question_types.get(question.question_type.value, 0) + 1
            depth_levels[question.inquiry_depth.value] = depth_levels.get(question.inquiry_depth.value, 0) + 1
        
        return {
            "inquiry_goal": inquiry_chain.inquiry_goal,
            "total_questions": len(inquiry_chain.questions),
            "question_types": question_types,
            "depth_levels": depth_levels,
            "complexity_level": inquiry_chain.complexity_level,
            "estimated_duration": inquiry_chain.estimated_duration,
            "confidence": inquiry_chain.confidence,
            "logical_flow_steps": len(inquiry_chain.logical_flow),
            "expected_outcomes": len(inquiry_chain.expected_outcomes),
            "methodology": "Advanced LLM-powered Socratic questioning"
        }
