"""
Enhanced External Factuality Checker - Local LLM-powered with global knowledge detection
Implements proper Socratic Q&A methodology: generate questions → search answers → compare
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
from modules.advanced_claim_extractor import AdvancedClaim, ClaimType, VerifiabilityLevel
from modules.advanced_question_generator import SocraticQuestion, QuestionType
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GlobalKnowledgeType(Enum):
    """Types of global knowledge that require external verification"""
    SCIENTIFIC_FACTS = "scientific_facts"
    HISTORICAL_EVENTS = "historical_events"
    STATISTICAL_DATA = "statistical_data"
    GEOGRAPHICAL_INFO = "geographical_info"
    BIOGRAPHICAL_INFO = "biographical_info"
    CURRENT_EVENTS = "current_events"
    TECHNICAL_SPECS = "technical_specs"
    LEGAL_FACTS = "legal_facts"
    NONE = "none"  # No global knowledge needed

class VerificationMethod(Enum):
    """Methods for external verification"""
    SOCRATIC_QA = "socratic_qa"           # Generate questions and find answers
    DIRECT_LOOKUP = "direct_lookup"       # Direct factual lookup
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # Compare multiple sources
    TEMPORAL_VERIFICATION = "temporal_verification"  # Time-based verification
    STATISTICAL_CHECK = "statistical_check"  # Numerical/statistical verification

@dataclass
class GlobalKnowledgeRequirement:
    """Requirement for global knowledge verification"""
    claim_text: str
    knowledge_type: GlobalKnowledgeType
    verification_method: VerificationMethod
    confidence: float
    reasoning: str
    key_concepts: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None
    geographical_context: Optional[str] = None

@dataclass
class SocraticAnswer:
    """Answer to a Socratic question with verification metadata"""
    question: str
    answer: str
    confidence: float
    sources: List[str]
    reasoning: str
    verification_method: str
    contradictions: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class FactualityVerificationResult:
    """Result of external factuality verification"""
    claim: str
    requires_global_knowledge: bool
    global_knowledge_type: GlobalKnowledgeType
    verification_method: VerificationMethod
    socratic_questions: List[SocraticQuestion]
    socratic_answers: List[SocraticAnswer]
    overall_status: str  # VERIFIED, CONTRADICTED, INSUFFICIENT_EVIDENCE, INCONCLUSIVE
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_reasoning: str
    knowledge_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0

class EnhancedExternalFactualityChecker:
    """
    Enhanced External Factuality Checker with local LLM integration
    Implements proper Socratic methodology and global knowledge detection
    """
    
    def __init__(self):
        """Initialize the enhanced external factuality checker"""
        self.llm_manager = get_llm_manager()
        self.verification_cache = {}
        
        # Initialize global knowledge patterns
        self._init_global_knowledge_patterns()
        
        # Initialize verification strategies
        self._init_verification_strategies()
        
        logger.info("EnhancedExternalFactualityChecker initialized with LLM integration")
    
    def _init_global_knowledge_patterns(self):
        """Initialize patterns for detecting global knowledge requirements"""
        self.global_knowledge_patterns = {
            GlobalKnowledgeType.SCIENTIFIC_FACTS: [
                r'\b(research|study|experiment|hypothesis|theory|scientific|discovery)\b',
                r'\b(DNA|protein|molecule|atom|physics|chemistry|biology)\b',
                r'\b(published|journal|peer.reviewed|Nature|Science)\b'
            ],
            GlobalKnowledgeType.HISTORICAL_EVENTS: [
                r'\b(in \d{4}|century|historical|ancient|medieval|war|battle)\b',
                r'\b(founded|established|occurred|happened|during|before|after)\b',
                r'\b(empire|kingdom|civilization|dynasty|revolution)\b'
            ],
            GlobalKnowledgeType.STATISTICAL_DATA: [
                r'\b(\d+(?:\.\d+)?(?:%|percent|million|billion|thousand))\b',
                r'\b(statistics|data|survey|census|poll|rate|average)\b',
                r'\b(according to|reports|shows|indicates|reveals)\b'
            ],
            GlobalKnowledgeType.GEOGRAPHICAL_INFO: [
                r'\b(located|situated|capital|country|city|continent|ocean)\b',
                r'\b(population|area|climate|geography|border|region)\b',
                r'\b(latitude|longitude|elevation|coordinates)\b'
            ],
            GlobalKnowledgeType.BIOGRAPHICAL_INFO: [
                r'\b(born|died|lived|age|biography|life|career)\b',
                r'\b(author|scientist|politician|artist|inventor|leader)\b',
                r'\b(Nobel|award|prize|achievement|accomplishment)\b'
            ],
            GlobalKnowledgeType.CURRENT_EVENTS: [
                r'\b(recently|today|yesterday|this year|current|latest)\b',
                r'\b(news|announced|reported|breaking|update)\b',
                r'\b(government|policy|election|pandemic|crisis)\b'
            ],
            GlobalKnowledgeType.TECHNICAL_SPECS: [
                r'\b(specification|technical|version|model|standard)\b',
                r'\b(software|hardware|protocol|algorithm|system)\b',
                r'\b(performance|capacity|speed|memory|storage)\b'
            ],
            GlobalKnowledgeType.LEGAL_FACTS: [
                r'\b(law|legal|court|judge|ruling|decision|statute)\b',
                r'\b(constitution|amendment|regulation|policy|act)\b',
                r'\b(rights|illegal|legal|lawsuit|trial|verdict)\b'
            ]
        }
    
    def _init_verification_strategies(self):
        """Initialize verification strategies for different knowledge types"""
        self.verification_strategies = {
            GlobalKnowledgeType.SCIENTIFIC_FACTS: {
                "method": VerificationMethod.SOCRATIC_QA,
                "question_types": [QuestionType.EVIDENCE_SEEKING, QuestionType.SOURCE_VERIFICATION],
                "answer_sources": ["scientific_literature", "research_databases", "expert_knowledge"],
                "confidence_threshold": 0.8
            },
            GlobalKnowledgeType.HISTORICAL_EVENTS: {
                "method": VerificationMethod.TEMPORAL_VERIFICATION,
                "question_types": [QuestionType.CONTEXTUAL_ANALYSIS, QuestionType.EVIDENCE_SEEKING],
                "answer_sources": ["historical_records", "academic_sources", "primary_documents"],
                "confidence_threshold": 0.7
            },
            GlobalKnowledgeType.STATISTICAL_DATA: {
                "method": VerificationMethod.STATISTICAL_CHECK,
                "question_types": [QuestionType.SOURCE_VERIFICATION, QuestionType.ASSUMPTION_CHALLENGING],
                "answer_sources": ["official_statistics", "government_data", "research_reports"],
                "confidence_threshold": 0.9
            },
            GlobalKnowledgeType.GEOGRAPHICAL_INFO: {
                "method": VerificationMethod.DIRECT_LOOKUP,
                "question_types": [QuestionType.EVIDENCE_SEEKING],
                "answer_sources": ["geographical_databases", "official_sources", "maps"],
                "confidence_threshold": 0.9
            },
            GlobalKnowledgeType.BIOGRAPHICAL_INFO: {
                "method": VerificationMethod.COMPARATIVE_ANALYSIS,
                "question_types": [QuestionType.SOURCE_VERIFICATION, QuestionType.CONTRADICTION_REVEALING],
                "answer_sources": ["biographical_databases", "official_records", "multiple_sources"],
                "confidence_threshold": 0.8
            }
        }
    
    async def verify_claim(self, claim: AdvancedClaim, context: Dict[str, Any] = None) -> FactualityVerificationResult:
        """
        Verify claim using enhanced methodology with global knowledge detection
        
        Args:
            claim: AdvancedClaim to verify
            context: Additional context for verification
            
        Returns:
            FactualityVerificationResult with comprehensive verification data
        """
        start_time = datetime.now()
        logger.info(f"Verifying claim: {claim.text[:50]}...")
        # Defensive guard: detect placeholder/empty claim text used in logs/prompts
        _ct = (getattr(claim, "text", None) or "").strip()
        if not _ct:
            logger.warning("[ClaimInput] Empty claim text passed; provide actual claim content for verification and logging clarity")
        elif _ct.lower() in {"claim", "statement", "assertion", "hypothesis"}:
            logger.warning(f"[ClaimInput] Placeholder-like claim text detected: '{_ct}'. Use the real claim content in prompts/logs.")
        
        try:
            # Stage 1: Detect if global knowledge is required
            global_knowledge_req = await self._detect_global_knowledge_requirement(claim)
            
            # Stage 2: If no global knowledge needed, skip external verification
            if global_knowledge_req.knowledge_type == GlobalKnowledgeType.NONE:
                return self._create_no_verification_result(claim, "No global knowledge required")
            
            # Stage 3: Generate Socratic questions for verification
            socratic_questions = await self._generate_verification_questions(claim, global_knowledge_req)
            
            # Stage 4: Find answers to Socratic questions using local LLM
            socratic_answers = await self._find_socratic_answers(socratic_questions, claim, context)
            
            # Stage 5: Compare answers and assess factuality
            verification_result = await self._assess_factuality_from_answers(
                claim, socratic_questions, socratic_answers, global_knowledge_req
            )
            
            # Stage 6: Generate verification reasoning
            verification_result.verification_reasoning = await self._generate_verification_reasoning(
                claim, socratic_questions, socratic_answers, verification_result
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            verification_result.processing_time = processing_time
            
            # Log claim with verdict and evidence counts
            logger.info(
                f"[Claim+Verdict] claim='{claim.text}' | status={verification_result.overall_status} | "
                f"conf={verification_result.confidence:.2f} | "
                f"support_n={len(verification_result.supporting_evidence)} | "
                f"contradict_n={len(verification_result.contradicting_evidence)}"
            )
            return verification_result
            
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_error_result(claim, str(e), processing_time)
    
    async def _detect_global_knowledge_requirement(self, claim: AdvancedClaim) -> GlobalKnowledgeRequirement:
        """Detect if claim requires global knowledge verification"""
        logger.debug("Detecting global knowledge requirements")
        
        claim_text = claim.text.lower()
        detected_types = []
        
        # Pattern-based detection
        for knowledge_type, patterns in self.global_knowledge_patterns.items():
            for pattern in patterns:
                if re.search(pattern, claim_text, re.IGNORECASE):
                    detected_types.append(knowledge_type)
                    break
        
        # Use LLM for sophisticated detection
        llm_context = {
            "claim": claim.text,
            "claim_type": claim.claim_type.value,
            "entities": [e.text for e in claim.entities],
            "verifiability": claim.verifiability.value
        }
        
        response = await self.llm_manager.generate_reasoning(
            "Does this claim require external/global knowledge verification? What type of global knowledge is needed?",
            [claim.text],
            llm_context
        )
        
        # Determine primary knowledge type
        if detected_types:
            primary_type = detected_types[0]  # Take first detected type
        else:
            # Parse from LLM response
            primary_type = self._parse_knowledge_type_from_response(response.content)
        
        # Determine verification method
        verification_method = self.verification_strategies.get(
            primary_type, {}
        ).get("method", VerificationMethod.SOCRATIC_QA)
        
        return GlobalKnowledgeRequirement(
            claim_text=claim.text,
            knowledge_type=primary_type,
            verification_method=verification_method,
            confidence=0.8,
            reasoning=response.content[:200] if response.content else "Pattern-based detection",
            key_concepts=[e.text for e in claim.entities]
        )
    
    def _parse_knowledge_type_from_response(self, response_content: str) -> GlobalKnowledgeType:
        """Parse knowledge type from LLM response"""
        content_lower = response_content.lower()
        
        # Check for knowledge type keywords
        type_keywords = {
            GlobalKnowledgeType.SCIENTIFIC_FACTS: ["scientific", "research", "study", "experiment"],
            GlobalKnowledgeType.HISTORICAL_EVENTS: ["historical", "history", "past", "event"],
            GlobalKnowledgeType.STATISTICAL_DATA: ["statistical", "data", "number", "percentage"],
            GlobalKnowledgeType.GEOGRAPHICAL_INFO: ["geographical", "location", "place", "country"],
            GlobalKnowledgeType.BIOGRAPHICAL_INFO: ["biographical", "person", "life", "born"],
            GlobalKnowledgeType.CURRENT_EVENTS: ["current", "recent", "news", "today"],
            GlobalKnowledgeType.TECHNICAL_SPECS: ["technical", "specification", "technology"],
            GlobalKnowledgeType.LEGAL_FACTS: ["legal", "law", "court", "regulation"]
        }
        
        for knowledge_type, keywords in type_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return knowledge_type
        
        # Check if LLM says no global knowledge needed
        if any(phrase in content_lower for phrase in ["no global", "not require", "no external", "local knowledge"]):
            return GlobalKnowledgeType.NONE
        
        # Default to scientific facts if uncertain
        return GlobalKnowledgeType.SCIENTIFIC_FACTS
    
    async def _generate_verification_questions(self, claim: AdvancedClaim, 
                                             global_knowledge_req: GlobalKnowledgeRequirement) -> List[SocraticQuestion]:
        """Generate Socratic questions for verification"""
        logger.debug("Generating verification questions")
        
        # Get strategy for this knowledge type
        strategy = self.verification_strategies.get(
            global_knowledge_req.knowledge_type,
            self.verification_strategies[GlobalKnowledgeType.SCIENTIFIC_FACTS]
        )
        
        # Generate questions using LLM with specific focus on verification
        context = {
            "claim": claim.text,
            "knowledge_type": global_knowledge_req.knowledge_type.value,
            "verification_method": global_knowledge_req.verification_method.value,
            "key_concepts": global_knowledge_req.key_concepts,
            "question_types": [qt.value for qt in strategy["question_types"]],
            "answer_sources": strategy["answer_sources"]
        }
        
        response = await self.llm_manager.generate_socratic_questions(claim.text, context)
        
        verification_questions = []
        if response.structured_output and 'questions' in response.structured_output:
            for q_data in response.structured_output['questions']:
                question = SocraticQuestion(
                    question=q_data.get('question', ''),
                    question_type=QuestionType(q_data.get('focus_area', 'evidence_seeking')),
                    inquiry_depth=q_data.get('depth', 'analytical'),
                    reasoning=q_data.get('reasoning', ''),
                    expected_answer_type=q_data.get('expected_answer_type', 'factual'),
                    confidence=q_data.get('confidence', 0.8),
                    context=context,
                    verification_strategy="external_factuality"
                )
                verification_questions.append(question)
        
        # Fallback: generate basic verification questions
        if not verification_questions:
            verification_questions = self._generate_basic_verification_questions(claim, global_knowledge_req)
        
        return verification_questions
    
    def _generate_basic_verification_questions(self, claim: AdvancedClaim, 
                                            global_knowledge_req: GlobalKnowledgeRequirement) -> List[SocraticQuestion]:
        """Generate basic verification questions as fallback"""
        basic_questions = [
            SocraticQuestion(
                question=f"What reliable sources confirm that {claim.text}?",
                question_type=QuestionType.SOURCE_VERIFICATION,
                inquiry_depth="surface",
                reasoning="Basic source verification question",
                expected_answer_type="factual",
                confidence=0.7,
                context={"fallback": True}
            ),
            SocraticQuestion(
                question=f"What evidence supports the claim that {claim.text}?",
                question_type=QuestionType.EVIDENCE_SEEKING,
                inquiry_depth="analytical",
                reasoning="Basic evidence-seeking question",
                expected_answer_type="evidential",
                confidence=0.7,
                context={"fallback": True}
            )
        ]
        
        return basic_questions
    
    async def _find_socratic_answers(self, questions: List[SocraticQuestion], 
                                   claim: AdvancedClaim, context: Dict[str, Any] = None) -> List[SocraticAnswer]:
        """Find answers to Socratic questions using local LLM"""
        logger.debug(f"Finding answers to {len(questions)} Socratic questions")
        
        socratic_answers = []
        
        for question in questions:
            # Use LLM to generate answer based on its knowledge
            answer_context = {
                "question": question.question,
                "claim": claim.text,
                "question_type": question.question_type.value,
                "expected_answer_type": question.expected_answer_type,
                "verification_context": context or {}
            }
            
            response = await self.llm_manager.generate_reasoning(
                question.question,
                [claim.text],
                answer_context
            )
            
            # Parse answer and assess confidence
            answer_confidence = self._assess_answer_confidence(response.content, question)
            
            # Extract supporting evidence and contradictions
            supporting_evidence, contradictions = self._extract_evidence_from_answer(response.content)
            
            socratic_answer = SocraticAnswer(
                question=question.question,
                answer=response.content,
                confidence=answer_confidence,
                sources=["local_llm_knowledge"],  # Since we're using local LLM
                reasoning=response.reasoning or "LLM-generated answer",
                verification_method="llm_knowledge_base",
                supporting_evidence=supporting_evidence,
                contradictions=contradictions
            )
            
            # Log claim + evidence pair from Socratic answer
            try:
                support_prev = "; ".join([str(ev)[:180] for ev in (supporting_evidence or [])[:2]])
                contradict_prev = "; ".join([str(ev)[:180] for ev in (contradictions or [])[:2]])
                ans_prev = (response.content or "")[:200].replace("\n", " ")
                logger.info(
                    f"[Claim+Evidence] claim='{claim.text}' | source=SocraticLLM | q='{question.question}' | "
                    f"conf={answer_confidence:.2f} | support='{support_prev}' | contradict='{contradict_prev}' | "
                    f"answer='{ans_prev}'"
                )
            except Exception:
                # Avoid any logging exceptions from odd content
                logger.info(f"[Claim+Evidence] claim='{claim.text}' | source=SocraticLLM | q='{question.question}'")
            
            socratic_answers.append(socratic_answer)
        
        return socratic_answers
    
    def _assess_answer_confidence(self, answer_content: str, question: SocraticQuestion) -> float:
        """Assess confidence in the LLM's answer"""
        content_lower = answer_content.lower()
        
        # High confidence indicators
        high_confidence_indicators = ["definitely", "certainly", "clearly", "established", "confirmed"]
        # Low confidence indicators
        low_confidence_indicators = ["possibly", "might", "could", "uncertain", "unclear", "unknown"]
        # Uncertainty indicators
        uncertainty_indicators = ["i don't know", "not sure", "cannot determine", "insufficient information"]
        
        if any(indicator in content_lower for indicator in uncertainty_indicators):
            return 0.2
        elif any(indicator in content_lower for indicator in low_confidence_indicators):
            return 0.4
        elif any(indicator in content_lower for indicator in high_confidence_indicators):
            return 0.9
        else:
            return 0.6  # Default moderate confidence
    
    def _extract_evidence_from_answer(self, answer_content: str) -> Tuple[List[str], List[str]]:
        """Extract supporting evidence and contradictions from answer"""
        supporting_evidence = []
        contradictions = []
        
        # Simple pattern-based extraction
        sentences = re.split(r'[.!?]+', answer_content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            sentence_lower = sentence.lower()
            
            # Look for supporting evidence patterns
            if any(pattern in sentence_lower for pattern in ["evidence shows", "research indicates", "studies confirm"]):
                supporting_evidence.append(sentence)
            
            # Look for contradiction patterns
            elif any(pattern in sentence_lower for pattern in ["however", "but", "contradicts", "disputes"]):
                contradictions.append(sentence)
        
        return supporting_evidence, contradictions
    
    async def _assess_factuality_from_answers(self, claim: AdvancedClaim, 
                                           questions: List[SocraticQuestion],
                                           answers: List[SocraticAnswer],
                                           global_knowledge_req: GlobalKnowledgeRequirement) -> FactualityVerificationResult:
        """Assess overall factuality by comparing Socratic answers"""
        logger.debug("Assessing factuality from Socratic answers")
        
        # Collect all evidence
        all_supporting_evidence = []
        all_contradicting_evidence = []
        
        for answer in answers:
            all_supporting_evidence.extend(answer.supporting_evidence)
            all_contradicting_evidence.extend(answer.contradictions)
        
        # Calculate overall confidence
        answer_confidences = [answer.confidence for answer in answers]
        overall_confidence = sum(answer_confidences) / len(answer_confidences) if answer_confidences else 0.5
        
        # Determine overall status
        if overall_confidence >= 0.8 and len(all_supporting_evidence) > len(all_contradicting_evidence):
            overall_status = "VERIFIED"
        elif overall_confidence <= 0.3 or len(all_contradicting_evidence) > len(all_supporting_evidence):
            overall_status = "CONTRADICTED"
        elif overall_confidence >= 0.5:
            overall_status = "INCONCLUSIVE"
        else:
            overall_status = "INSUFFICIENT_EVIDENCE"
        
        # Identify knowledge gaps
        knowledge_gaps = []
        for answer in answers:
            if answer.confidence < 0.4:
                knowledge_gaps.append(f"Insufficient information for: {answer.question}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_status, knowledge_gaps, global_knowledge_req)
        
        # Log summary of claim + evidence prior to returning
        try:
            support_preview = "; ".join([str(ev) for ev in all_supporting_evidence[:2]])[:250].replace("\n", " ")
            contradict_preview = "; ".join([str(ev) for ev in all_contradicting_evidence[:2]])[:250].replace("\n", " ")
            logger.info(
                f"[Claim+Evidence Summary] claim='{claim.text}' | support='{support_preview}' | contradict='{contradict_preview}'"
            )
        except Exception:
            logger.info(f"[Claim+Evidence Summary] claim='{claim.text}'")

        return FactualityVerificationResult(
            claim=claim.text,
            requires_global_knowledge=True,
            global_knowledge_type=global_knowledge_req.knowledge_type,
            verification_method=global_knowledge_req.verification_method,
            socratic_questions=questions,
            socratic_answers=answers,
            overall_status=overall_status,
            confidence=overall_confidence,
            supporting_evidence=all_supporting_evidence,
            contradicting_evidence=all_contradicting_evidence,
            verification_reasoning="",  # Will be filled later
            knowledge_gaps=knowledge_gaps,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, overall_status: str, knowledge_gaps: List[str], 
                                global_knowledge_req: GlobalKnowledgeRequirement) -> List[str]:
        """Generate recommendations based on verification results"""
        recommendations = []
        
        if overall_status == "INSUFFICIENT_EVIDENCE":
            recommendations.append("Seek additional authoritative sources")
            recommendations.append(f"Consult {global_knowledge_req.knowledge_type.value} experts")
        
        elif overall_status == "CONTRADICTED":
            recommendations.append("Review contradicting evidence carefully")
            recommendations.append("Consider alternative interpretations")
        
        elif overall_status == "INCONCLUSIVE":
            recommendations.append("Gather more specific evidence")
            recommendations.append("Clarify ambiguous aspects of the claim")
        
        if knowledge_gaps:
            recommendations.append("Address identified knowledge gaps")
        
        return recommendations
    
    async def _generate_verification_reasoning(self, claim: AdvancedClaim,
                                             questions: List[SocraticQuestion],
                                             answers: List[SocraticAnswer],
                                             result: FactualityVerificationResult) -> str:
        """Generate comprehensive verification reasoning"""
        
        reasoning_context = {
            "claim": claim.text,
            "questions": [q.question for q in questions],
            "answers": [a.answer for a in answers],
            "overall_status": result.overall_status,
            "confidence": result.confidence,
            "supporting_evidence": result.supporting_evidence,
            "contradicting_evidence": result.contradicting_evidence
        }
        
        response = await self.llm_manager.generate_reasoning(
            f"Provide comprehensive reasoning for why the claim '{claim.text}' has verification status '{result.overall_status}' with confidence {result.confidence:.2f}",
            [claim.text] + [a.answer for a in answers],
            reasoning_context
        )
        
        return response.content
    
    def _create_no_verification_result(self, claim: AdvancedClaim, reason: str) -> FactualityVerificationResult:
        """Create result for claims that don't require global knowledge verification"""
        return FactualityVerificationResult(
            claim=claim.text,
            requires_global_knowledge=False,
            global_knowledge_type=GlobalKnowledgeType.NONE,
            verification_method=VerificationMethod.DIRECT_LOOKUP,
            socratic_questions=[],
            socratic_answers=[],
            overall_status="SKIPPED",
            confidence=1.0,
            supporting_evidence=[],
            contradicting_evidence=[],
            verification_reasoning=reason,
            processing_time=0.0
        )
    
    def _create_error_result(self, claim: AdvancedClaim, error_message: str, processing_time: float) -> FactualityVerificationResult:
        """Create error result for failed verification"""
        return FactualityVerificationResult(
            claim=claim.text,
            requires_global_knowledge=True,
            global_knowledge_type=GlobalKnowledgeType.SCIENTIFIC_FACTS,
            verification_method=VerificationMethod.SOCRATIC_QA,
            socratic_questions=[],
            socratic_answers=[],
            overall_status="ERROR",
            confidence=0.0,
            supporting_evidence=[],
            contradicting_evidence=[],
            verification_reasoning=f"Verification failed: {error_message}",
            processing_time=processing_time
        )
    
    def get_verification_summary(self, result: FactualityVerificationResult) -> Dict[str, Any]:
        """Get comprehensive summary of verification result"""
        return {
            "claim": result.claim,
            "requires_global_knowledge": result.requires_global_knowledge,
            "knowledge_type": result.global_knowledge_type.value,
            "verification_method": result.verification_method.value,
            "overall_status": result.overall_status,
            "confidence": result.confidence,
            "questions_asked": len(result.socratic_questions),
            "answers_found": len(result.socratic_answers),
            "supporting_evidence_count": len(result.supporting_evidence),
            "contradicting_evidence_count": len(result.contradicting_evidence),
            "knowledge_gaps": len(result.knowledge_gaps),
            "recommendations": len(result.recommendations),
            "processing_time": result.processing_time,
            "methodology": "Enhanced Socratic Q&A with local LLM"
        }
