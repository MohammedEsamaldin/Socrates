"""
Advanced Socrates Agent - Enhanced central coordinator with local LLM integration
Implements sophisticated Socratic dialogue methodology with ZeroFEC and KALA methodologies
"""
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_manager import get_llm_manager
from modules.advanced_claim_extractor import AdvancedClaimExtractor, AdvancedClaim
from modules.advanced_question_generator import AdvancedQuestionGenerator, SocraticInquiryChain
from modules.kala_knowledge_graph import KALAKnowledgeGraph
from modules.enhanced_external_factuality_checker import EnhancedExternalFactualityChecker, FactualityVerificationResult
from modules.cross_alignment_checker import CrossAlignmentChecker
from modules.self_contradiction_checker import SelfContradictionChecker
from modules.ambiguity_checker import AmbiguityChecker
from modules.clarification_handler import ClarificationHandler
from utils.logger import setup_logger

logger = setup_logger(__name__)

class VerificationStage(Enum):
    """Stages of the advanced verification pipeline"""
    CLAIM_EXTRACTION = "claim_extraction"
    CROSS_ALIGNMENT = "cross_alignment"
    EXTERNAL_FACTUALITY = "external_factuality"
    SELF_CONTRADICTION = "self_contradiction"
    AMBIGUITY_CHECK = "ambiguity_check"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    SOCRATIC_SYNTHESIS = "socratic_synthesis"

class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    SKIP = "SKIP"
    ERROR = "ERROR"

@dataclass
class AdvancedVerificationResult:
    """Enhanced verification result with comprehensive metadata"""
    claim: AdvancedClaim
    stage_results: Dict[VerificationStage, Dict[str, Any]]
    overall_status: CheckStatus
    confidence: float
    socratic_inquiry: Optional[SocraticInquiryChain]
    factuality_result: Optional[FactualityVerificationResult]
    evidence: List[str]
    contradictions: List[str]
    clarification_needed: Optional[str]
    knowledge_integrated: bool
    processing_time: float
    timestamp: datetime

class AdvancedSocratesAgent:
    """
    Advanced Socrates Agent - Enhanced central coordinator implementing sophisticated
    Socratic methodology with local LLM integration, ZeroFEC claim extraction,
    and KALA knowledge graph management
    """
    
    def __init__(self):
        """Initialize the Advanced Socrates Agent"""
        logger.info("Initializing Advanced Socrates Agent...")
        
        # Initialize LLM manager
        self.llm_manager = get_llm_manager()
        
        # Initialize advanced modules
        self.claim_extractor = AdvancedClaimExtractor()
        self.question_generator = AdvancedQuestionGenerator()
        self.knowledge_graph = KALAKnowledgeGraph()
        self.external_factuality_checker = EnhancedExternalFactualityChecker()
        
        # Initialize existing modules (enhanced compatibility)
        self.cross_alignment_checker = CrossAlignmentChecker()
        self.self_contradiction_checker = SelfContradictionChecker()
        self.ambiguity_checker = AmbiguityChecker()
        self.clarification_handler = ClarificationHandler()
        
        # Session state
        self.session_id = None
        self.conversation_history = []
        self.verified_claims = []
        self.session_domain = "general"
        
        logger.info("Advanced Socrates Agent initialized successfully")
    
    def start_session(self, session_id: str = None, domain: str = "general") -> str:
        """Start a new advanced verification session"""
        if session_id is None:
            session_id = f"advanced_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.session_domain = domain
        self.conversation_history = []
        self.verified_claims = []
        
        # Initialize session knowledge graph
        self.knowledge_graph.initialize_session(session_id, domain)
        
        logger.info(f"Started advanced session: {session_id} (domain: {domain})")
        return session_id
    
    async def process_user_input(self, user_input: str, image_path: Optional[str] = None, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Advanced processing pipeline implementing sophisticated Socratic methodology
        
        Args:
            user_input: User's text input
            image_path: Optional path to image for multimodal analysis
            context: Additional context for processing
            
        Returns:
            Comprehensive verification results with Socratic dialogue
        """
        start_time = datetime.now()
        logger.info(f"Processing user input with advanced methodology: {user_input[:100]}...")
        
        if not self.session_id:
            self.start_session()
        
        # Store input in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "image_path": image_path,
            "context": context or {}
        })
        
        try:
            # Stage 1: Advanced Claim Extraction (ZeroFEC-inspired)
            logger.info("Stage 1: Advanced Claim Extraction")
            claims = await self.claim_extractor.extract_claims(user_input, context)
            
            if not claims:
                return self._create_no_claims_response(user_input)
            
            # Stage 2: Process each claim through advanced verification pipeline
            verification_results = []
            for claim in claims:
                result = await self._verify_claim_advanced(claim, user_input, image_path, context)
                verification_results.append(result)
                
                # Store verified claim
                self.verified_claims.append(result)
            
            # Stage 3: Generate comprehensive Socratic response
            response = await self._compile_advanced_socratic_response(
                verification_results, user_input, context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            response["processing_time"] = processing_time
            
            logger.info(f"Advanced processing complete in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Advanced processing failed: {e}")
            return self._create_error_response(user_input, str(e))
    
    async def _verify_claim_advanced(self, claim: AdvancedClaim, original_input: str, 
                                   image_path: Optional[str], context: Dict[str, Any] = None) -> AdvancedVerificationResult:
        """
        Advanced claim verification using sophisticated multi-stage pipeline
        """
        start_time = datetime.now()
        logger.info(f"Advanced verification for claim: {claim.text[:50]}...")
        
        stage_results = {}
        overall_status = CheckStatus.PASS
        confidence = 1.0
        evidence = []
        contradictions = []
        clarification_needed = None
        socratic_inquiry = None
        factuality_result = None
        
        try:
            # Stage 1: Generate Socratic Inquiry Chain
            logger.debug("Generating Socratic inquiry chain")
            socratic_inquiry = await self.question_generator.generate_socratic_inquiry(
                claim, "verification", context
            )
            stage_results[VerificationStage.CLAIM_EXTRACTION] = {
                "status": "COMPLETED",
                "questions_generated": len(socratic_inquiry.questions),
                "inquiry_complexity": socratic_inquiry.complexity_level
            }
            
            # Stage 2: Cross-alignment check (if image provided)
            if image_path:
                logger.debug("Performing cross-alignment check")
                alignment_result = await self._perform_cross_alignment_check(claim, image_path)
                stage_results[VerificationStage.CROSS_ALIGNMENT] = alignment_result
                
                if alignment_result["status"] == CheckStatus.FAIL:
                    overall_status = CheckStatus.FAIL
                    confidence *= alignment_result["confidence"]
                    contradictions.extend(alignment_result.get("contradictions", []))
                    
                    if not clarification_needed:
                        clarification_needed = alignment_result.get("clarification_needed")
                else:
                    evidence.extend(alignment_result.get("evidence", []))
            
            # Stage 3: Enhanced External Factuality Check (with global knowledge detection)
            if overall_status != CheckStatus.FAIL:
                logger.debug("Performing enhanced external factuality check")
                factuality_result = await self.external_factuality_checker.verify_claim(claim, context)
                
                stage_results[VerificationStage.EXTERNAL_FACTUALITY] = {
                    "status": factuality_result.overall_status,
                    "requires_global_knowledge": factuality_result.requires_global_knowledge,
                    "knowledge_type": factuality_result.global_knowledge_type.value,
                    "questions_asked": len(factuality_result.socratic_questions),
                    "confidence": factuality_result.confidence
                }
                
                if factuality_result.overall_status in ["CONTRADICTED", "ERROR"]:
                    overall_status = CheckStatus.FAIL
                    confidence *= factuality_result.confidence
                    contradictions.extend(factuality_result.contradicting_evidence)
                    
                    if not clarification_needed and factuality_result.recommendations:
                        clarification_needed = "; ".join(factuality_result.recommendations)
                elif factuality_result.overall_status == "VERIFIED":
                    evidence.extend(factuality_result.supporting_evidence)
                elif factuality_result.overall_status == "SKIPPED":
                    # No global knowledge needed - this is fine
                    pass
            
            # Stage 4: Self-contradiction check (if previous checks passed)
            if overall_status != CheckStatus.FAIL:
                logger.debug("Performing self-contradiction check")
                contradiction_result = await self._perform_self_contradiction_check(claim)
                stage_results[VerificationStage.SELF_CONTRADICTION] = contradiction_result
                
                if contradiction_result["status"] == CheckStatus.FAIL:
                    overall_status = CheckStatus.FAIL
                    confidence *= contradiction_result["confidence"]
                    contradictions.extend(contradiction_result.get("contradictions", []))
                    
                    if not clarification_needed:
                        clarification_needed = contradiction_result.get("clarification_needed")
                else:
                    evidence.extend(contradiction_result.get("evidence", []))
            
            # Stage 5: Conditional Ambiguity Check
            ambiguity_needed = self._should_check_ambiguity(claim, overall_status, stage_results)
            if ambiguity_needed:
                logger.debug("Performing ambiguity check")
                ambiguity_result = await self._perform_ambiguity_check(claim, original_input)
                stage_results[VerificationStage.AMBIGUITY_CHECK] = ambiguity_result
                
                if ambiguity_result.get("needs_clarification"):
                    if not clarification_needed:
                        clarification_needed = ambiguity_result.get("clarification_questions")
            
            # Stage 6: Knowledge Integration (KALA methodology)
            logger.debug("Integrating knowledge using KALA methodology")
            verification_context = {
                "status": overall_status.value,
                "confidence": confidence,
                "evidence": evidence,
                "contradictions": contradictions
            }
            
            integration_result = await self.knowledge_graph.integrate_claim_knowledge(
                self.session_id, claim, verification_context
            )
            stage_results[VerificationStage.KNOWLEDGE_INTEGRATION] = integration_result
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AdvancedVerificationResult(
                claim=claim,
                stage_results=stage_results,
                overall_status=overall_status,
                confidence=confidence,
                socratic_inquiry=socratic_inquiry,
                factuality_result=factuality_result,
                evidence=evidence,
                contradictions=contradictions,
                clarification_needed=clarification_needed,
                knowledge_integrated=integration_result.get("knowledge_updated", False),
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Advanced claim verification failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AdvancedVerificationResult(
                claim=claim,
                stage_results=stage_results,
                overall_status=CheckStatus.ERROR,
                confidence=0.0,
                socratic_inquiry=socratic_inquiry,
                factuality_result=factuality_result,
                evidence=[],
                contradictions=[f"Verification error: {str(e)}"],
                clarification_needed="Technical error occurred during verification",
                knowledge_integrated=False,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    async def _perform_cross_alignment_check(self, claim: AdvancedClaim, image_path: str) -> Dict[str, Any]:
        """Perform cross-alignment check with enhanced error handling"""
        try:
            result = self.cross_alignment_checker.check_alignment(claim.text, image_path)
            return {
                "status": CheckStatus(result.get("status", "PASS")),
                "confidence": result.get("confidence", 0.8),
                "evidence": result.get("evidence", []),
                "contradictions": result.get("contradictions", []),
                "clarification_needed": result.get("clarification_needed")
            }
        except Exception as e:
            logger.error(f"Cross-alignment check failed: {e}")
            return {
                "status": CheckStatus.ERROR,
                "confidence": 0.0,
                "evidence": [],
                "contradictions": [f"Cross-alignment error: {str(e)}"],
                "clarification_needed": "Unable to verify image-text alignment"
            }
    
    async def _perform_self_contradiction_check(self, claim: AdvancedClaim) -> Dict[str, Any]:
        """Perform self-contradiction check with session context"""
        try:
            result = self.self_contradiction_checker.check_contradiction(claim.text, self.session_id)
            return {
                "status": CheckStatus(result.get("status", "PASS")),
                "confidence": result.get("confidence", 0.8),
                "evidence": result.get("evidence", []),
                "contradictions": result.get("contradictions", []),
                "clarification_needed": result.get("clarification_needed")
            }
        except Exception as e:
            logger.error(f"Self-contradiction check failed: {e}")
            return {
                "status": CheckStatus.ERROR,
                "confidence": 0.0,
                "evidence": [],
                "contradictions": [f"Self-contradiction error: {str(e)}"],
                "clarification_needed": "Unable to check for contradictions"
            }
    
    async def _perform_ambiguity_check(self, claim: AdvancedClaim, original_input: str) -> Dict[str, Any]:
        """Perform ambiguity check with enhanced logic"""
        try:
            result = self.ambiguity_checker.check_ambiguity(claim.text, original_input)
            return {
                "needs_clarification": result.get("needs_clarification", False),
                "clarification_questions": result.get("clarification_questions", ""),
                "ambiguity_score": result.get("ambiguity_score", 0.0),
                "ambiguous_terms": result.get("ambiguous_terms", [])
            }
        except Exception as e:
            logger.error(f"Ambiguity check failed: {e}")
            return {
                "needs_clarification": True,
                "clarification_questions": "Unable to assess ambiguity due to technical error",
                "ambiguity_score": 1.0,
                "ambiguous_terms": []
            }
    
    def _should_check_ambiguity(self, claim: AdvancedClaim, overall_status: CheckStatus, 
                              stage_results: Dict[VerificationStage, Dict[str, Any]]) -> bool:
        """Determine if ambiguity check is needed based on current state"""
        # Skip ambiguity check if claim already failed other checks
        if overall_status == CheckStatus.FAIL:
            return False
        
        # Check ambiguity if claim has low verifiability
        if claim.verifiability.value in ["low", "subjective"]:
            return True
        
        # Check ambiguity if external factuality was inconclusive
        factuality_stage = stage_results.get(VerificationStage.EXTERNAL_FACTUALITY, {})
        if factuality_stage.get("status") in ["INCONCLUSIVE", "INSUFFICIENT_EVIDENCE"]:
            return True
        
        # Check ambiguity if claim has complex relationships
        if len(claim.relationships) > 2:
            return True
        
        return False
    
    async def _compile_advanced_socratic_response(self, verification_results: List[AdvancedVerificationResult], 
                                                original_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compile comprehensive Socratic response with advanced insights"""
        logger.info("Compiling advanced Socratic response...")
        
        # Analyze overall verification status
        passed_claims = [r for r in verification_results if r.overall_status == CheckStatus.PASS]
        failed_claims = [r for r in verification_results if r.overall_status == CheckStatus.FAIL]
        error_claims = [r for r in verification_results if r.overall_status == CheckStatus.ERROR]
        
        # Generate advanced Socratic dialogue
        socratic_dialogue = await self._generate_advanced_socratic_dialogue(verification_results)
        
        # Compile comprehensive response
        response = {
            "session_id": self.session_id,
            "session_domain": self.session_domain,
            "timestamp": datetime.now().isoformat(),
            "original_input": original_input,
            "methodology": "Advanced Socratic with ZeroFEC + KALA + Local LLM",
            
            "verification_summary": {
                "total_claims": len(verification_results),
                "verified_claims": len(passed_claims),
                "failed_claims": len(failed_claims),
                "error_claims": len(error_claims),
                "overall_status": "PASS" if len(failed_claims) == 0 and len(error_claims) == 0 else "FAIL",
                "average_confidence": sum(r.confidence for r in verification_results) / len(verification_results) if verification_results else 0.0
            },
            
            "socratic_dialogue": socratic_dialogue,
            
            "detailed_results": [
                {
                    "claim": result.claim.text,
                    "claim_type": result.claim.claim_type.value,
                    "verifiability": result.claim.verifiability.value,
                    "status": result.overall_status.value,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "contradictions": result.contradictions,
                    "clarification_needed": result.clarification_needed,
                    "knowledge_integrated": result.knowledge_integrated,
                    "processing_time": result.processing_time,
                    "stage_results": {stage.value: data for stage, data in result.stage_results.items()},
                    "socratic_questions": [q.question for q in result.socratic_inquiry.questions] if result.socratic_inquiry else [],
                    "factuality_analysis": {
                        "requires_global_knowledge": result.factuality_result.requires_global_knowledge if result.factuality_result else False,
                        "knowledge_type": result.factuality_result.global_knowledge_type.value if result.factuality_result else "none",
                        "verification_method": result.factuality_result.verification_method.value if result.factuality_result else "none"
                    } if result.factuality_result else None
                }
                for result in verification_results
            ],
            
            "knowledge_graph_updates": {
                "session_graph_size": self.knowledge_graph.get_graph_size(self.session_id),
                "domain_summary": self.knowledge_graph.get_domain_summary(self.session_domain)
            },
            
            "next_steps": await self._generate_advanced_next_steps(verification_results),
            
            "insights": await self._generate_verification_insights(verification_results)
        }
        
        return response
    
    async def _generate_advanced_socratic_dialogue(self, verification_results: List[AdvancedVerificationResult]) -> List[Dict[str, Any]]:
        """Generate sophisticated Socratic dialogue based on verification results"""
        dialogue = []
        
        for result in verification_results:
            # Add claim introduction
            dialogue.append({
                "type": "claim_introduction",
                "content": f"Let us examine the claim: '{result.claim.text}'",
                "claim_type": result.claim.claim_type.value,
                "verifiability": result.claim.verifiability.value
            })
            
            # Add Socratic questions from inquiry chain
            if result.socratic_inquiry:
                for question in result.socratic_inquiry.questions:
                    dialogue.append({
                        "type": "socratic_question",
                        "content": question.question,
                        "question_type": question.question_type.value,
                        "reasoning": question.reasoning,
                        "depth": question.inquiry_depth.value if hasattr(question.inquiry_depth, 'value') else str(question.inquiry_depth),
                        "confidence": question.confidence
                    })
            
            # Add factuality analysis if available
            if result.factuality_result and result.factuality_result.socratic_answers:
                for answer in result.factuality_result.socratic_answers:
                    dialogue.append({
                        "type": "socratic_answer",
                        "question": answer.question,
                        "content": answer.answer,
                        "confidence": answer.confidence,
                        "sources": answer.sources
                    })
            
            # Add verification outcome
            if result.overall_status == CheckStatus.PASS:
                dialogue.append({
                    "type": "verification_result",
                    "content": f"Through systematic Socratic inquiry, the claim '{result.claim.text}' appears to be consistent with available evidence.",
                    "evidence": result.evidence,
                    "confidence": result.confidence
                })
            else:
                dialogue.append({
                    "type": "contradiction_found",
                    "content": f"Our Socratic investigation reveals concerns about the claim '{result.claim.text}'.",
                    "contradictions": result.contradictions,
                    "clarification_needed": result.clarification_needed
                })
        
        return dialogue
    
    async def _generate_advanced_next_steps(self, verification_results: List[AdvancedVerificationResult]) -> List[str]:
        """Generate sophisticated next steps based on verification results"""
        next_steps = []
        
        failed_results = [r for r in verification_results if r.overall_status == CheckStatus.FAIL]
        error_results = [r for r in verification_results if r.overall_status == CheckStatus.ERROR]
        
        if failed_results:
            next_steps.append("Review contradicted claims using Socratic methodology")
            for result in failed_results:
                if result.clarification_needed:
                    next_steps.append(f"Clarification needed: {result.clarification_needed}")
                
                # Add specific recommendations from factuality checker
                if result.factuality_result and result.factuality_result.recommendations:
                    next_steps.extend(result.factuality_result.recommendations)
        
        if error_results:
            next_steps.append("Address technical errors in verification pipeline")
        
        if len(failed_results) == 0 and len(error_results) == 0:
            next_steps.append("All claims verified - knowledge successfully integrated into graph")
            next_steps.append("Ready for deeper Socratic exploration or new inquiries")
        
        # Add knowledge graph recommendations
        graph_size = self.knowledge_graph.get_graph_size(self.session_id)
        if graph_size["nodes"] > 10:
            next_steps.append("Consider exploring relationships in the growing knowledge graph")
        
        return next_steps
    
    async def _generate_verification_insights(self, verification_results: List[AdvancedVerificationResult]) -> Dict[str, Any]:
        """Generate insights about the verification process"""
        insights = {
            "methodology_effectiveness": {},
            "knowledge_patterns": {},
            "verification_challenges": [],
            "recommendations": []
        }
        
        # Analyze methodology effectiveness
        total_processing_time = sum(r.processing_time for r in verification_results)
        avg_confidence = sum(r.confidence for r in verification_results) / len(verification_results) if verification_results else 0.0
        
        insights["methodology_effectiveness"] = {
            "average_processing_time": total_processing_time / len(verification_results) if verification_results else 0.0,
            "average_confidence": avg_confidence,
            "success_rate": len([r for r in verification_results if r.overall_status == CheckStatus.PASS]) / len(verification_results) if verification_results else 0.0
        }
        
        # Analyze knowledge patterns
        claim_types = {}
        global_knowledge_types = {}
        
        for result in verification_results:
            claim_type = result.claim.claim_type.value
            claim_types[claim_type] = claim_types.get(claim_type, 0) + 1
            
            if result.factuality_result:
                gk_type = result.factuality_result.global_knowledge_type.value
                global_knowledge_types[gk_type] = global_knowledge_types.get(gk_type, 0) + 1
        
        insights["knowledge_patterns"] = {
            "claim_types_distribution": claim_types,
            "global_knowledge_types": global_knowledge_types,
            "knowledge_integration_rate": len([r for r in verification_results if r.knowledge_integrated]) / len(verification_results) if verification_results else 0.0
        }
        
        return insights
    
    def _create_no_claims_response(self, user_input: str) -> Dict[str, Any]:
        """Create response when no claims are extracted"""
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_input": user_input,
            "verification_summary": {
                "total_claims": 0,
                "overall_status": "NO_CLAIMS"
            },
            "socratic_dialogue": [{
                "type": "no_claims_found",
                "content": "I did not identify any verifiable factual claims in your input. Could you provide more specific statements that can be examined through Socratic inquiry?"
            }],
            "next_steps": ["Provide more specific factual claims for verification"]
        }
    
    def _create_error_response(self, user_input: str, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_input": user_input,
            "verification_summary": {
                "total_claims": 0,
                "overall_status": "ERROR"
            },
            "error": error_message,
            "socratic_dialogue": [{
                "type": "error",
                "content": f"I encountered an error during verification: {error_message}"
            }],
            "next_steps": ["Please try again or contact support if the error persists"]
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        return {
            "session_id": self.session_id,
            "session_domain": self.session_domain,
            "total_inputs": len(self.conversation_history),
            "verified_claims": len(self.verified_claims),
            "knowledge_graph_size": self.knowledge_graph.get_graph_size(self.session_id),
            "domain_knowledge": self.knowledge_graph.get_domain_summary(self.session_domain),
            "methodology": "Advanced Socratic with ZeroFEC + KALA + Local LLM",
            "capabilities": [
                "Advanced claim extraction with relationships",
                "Sophisticated Socratic questioning",
                "Global knowledge detection",
                "Local LLM-powered verification",
                "KALA knowledge graph integration",
                "Multi-stage verification pipeline"
            ]
        }
    
    async def shutdown(self):
        """Shutdown the Advanced Socrates Agent"""
        logger.info("Shutting down Advanced Socrates Agent...")
        
        # Save knowledge graphs
        self.knowledge_graph.save_knowledge()
        
        # Shutdown LLM manager
        self.llm_manager.shutdown()
        
        logger.info("Advanced Socrates Agent shutdown complete")
