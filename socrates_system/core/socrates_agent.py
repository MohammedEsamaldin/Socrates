"""
Socrates Agent - The central coordinator for external hallucination detection
Implements sophisticated Socratic dialogue methodology for claim verification
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from ..modules.claim_extractor import ClaimExtractor
from ..modules.question_generator import QuestionGenerator
from ..modules.cross_alignment_checker import CrossAlignmentChecker
from ..modules.external_factuality_checker import ExternalFactualityChecker
from ..modules.self_contradiction_checker import SelfContradictionChecker
from ..modules.ambiguity_checker import AmbiguityChecker
from ..modules.clarification_handler import ClarificationHandler
from ..modules.knowledge_graph_manager import KnowledgeGraphManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    SKIP = "SKIP"

@dataclass
class SocraticInquiry:
    """Represents a Socratic inquiry with questions and reasoning"""
    question: str
    reasoning: str
    expected_answer_type: str
    confidence: float
    context: Dict[str, Any]

@dataclass
class ClaimVerificationResult:
    """Result of claim verification process"""
    claim: str
    status: CheckStatus
    confidence: float
    evidence: List[str]
    contradictions: List[str]
    socratic_questions: List[SocraticInquiry]
    clarification_needed: Optional[str]
    timestamp: datetime

class SocratesAgent:
    """
    The Socrates Agent - Central coordinator implementing Socratic methodology
    for external hallucination detection in multimodal contexts
    """
    
    def __init__(self):
        logger.info("Initializing Socrates Agent...")
        
        # Initialize all modules
        self.claim_extractor = ClaimExtractor()
        self.question_generator = QuestionGenerator()
        self.cross_alignment_checker = CrossAlignmentChecker()
        self.external_factuality_checker = ExternalFactualityChecker()
        self.self_contradiction_checker = SelfContradictionChecker()
        self.ambiguity_checker = AmbiguityChecker()
        self.clarification_handler = ClarificationHandler()
        self.kg_manager = KnowledgeGraphManager()
        
        # Session state
        self.session_id = None
        self.conversation_history = []
        self.verified_claims = []
        
        logger.info("Socrates Agent initialized successfully")
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new verification session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.conversation_history = []
        self.verified_claims = []
        
        # Initialize session knowledge graph
        self.kg_manager.initialize_session(session_id)
        
        logger.info(f"Started new session: {session_id}")
        return session_id
    
    def process_user_input(self, user_input: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main processing pipeline implementing Socratic methodology
        
        Args:
            user_input: User's text input
            image_path: Optional path to image for multimodal analysis
            
        Returns:
            Comprehensive verification results
        """
        logger.info(f"Processing user input: {user_input[:100]}...")
        
        if not self.session_id:
            self.start_session()
        
        # Store input in conversation history
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "image_path": image_path
        })
        
        try:
            # Stage 1: Claim Extraction
            logger.info("Stage 1: Claim Extraction")
            claims = self.claim_extractor.extract_claims(user_input)
            
            # Stage 2: Factuality Checks - Apply Socratic methodology
            logger.info("Stage 2: Factuality Checks")
            verification_results = []
            
            for claim in claims:
                result = self._verify_claim_socratically(claim, user_input, image_path)
                verification_results.append(result)
            
            # Stage 3: Knowledge Base Update
            logger.info("Stage 3: Knowledge Base Update")
            self._update_knowledge_base(verification_results)
            
            # Compile final response
            response = self._compile_socratic_response(verification_results, user_input)
            
            logger.info("Processing completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return {
                "status": "error",
                "message": f"An error occurred during processing: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _verify_claim_socratically(self, claim: str, original_input: str, image_path: Optional[str]) -> ClaimVerificationResult:
        """
        Apply Socratic methodology to verify a single claim
        Implements the four-stage verification process
        """
        logger.info(f"Verifying claim: {claim}")
        
        socratic_questions = []
        evidence = []
        contradictions = []
        overall_status = CheckStatus.PASS
        confidence = 1.0
        clarification_needed = None
        
        # Generate initial Socratic inquiry
        initial_inquiry = self.question_generator.generate_socratic_inquiry(claim, "verification")
        socratic_questions.append(initial_inquiry)
        
        # Check 1: Cross-alignment (if image provided)
        if image_path:
            logger.info("Performing cross-alignment check...")
            alignment_result = self.cross_alignment_checker.check_alignment(claim, image_path)
            
            if alignment_result["status"] == CheckStatus.FAIL:
                overall_status = CheckStatus.FAIL
                confidence *= alignment_result["confidence"]
                contradictions.extend(alignment_result["contradictions"])
                
                # Generate clarification inquiry
                clarification_inquiry = self.question_generator.generate_socratic_inquiry(
                    claim, "clarification", context=alignment_result
                )
                socratic_questions.append(clarification_inquiry)
                clarification_needed = self.clarification_handler.generate_clarification(
                    claim, alignment_result["visual_description"]
                )
            else:
                evidence.extend(alignment_result["evidence"])
        
        # Check 2: External factuality (if alignment passed or no image)
        # need to be changed and understand that not all the claims will be check from external factuality
        if overall_status != CheckStatus.FAIL:
            logger.info("Performing external factuality check...")
            factuality_result = self.external_factuality_checker.verify_claim(claim)
            
            if factuality_result["status"] == CheckStatus.FAIL:
                overall_status = CheckStatus.FAIL
                confidence *= factuality_result["confidence"]
                contradictions.extend(factuality_result["contradictions"])
                
                # Generate deeper Socratic inquiry
                deeper_inquiry = self.question_generator.generate_socratic_inquiry(
                    claim, "deeper_analysis", context=factuality_result
                )
                socratic_questions.append(deeper_inquiry)
                
                if not clarification_needed:
                    clarification_needed = self.clarification_handler.generate_clarification(
                        claim, factuality_result["external_facts"]
                    )
            else:
                evidence.extend(factuality_result["evidence"])
        
        # Check 3: Self-contradiction (if previous checks passed)
        if overall_status != CheckStatus.FAIL:
            logger.info("Performing self-contradiction check...")
            contradiction_result = self.self_contradiction_checker.check_contradiction(
                claim, self.session_id
            )
            
            if contradiction_result["status"] == CheckStatus.FAIL:
                overall_status = CheckStatus.FAIL
                confidence *= contradiction_result["confidence"]
                contradictions.extend(contradiction_result["contradictions"])
                
                # Generate consistency inquiry
                consistency_inquiry = self.question_generator.generate_socratic_inquiry(
                    claim, "consistency", context=contradiction_result
                )
                socratic_questions.append(consistency_inquiry)
                
                if not clarification_needed:
                    clarification_needed = self.clarification_handler.generate_clarification(
                        claim, contradiction_result["conflicting_claims"]
                    )
            else:
                evidence.extend(contradiction_result["evidence"])
        
        # Check 4: Ambiguity check
        logger.info("Performing ambiguity check...")
        ambiguity_result = self.ambiguity_checker.check_ambiguity(claim, original_input)
        
        if ambiguity_result["needs_clarification"]:
            # Generate clarification inquiry
            ambiguity_inquiry = self.question_generator.generate_socratic_inquiry(
                claim, "ambiguity", context=ambiguity_result
            )
            socratic_questions.append(ambiguity_inquiry)
            
            if not clarification_needed:
                clarification_needed = ambiguity_result["clarification_questions"]
        
        return ClaimVerificationResult(
            claim=claim,
            status=overall_status,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            socratic_questions=socratic_questions,
            clarification_needed=clarification_needed,
            timestamp=datetime.now()
        )
    
    def _update_knowledge_base(self, verification_results: List[ClaimVerificationResult]):
        """Update the knowledge graph with verified claims"""
        logger.info("Updating knowledge base...")
        
        for result in verification_results:
            if result.status == CheckStatus.PASS:
                # Add verified claim to knowledge graph
                self.kg_manager.add_claim(
                    claim=result.claim,
                    evidence=result.evidence,
                    confidence=result.confidence,
                    session_id=self.session_id
                )
                self.verified_claims.append(result)
    
    def _compile_socratic_response(self, verification_results: List[ClaimVerificationResult], 
                                 original_input: str) -> Dict[str, Any]:
        """
        Compile a comprehensive Socratic response based on verification results
        to provide a complete answer for the verification process
        """
        logger.info("Compiling Socratic response...")
        
        # Analyze overall verification status
        passed_claims = [r for r in verification_results if r.status == CheckStatus.PASS]
        failed_claims = [r for r in verification_results if r.status == CheckStatus.FAIL]
        
        # Generate Socratic dialogue summary
        socratic_dialogue = self._generate_socratic_dialogue(verification_results)
        
        # Compile response
        response = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "original_input": original_input,
            "verification_summary": {
                "total_claims": len(verification_results),
                "verified_claims": len(passed_claims),
                "failed_claims": len(failed_claims),
                "overall_status": "PASS" if len(failed_claims) == 0 else "FAIL"
            },
            "socratic_dialogue": socratic_dialogue,
            "detailed_results": [
                {
                    "claim": result.claim,
                    "status": result.status.value,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "contradictions": result.contradictions,
                    "clarification_needed": result.clarification_needed
                }
                for result in verification_results
            ],
            "next_steps": self._generate_next_steps(verification_results)
        }
        
        return response
    
    def _generate_socratic_dialogue(self, verification_results: List[ClaimVerificationResult]) -> List[Dict[str, str]]:
        """Generate a Socratic dialogue based on verification results"""
        dialogue = []
        
        for result in verification_results:
            # Add Socratic questions and reasoning
            for inquiry in result.socratic_questions:
                dialogue.append({
                    "type": "socratic_question",
                    "content": inquiry.question,
                    "reasoning": inquiry.reasoning,
                    "confidence": inquiry.confidence
                })
            
            # Add verification outcome
            if result.status == CheckStatus.PASS:
                dialogue.append({
                    "type": "verification_result",
                    "content": f"Through careful examination, the claim '{result.claim}' appears to be consistent with available evidence.",
                    "evidence": result.evidence
                })
            else:
                dialogue.append({
                    "type": "contradiction_found",
                    "content": f"Upon investigation, the claim '{result.claim}' appears to contradict available evidence.",
                    "contradictions": result.contradictions
                })
        
        return dialogue
    
    def _generate_next_steps(self, verification_results: List[ClaimVerificationResult]) -> List[str]:
        """Generate suggested next steps based on verification results"""
        next_steps = []
        
        failed_results = [r for r in verification_results if r.status == CheckStatus.FAIL]
        
        if failed_results:
            next_steps.append("Review and clarify contradicted claims")
            for result in failed_results:
                if result.clarification_needed:
                    next_steps.append(f"Clarification needed: {result.clarification_needed}")
        
        if len(failed_results) == 0:
            next_steps.append("All claims verified - ready to proceed with knowledge integration")
        
        return next_steps
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        return {
            "session_id": self.session_id,
            "total_inputs": len(self.conversation_history),
            "verified_claims": len(self.verified_claims),
            "knowledge_graph_size": self.kg_manager.get_graph_size(self.session_id)
        }
