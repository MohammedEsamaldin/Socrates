"""
Socrates Agent - The central coordinator for external hallucination detection
Implements sophisticated Socratic dialogue methodology for claim verification
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime

from ..modules.claim_extractor import ClaimExtractor
from ..modules.question_generator import (
    SocraticQuestionGenerator,
    VerificationCapabilities,
    SocraticConfig,
)
from ..modules.claim_categorizer import ClaimCategorizer
from ..modules.check_router import CheckRouter
from ..modules.shared_structures import (
    ExtractedClaim,
    ClaimCategoryType,
)
from ..modules.llm_manager import get_llm_manager
from ..modules.cross_alignment_checker import CrossAlignmentChecker
from ..modules.external_factuality_checker import ExternalFactualityChecker
from ..modules.self_contradiction_checker import SelfContradictionChecker
from ..modules.ambiguity_checker import AmbiguityChecker
from ..modules.clarification_handler import ClarificationHandler
from ..modules.knowledge_graph_manager import KnowledgeGraphManager
from ..modules.agla_client import AGLAClient
from ..config import AGLA_API_URL, AGLA_API_VERIFY_PATH, AGLA_API_TIMEOUT
from ..utils.logger import setup_logger
from .verification_pipeline import CheckStatus, VerificationPipeline

logger = setup_logger(__name__)

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

        # LLM and categorization/routing
        self.llm_manager = get_llm_manager()
        self.claim_categorizer = ClaimCategorizer(llm_manager=self.llm_manager)
        self.check_router = CheckRouter()

        # Capability-aware Socratic question generator
        self.verification_capabilities = VerificationCapabilities(
            visual_grounding=[
                "object_detection",
                "text_recognition",
                "spatial_relationships",
            ],
            external_knowledge=[
                "wikipedia_api",
                "wikidata_api",
                "google_fact_check_api",
            ],
            self_consistency=[
                "session_knowledge_graph",
                "prior_verified_claims",
            ],
        )
        self.socratic_generator = SocraticQuestionGenerator(
            verification_capabilities=self.verification_capabilities,
            config=SocraticConfig(),
        )
        self.cross_alignment_checker = CrossAlignmentChecker()
        self.external_factuality_checker = ExternalFactualityChecker()
        self.self_contradiction_checker = SelfContradictionChecker()
        self.ambiguity_checker = AmbiguityChecker()
        self.clarification_handler = ClarificationHandler()
        self.kg_manager = KnowledgeGraphManager()
        # Remote AGLA client (remote-only)
        self.agla_client = None
        if AGLA_API_URL:
            try:
                self.agla_client = AGLAClient(AGLA_API_URL, AGLA_API_VERIFY_PATH, AGLA_API_TIMEOUT)
                logger.info(f"Configured remote AGLA API: {AGLA_API_URL}{AGLA_API_VERIFY_PATH}")
            except Exception as e:
                logger.warning(f"Failed to configure AGLAClient: {e}")
                self.agla_client = None

        self.pipeline = VerificationPipeline(
            cross_alignment_checker=self.cross_alignment_checker,
            external_factuality_checker=self.external_factuality_checker,
            self_contradiction_checker=self.self_contradiction_checker,
            ambiguity_checker=self.ambiguity_checker,
            kg_manager=self.kg_manager,
            agla_client=self.agla_client,
            clarification_handler=self.clarification_handler,
        )
        
        # Session state
        self.session_id = None
        self.conversation_history = []
        self.verified_claims = []
        
        logger.info("Socrates Agent initialized successfully")
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new verification session.

        Resets conversation history and verified claims, then initialises the
        session knowledge graph for the given session ID.

        Args:
            session_id: Optional explicit session identifier. If ``None``, a
                timestamp-based ID is generated automatically.

        Returns:
            The session ID that was created or provided.
        """
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
            
            # Stage 1.5: Claim Categorization and Routing
            logger.info("Stage 1.5: Claim Categorization and Routing")
            categorized_claims: List[ExtractedClaim] = []
            for c in claims:
                try:
                    c = self.claim_categorizer.categorize_claim(c)
                except Exception as e:
                    logger.error(f"Claim categorization failed: {e}")
                try:
                    route = self.check_router.route_claim(c)
                    c.verification_route = route
                except Exception as e:
                    logger.error(f"Claim routing failed: {e}")
                categorized_claims.append(c)

            # Stage 2: Factuality Checks - Apply Socratic methodology
            logger.info("Stage 2: Factuality Checks")
            verification_results = []
            
            for claim in categorized_claims:
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
    
    def _verify_claim_socratically(self, claim: ExtractedClaim, original_input: str, image_path: Optional[str]) -> ClaimVerificationResult:
        claim_text = claim.text if isinstance(claim, ExtractedClaim) else str(claim)
        logger.info(f"Verifying claim: {claim_text}")

        categories = [cat.name for cat in (claim.categories or [])]
        route = getattr(claim, "verification_route", None)

        if any(cat in {ClaimCategoryType.SUBJECTIVE_OPINION, ClaimCategoryType.PROCEDURAL_DESCRIPTIVE} for cat in categories):
            return ClaimVerificationResult(
                claim=claim_text,
                status=CheckStatus.SKIP,
                confidence=getattr(route, "confidence", 1.0),
                evidence=[],
                contradictions=[],
                socratic_questions=[],
                clarification_needed=None,
                timestamp=datetime.now(),
            )

        socratic_questions = []
        try:
            gen_results = self.socratic_generator.handle_multi_category_claims(
                claim_text, categories, num_questions_per_category=1,
            ) if categories else {}
            socratic_questions.extend(self._map_socratic_questions(gen_results))
        except Exception as e:
            logger.error(f"Error generating Socratic questions: {e}")

        result = self.pipeline.run(
            claim_text,
            route=route,
            session_id=self.session_id,
            image_path=image_path,
            original_input=original_input,
        )

        if result.agla_context:
            socratic_questions.append(self._fallback_clarification_inquiry(claim_text, result.agla_context))

        return ClaimVerificationResult(
            claim=claim_text,
            status=result.status,
            confidence=result.confidence,
            evidence=result.evidence,
            contradictions=result.contradictions,
            socratic_questions=socratic_questions,
            clarification_needed=result.clarification_needed,
            timestamp=datetime.now(),
        )

    def _map_socratic_questions(self, generated: Dict[str, List[Any]]) -> List[SocraticInquiry]:
        """Convert category-aware SocraticQuestion objects to SocraticInquiry for dialogue."""
        mapped: List[SocraticInquiry] = []
        if not generated:
            return mapped
        category_to_answer_type = {
            "VISUAL_GROUNDING_REQUIRED": "visual_evidence",
            "EXTERNAL_KNOWLEDGE_REQUIRED": "external_evidence",
            "SELF_CONSISTENCY_REQUIRED": "consistency_check",
            "AMBIGUOUS_RESOLUTION_REQUIRED": "clarification",
        }
        for category, questions in generated.items():
            for q in questions:
                try:
                    question_text = getattr(q, "question", str(q))
                    confidence_score = float(getattr(q, "confidence_score", 0.7))
                    verification_hint = getattr(q, "verification_hint", "")
                    mapped.append(
                        SocraticInquiry(
                            question=question_text,
                            reasoning=f"Auto-generated for {category}. {verification_hint}",
                            expected_answer_type=category_to_answer_type.get(category, "general_response"),
                            confidence=confidence_score,
                            context={"category": category, "verification_hint": verification_hint},
                        )
                    )
                except Exception:
                    continue
        return mapped

    def _fallback_clarification_inquiry(self, claim_text: str, context: Dict[str, Any]) -> SocraticInquiry:
        """Create a basic clarification inquiry when deeper generation isn't available."""
        return SocraticInquiry(
            question=f"Could you clarify the aspect that conflicts with the image regarding: '{claim_text}'?",
            reasoning="Clarify mismatch between textual claim and visual evidence.",
            expected_answer_type="clarification_and_context",
            confidence=0.6,
            context=context,
        )
    
    def _update_knowledge_base(self, verification_results: List[ClaimVerificationResult]):
        logger.info("Updating knowledge base...")
        for result in verification_results:
            if result.status == CheckStatus.PASS:
                self.pipeline.persist(result.claim, result.evidence, result.confidence, self.session_id)
                self.verified_claims.append(result)
    
    def _compile_socratic_response(self, verification_results: List[ClaimVerificationResult],
                                 original_input: str) -> Dict[str, Any]:
        """Compile a comprehensive Socratic response from per-claim verification results.

        Aggregates individual claim statuses, generates the Socratic dialogue
        transcript, and produces a structured dict suitable for JSON serialisation
        and return to the caller.

        Args:
            verification_results: List of :class:`ClaimVerificationResult` objects
                from the current processing turn.
            original_input: The original user input string included verbatim in
                the response.

        Returns:
            A dict with keys: ``session_id``, ``timestamp``, ``original_input``,
            ``verification_summary``, ``socratic_dialogue``, ``detailed_results``,
            and ``next_steps``.
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
        """Build a Socratic dialogue transcript from per-claim verification results.

        For each claim, appends one entry per Socratic question followed by a
        ``verification_result`` or ``contradiction_found`` entry.

        Args:
            verification_results: List of :class:`ClaimVerificationResult` objects.

        Returns:
            List of dialogue-entry dicts, each with at minimum ``"type"`` and
            ``"content"`` keys.
        """
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
        """Generate a list of suggested next-step action strings.

        If all claims passed, returns a single "all verified" message. If any
        failed, lists clarification requests for each failed claim.

        Args:
            verification_results: List of :class:`ClaimVerificationResult` objects.

        Returns:
            List of human-readable next-step strings.
        """
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
        """Return a summary dict of the current session state.

        Returns:
            Dict with keys ``session_id``, ``total_claims_verified``,
            ``verified_claims`` (count of PASS results), ``timestamp``, and
            ``knowledge_graph_size`` (number of nodes in the session KG).
        """
        return {
            "session_id": self.session_id,
            "total_inputs": len(self.conversation_history),
            "verified_claims": len(self.verified_claims),
            "knowledge_graph_size": self.kg_manager.get_graph_size(self.session_id)
        }
