"""
Self-Contradiction Checker - Session consistency verification
Checks claims against established session knowledge for internal contradictions
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.logger import setup_logger
from .knowledge_graph_manager import KnowledgeGraphManager
from .llm_manager import get_llm_manager

logger = setup_logger(__name__)

@dataclass
class ContradictionResult:
    """Result of self-contradiction check"""
    status: str  # PASS, FAIL
    confidence: float
    contradictions: List[str]
    evidence: List[str]
    conflicting_claims: List[str]

class SelfContradictionChecker:
    """
    Self-contradiction checker using session knowledge graph
    Implements consistency verification against established claims
    """
    
    def __init__(self):
        logger.info("Initializing Self-Contradiction Checker...")
        # Will use the shared KG manager instance
        self.kg_manager = None
        # Initialize local LLM manager (singleton)
        try:
            self.llm = get_llm_manager()
        except Exception as e:
            logger.error(f"Failed to initialize LLM manager for contradiction detection: {e}")
            self.llm = None
        logger.info("Self-Contradiction Checker initialized")
    
    def set_kg_manager(self, kg_manager: KnowledgeGraphManager):
        """Set the knowledge graph manager instance"""
        self.kg_manager = kg_manager
    
    def check_contradiction(self, claim: str, session_id: str) -> Dict[str, Any]:
        """
        Check if claim contradicts existing session knowledge
        
        Args:
            claim: The claim to check
            session_id: Current session ID
            
        Returns:
            Dictionary containing contradiction check results
        """
        logger.info(f"Checking self-contradiction for: {claim[:50]}...")
        
        if not self.kg_manager:
            logger.warning("No KG manager available for contradiction check")
            return {
                "status": "PASS",
                "confidence": 0.5,
                "contradictions": [],
                "evidence": [],
                "conflicting_claims": []
            }
        
        try:
            # Gather existing session claims as context for LLM
            try:
                session_claims = self.kg_manager._get_session_claims(session_id)  # Returns [{text, confidence, evidence}]
                existing_texts = [c.get("text", "") for c in session_claims if c.get("text")]
            except Exception as e:
                logger.warning(f"Failed to load session claims for contradiction LLM context: {e}")
                existing_texts = []

            # If LLM is available, use it for strict JSON contradiction detection
            if self.llm is not None:
                llm_resp = self.llm.detect_contradictions_sync(
                    claim=claim,
                    existing_claims=existing_texts,
                    context={"session_id": session_id}
                )

                # Attempt to parse the structured output
                data = llm_resp.structured_output or {}
                status = str(data.get("status", "")).upper()

                if status in {"PASS", "FAIL"}:
                    contradictions = data.get("contradictions", []) or []
                    # Map contradictions to list of strings for backward compatibility
                    contradictions_as_text = []
                    for c in contradictions:
                        if isinstance(c, dict):
                            against = c.get("against") or c.get("existing_claim") or ""
                            ctype = c.get("type")
                            expl = c.get("explanation")
                            # Compose a concise textual summary for the 'contradictions' field
                            parts = [p for p in [against, ctype, expl] if p]
                            contradictions_as_text.append(" | ".join(parts) if parts else against)
                        elif isinstance(c, str):
                            contradictions_as_text.append(c)
                    conflicting_claims = data.get("conflicting_claims") or [c.get("against") for c in contradictions if isinstance(c, dict) and c.get("against")]
                    evidence = data.get("evidence", []) or []
                    confidence = float(data.get("confidence", 0.8))

                    result = {
                        "status": status,
                        "confidence": confidence,
                        "contradictions": contradictions_as_text,
                        "evidence": evidence,
                        "conflicting_claims": conflicting_claims or []
                    }
                    logger.info(f"Self-contradiction LLM check completed: {result['status']}")
                    return result
                else:
                    logger.warning("LLM contradiction response missing/invalid status; falling back to KG method")
            else:
                logger.warning("LLM manager unavailable; using KG-based contradiction check")

            # Fallback: Use KG manager's contradiction check (semantic+pattern)
            kg_result = self.kg_manager.check_contradiction(claim, session_id)

            result = {
                "status": kg_result.get("status", "PASS"),
                "confidence": 0.8 if kg_result.get("status") == "FAIL" else 0.9,
                "contradictions": [c.get("existing_claim", "") for c in kg_result.get("contradictions", [])],
                "evidence": [],
                "conflicting_claims": kg_result.get("conflicting_claims", [])
            }
            logger.info(f"Self-contradiction KG fallback completed: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in self-contradiction check: {str(e)}")
            return {
                "status": "PASS",  # Default to pass on error
                "confidence": 0.5,
                "contradictions": [],
                "evidence": [],
                "conflicting_claims": []
            }
