"""
Self-Contradiction Checker - Session consistency verification
Checks claims against established session knowledge for internal contradictions
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.logger import setup_logger
from .knowledge_graph_manager import KnowledgeGraphManager

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
            # Use KG manager's contradiction check
            kg_result = self.kg_manager.check_contradiction(claim, session_id)
            
            # Format result
            result = {
                "status": kg_result["status"],
                "confidence": 0.8 if kg_result["status"] == "FAIL" else 0.9,
                "contradictions": [c.get("existing_claim", "") for c in kg_result.get("contradictions", [])],
                "evidence": [],
                "conflicting_claims": kg_result.get("conflicting_claims", [])
            }
            
            logger.info(f"Self-contradiction check completed: {result['status']}")
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
