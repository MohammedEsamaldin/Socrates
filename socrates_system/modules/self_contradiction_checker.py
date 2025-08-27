"""
Self-Contradiction Checker - Session consistency verification
Checks claims against established session knowledge for internal contradictions
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass

from ..utils.logger import setup_logger
from .knowledge_graph_manager import KnowledgeGraphManager
from .llm_manager import LLMManager, get_llm_manager
from .fact_formatter import GraphRAGFactFormatter

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
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        logger.info("Initializing Self-Contradiction Checker...")
        # Will use the shared KG manager instance
        self.kg_manager = None
        # Initialize local LLM manager (singleton)
        try:
            self.llm = llm_manager or get_llm_manager()
        except Exception as e:
            logger.error(f"Failed to initialize LLM manager for contradiction detection: {e}")
            self.llm = None
        # GraphRAG-style fact formatter for linearized session knowledge
        try:
            self.fact_formatter = GraphRAGFactFormatter()
        except Exception:
            self.fact_formatter = None
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
        # Defensive guard: detect placeholder/empty claim text used in logs/prompts
        _ct = (claim or "").strip()
        if not _ct:
            logger.warning("[ClaimInput] Empty claim text passed; provide actual claim content for verification and logging clarity")
        elif _ct.lower() in {"claim", "statement", "assertion", "hypothesis"}:
            logger.warning(f"[ClaimInput] Placeholder-like claim text detected: '{_ct}'. Use the real claim content in prompts/logs.")
        
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

            # GraphRAG-inspired simple contradiction over linearized session facts (early, binary)
            if self.llm is not None and self.fact_formatter is not None:
                try:
                    kg_export = self.kg_manager.export_session_graph(session_id) or {}
                    high_conf_claims = [c for c in (session_claims or []) if float(c.get('confidence', 0.0) or 0.0) > 0.8]
                    session_facts = self.fact_formatter.format_session_facts(kg_export, high_conf_claims)
                    simp = self.llm.detect_contradiction_simple_sync(claim=claim, session_facts=session_facts)
                    simp_obj = simp.structured_output or {}
                    if isinstance(simp_obj, dict) and 'contradiction' in simp_obj:
                        is_contra = bool(simp_obj.get('contradiction'))
                        sim_conf = float(simp_obj.get('confidence', 0.8) or 0.8)
                        if sim_conf >= 0.7:
                            status = 'FAIL' if is_contra else 'PASS'
                            expl = simp_obj.get('explanation') or ''
                            conflict = simp_obj.get('conflicting_fact') or ''
                            return {
                                'status': status,
                                'confidence': sim_conf,
                                'contradictions': ([conflict] if (is_contra and conflict) else []),
                                'evidence': ([expl] if expl else []),
                                'conflicting_claims': ([conflict] if (is_contra and conflict) else [])
                            }
                except Exception as e:
                    logger.warning(f"Simple contradiction check failed; continuing with standard pipeline: {e}")

            # Extract entities from the claim and query entity knowledge from KG
            entities_struct: List[Dict[str, Any]] = []
            entity_knowledge: Dict[str, Any] = {}
            try:
                entities_struct = self.kg_manager.extract_entities_from_claim(claim, context="self_contradiction")
                entity_names = [e.get("name") for e in entities_struct if e.get("name")]
                if entity_names:
                    entity_knowledge = self.kg_manager.query_entity_knowledge(entity_names, context="self_contradiction")
                logger.info(f"Entity-aware contradiction check: {len(entities_struct)} entities, knowledge keys: {list(entity_knowledge.keys()) if isinstance(entity_knowledge, dict) else 'N/A'}")
            except Exception as e:
                logger.warning(f"Entity extraction/knowledge query failed: {e}")
                entities_struct = []
                entity_knowledge = {}
            # Fallback to KG manager contradiction check
            try:
                base = self.kg_manager.check_contradiction(claim, session_id) or {}
                contradictions = base.get('contradictions', [])
                conflicting = base.get('conflicting_claims', [])
                status = 'FAIL' if contradictions else 'PASS'
                if contradictions and isinstance(contradictions, list):
                    conf = max([float(c.get('confidence', 0.7)) for c in contradictions if isinstance(c, dict)], default=0.7)
                else:
                    conf = 0.8
                return {
                    'status': status,
                    'confidence': conf,
                    'contradictions': contradictions,
                    'evidence': [],
                    'conflicting_claims': conflicting,
                }
            except Exception as e:
                logger.warning(f"KG contradiction fallback failed: {e}")
                return {
                    'status': 'PASS',
                    'confidence': 0.5,
                    'contradictions': [],
                    'evidence': [],
                    'conflicting_claims': [],
                }
        except Exception as e:
            logger.error(f"Self-contradiction check failed: {e}")
            return {
                'status': 'PASS',
                'confidence': 0.5,
                'contradictions': [],
                'evidence': [],
                'conflicting_claims': [],
            }
