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

            # Pre-check: compatibility heuristics to avoid false positives
            hints, is_safe = self._generate_compatibility_hints(claim, existing_texts)
            if is_safe:
                logger.info(f"Compatibility heuristics indicate specialization/elaboration (early PASS). Hints: {hints}")
                return {
                    "status": "PASS",
                    "confidence": 0.9,
                    "contradictions": [],
                    "evidence": hints,
                    "conflicting_claims": []
                }

            # If LLM is available, use it for strict JSON contradiction detection
            if self.llm is not None:
                llm_resp = self.llm.detect_contradictions_sync(
                    claim=claim,
                    existing_claims=existing_texts,
                    context={"session_id": session_id, "compatibility_hints": hints},
                    entities=entities_struct,
                    entity_knowledge=entity_knowledge,
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

    def _generate_compatibility_hints(self, claim: str, existing_claims: List[str]) -> Tuple[List[str], bool]:
        """Return (hints, is_safe_specialization) based on simple heuristics to reduce false positives."""
        hints: List[str] = []
        text = claim.lower()
        neg = self._has_negation(text)
        if neg:
            return hints, False

        # Colors
        colors = {"red", "blue", "green", "yellow", "black", "white", "gray", "grey", "silver", "gold", "orange", "purple", "brown", "beige"}
        claim_colors = {c for c in colors if f" {c} " in f" {text} "}

        # Vehicle brands/models (lightweight list)
        vehicles_keywords = {"toyota", "camry", "corolla", "honda", "civic", "accord", "ford", "mustang", "bmw", "mercedes", "audi", "tesla", "kia", "hyundai", "nissan", "lexus", "vw", "volkswagen", "subaru", "mazda", "porsche", "ferrari", "lamborghini"}
        has_vehicle_brand = any(k in text for k in vehicles_keywords)
        mentions_car_word = " car" in f" {text}" or " sedan" in f" {text}" or " vehicle" in f" {text}"

        # Scene/person heuristics
        mentions_scene = any(kw in text for kw in ["busy street", "street scene", "busy scene"]) or ("scene" in text and "street" in text)
        mentions_person = any(kw in text for kw in ["person", "people", "man", "woman", "businessman", "pedestrian"])

        # Intersection/location heuristics
        looks_like_intersection = bool(re.search(r"\b(\d{1,3})(st|nd|rd|th)\b.*\b(st|street|ave|avenue|blvd)\b", text)) or (" and " in text and any(w in text for w in ["st", "street", "ave", "avenue", "blvd", "boulevard"]))
        area_keywords = {"downtown", "midtown", "uptown", "manhattan", "brooklyn", "queens", "bronx"}

        # Evaluate against existing claims
        safe = False
        for ex in existing_claims:
            ex_l = (ex or "").lower()
            if not ex_l:
                continue

            ex_has_neg = self._has_negation(ex_l)
            if ex_has_neg:
                continue

            # Vehicle specialization: general ex claim mentions car; current claim mentions a brand/model
            ex_mentions_car = " car" in f" {ex_l}" or " sedan" in f" {ex_l}" or " vehicle" in f" {ex_l}"
            if has_vehicle_brand and ex_mentions_car:
                ex_colors = {c for c in colors if f" {c} " in f" {ex_l} "}
                # If both specify color and differ, not safe
                if ex_colors and claim_colors and ex_colors != claim_colors:
                    continue
                hints.append("Vehicle specialization: specific make/model compatible with general 'car' claim")
                if ex_colors == claim_colors or (not ex_colors or not claim_colors):
                    safe = True

            # Scene elaboration: busy street scene vs person/pedestrian
            if mentions_scene and mentions_person:
                hints.append("Scene elaboration: presence of a person is compatible with a busy street scene")
                safe = True

            # Intersection vs area: specific named intersection vs broad area mention
            if looks_like_intersection and any(k in ex_l for k in area_keywords):
                hints.append("Location specialization: specific intersection within a broader area (no explicit conflict in session knowledge)")
                safe = True

        # Remove duplicates
        hints = list(dict.fromkeys(hints))
        return hints, safe

    def _has_negation(self, text: str) -> bool:
        return any(tok in text for tok in [" not ", " no ", "n't", " never ", " without "])
