"""
Deterministic Router Module

Routes claims using explicit heuristics based on:
- Claim categories (MLLM hallucination detection categories)
- LLM-emitted routing hints and vision flags
- Knowledge Graph coverage within the current session
- Session self-contradiction checks via KnowledgeGraphManager

Outputs a VerificationRoute with confidence, justification, cost/latency, and
secondary_actions/metadata to help downstream verification and mitigation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Any

from socrates_system.modules.shared_structures import (
    ExtractedClaim,
    ClaimCategoryType,
    VerificationMethod,
    VerificationRoute,
)

logger = logging.getLogger(__name__)


class DeterministicRouter:
    """Heuristic-based claim router leveraging the session Knowledge Graph.

    Args:
        available_methods: Set of methods usable in the current environment
        kg_manager: KnowledgeGraphManager instance
        session_id: Current session identifier for KG queries
    """

    # Costs/latencies aligned with CheckRouter for consistency
    VERIFICATION_METHODS = {
        VerificationMethod.CROSS_MODAL: {"cost": 0.8, "latency": 1.5},
        VerificationMethod.EXTERNAL_SOURCE: {"cost": 0.7, "latency": 2.0},
        VerificationMethod.KNOWLEDGE_GRAPH: {"cost": 0.2, "latency": 0.5},
        VerificationMethod.EXPERT_VERIFICATION: {"cost": 1.0, "latency": 86400.0},
        VerificationMethod.UNVERIFIABLE: {"cost": 0.0, "latency": 0.0},
    }

    def __init__(
        self,
        available_methods: Optional[Set[VerificationMethod]] = None,
        *,
        kg_manager: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.available_methods = available_methods or set(self.VERIFICATION_METHODS.keys())
        self.kg_manager = kg_manager
        self.session_id = session_id
        logger.info(
            "DeterministicRouter initialized with methods: %s",
            [m.name for m in self.available_methods],
        )

    def route_claim(self, claim: ExtractedClaim) -> VerificationRoute:
        if not claim or not getattr(claim, "text", "").strip():
            return self._create_route(
                VerificationMethod.UNVERIFIABLE,
                0.0,
                "Invalid or empty claim provided",
            )

        # Gather signals
        categories = {c.name for c in (claim.categories or [])}
        route_hint = (getattr(claim, "route_hint", None) or "").strip().lower()
        vision_required = bool(getattr(claim, "vision_flag", False)) or (
            ClaimCategoryType.VISUAL_GROUNDING_REQUIRED in categories
        )

        # Compute KG coverage and contradiction signal
        kg_cov = 0.0
        contradiction_status = None
        contradiction_count = 0
        try:
            kg_cov = self._kg_coverage_ratio(claim)
        except Exception as e:
            logger.debug(f"KG coverage computation failed: {e}")
        try:
            if self.kg_manager and self.session_id:
                ctr = self.kg_manager.check_contradiction(claim.text, self.session_id) or {}
                contradiction_status = ctr.get("status")
                contradiction_count = len(ctr.get("contradictions", []) or [])
        except Exception as e:
            logger.debug(f"KG contradiction check failed: {e}")

        # Unverifiable categories shortcut
        if ClaimCategoryType.SUBJECTIVE_OPINION in categories or ClaimCategoryType.PROCEDURAL_DESCRIPTIVE in categories:
            return self._create_route(
                VerificationMethod.UNVERIFIABLE,
                1.0,
                "Subjective/procedural content - not suitable for factual verification",
                metadata={
                    "kg_coverage": kg_cov,
                    "contradiction_status": contradiction_status,
                    "contradictions_count": contradiction_count,
                    "route_hint": route_hint,
                    "vision_flag": vision_required,
                },
            )

        # Vision-first routing
        if vision_required and VerificationMethod.CROSS_MODAL in self.available_methods:
            conf = 0.9 if ClaimCategoryType.VISUAL_GROUNDING_REQUIRED in categories else 0.85
            return self._create_route(
                VerificationMethod.CROSS_MODAL,
                conf,
                "Visual grounding indicated by category/flag - cross-modal verification",
                secondary_actions=[
                    {"action": "FALLBACK", "method": "EXTERNAL_SOURCE", "trigger": "fail_or_uncertain"},
                    {"action": "FALLBACK", "method": "KNOWLEDGE_GRAPH", "trigger": "ambiguous"},
                ],
                metadata={
                    "kg_coverage": kg_cov,
                    "contradiction_status": contradiction_status,
                    "contradictions_count": contradiction_count,
                    "route_hint": route_hint,
                    "vision_flag": vision_required,
                },
            )

        # KG-first routing when self-consistency is suggested or contradictions detected
        if (
            ClaimCategoryType.SELF_CONSISTENCY_REQUIRED in categories
            or contradiction_status == "FAIL"
            or (route_hint in {"kg", "knowledge_graph", "self", "consistency"})
        ) and VerificationMethod.KNOWLEDGE_GRAPH in self.available_methods:
            # Confidence boosted by coverage and contradiction evidence
            base = 0.75
            boost = min(0.25, max(0.0, kg_cov * 0.25))
            conf = min(0.95, base + boost)
            just = "Self-consistency/contradiction signal - knowledge graph verification"
            if contradiction_status == "FAIL":
                just += f"; contradictions: {contradiction_count}"
            return self._create_route(
                VerificationMethod.KNOWLEDGE_GRAPH,
                conf,
                just,
                secondary_actions=[
                    {"action": "FALLBACK", "method": "EXTERNAL_SOURCE", "trigger": "fail_or_uncertain"}
                ],
                metadata={
                    "kg_coverage": kg_cov,
                    "contradiction_status": contradiction_status,
                    "contradictions_count": contradiction_count,
                    "route_hint": route_hint,
                    "vision_flag": vision_required,
                },
            )

        # External knowledge routing when coverage is low or category suggests it
        if (
            ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED in categories
            or kg_cov < 0.3
            or route_hint in {"wiki", "external", "search", "web"}
        ) and VerificationMethod.EXTERNAL_SOURCE in self.available_methods:
            conf = max(0.6, 0.9 - kg_cov * 0.2)
            return self._create_route(
                VerificationMethod.EXTERNAL_SOURCE,
                conf,
                "External knowledge likely required (low KG coverage/category/hint)",
                secondary_actions=[
                    {"action": "FALLBACK", "method": "KNOWLEDGE_GRAPH", "trigger": "fail_or_uncertain"}
                ],
                metadata={
                    "kg_coverage": kg_cov,
                    "contradiction_status": contradiction_status,
                    "contradictions_count": contradiction_count,
                    "route_hint": route_hint,
                    "vision_flag": vision_required,
                },
            )

        # Tie-breaker: choose KG if we have moderate coverage; else external
        if kg_cov >= 0.5 and VerificationMethod.KNOWLEDGE_GRAPH in self.available_methods:
            conf = min(0.9, 0.65 + kg_cov * 0.3)
            return self._create_route(
                VerificationMethod.KNOWLEDGE_GRAPH,
                conf,
                "Moderate KG coverage suggests self-consistency verification",
                secondary_actions=[
                    {"action": "FALLBACK", "method": "EXTERNAL_SOURCE", "trigger": "fail_or_uncertain"}
                ],
                metadata={
                    "kg_coverage": kg_cov,
                    "contradiction_status": contradiction_status,
                    "contradictions_count": contradiction_count,
                    "route_hint": route_hint,
                    "vision_flag": vision_required,
                },
            )

        # Default fallback: external source
        return self._create_route(
            VerificationMethod.EXTERNAL_SOURCE,
            0.6,
            "Defaulting to external source verification",
            secondary_actions=[
                {"action": "FALLBACK", "method": "KNOWLEDGE_GRAPH", "trigger": "fail_or_uncertain"}
            ],
            metadata={
                "kg_coverage": kg_cov,
                "contradiction_status": contradiction_status,
                "contradictions_count": contradiction_count,
                "route_hint": route_hint,
                "vision_flag": vision_required,
            },
        )

    # ------------------------ helpers ------------------------

    def _kg_coverage_ratio(self, claim: ExtractedClaim) -> float:
        """Estimate how much of the claim is represented in the session KG.

        Heuristic: fraction of claim entities whose text appears in any node 'text'.
        """
        if not self.kg_manager or not self.session_id:
            return 0.0
        try:
            if self.session_id not in getattr(self.kg_manager, "session_graphs", {}):
                return 0.0
            graph = self.kg_manager.session_graphs[self.session_id]
            if not graph or graph.number_of_nodes() == 0:
                return 0.0
        except Exception:
            return 0.0

        ents = getattr(claim, "entities", []) or []
        if not ents:
            # try to extract entities via KG manager to have consistent normalization
            try:
                ex_ents, _ = self.kg_manager.extract_entities_and_relations(claim.text)
                # Normalize to just texts
                ent_texts = [getattr(e, "text", None) or getattr(e, "id", None) for e in ex_ents if e]
            except Exception:
                ent_texts = []
        else:
            ent_texts = [getattr(e, "text", None) for e in ents if e]

        if not ent_texts:
            return 0.0

        node_texts = []
        try:
            for _, data in graph.nodes(data=True):
                node_texts.append((data.get("text") or "").lower())
        except Exception:
            return 0.0

        hits = 0
        for t in ent_texts:
            if not t:
                continue
            tl = str(t).lower()
            if any(tl in nt for nt in node_texts):
                hits += 1
        return hits / max(1, len(ent_texts))

    def _create_route(
        self,
        method: VerificationMethod,
        confidence: float,
        justification: str,
        *,
        secondary_actions: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationRoute:
        info = self.VERIFICATION_METHODS.get(method, {})
        route = VerificationRoute(
            method=method,
            confidence=confidence,
            justification=justification,
            estimated_cost=info.get("cost", 1.0),
            estimated_latency=info.get("latency", 60.0),
        )
        if secondary_actions:
            route.secondary_actions = secondary_actions
        if metadata:
            route.metadata = metadata
        return route
