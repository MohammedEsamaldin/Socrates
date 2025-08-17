"""
Conflict Resolver - Evidence-weighted aggregation of verification results
Combines external factuality and self-consistency (KG) checks into a single decision.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def _status_to_score(status: Optional[str]) -> float:
    if not status:
        return 0.0
    s = str(status).upper()
    if s == "PASS":
        return 1.0
    if s == "FAIL":
        return -1.0
    # UNCERTAIN or ERROR
    return 0.0


def _safe_conf(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class ConflictResolverConfig:
    weight_external: float = 0.6
    weight_self: float = 0.4
    add_to_kg_threshold: float = 0.7  # final confidence threshold to add claim to KG


class ConflictResolver:
    """Aggregates verification signals into a final verdict.

    Inputs:
      - external_result: dict from ExternalFactualityChecker.verify_claim
         {status: PASS/FAIL/UNCERTAIN/ERROR, confidence: float, evidence: [..], sources: [..], reasoning: str}
      - self_result: dict from SelfContradictionChecker.check_contradiction
         {status: PASS/FAIL, confidence: float, contradictions: [..], evidence: [..], conflicting_claims: [..]}

    Output:
      dict with fields: {status, confidence, reasoning, sources, contradictions, evidence, should_add_to_kg}
    """

    def __init__(self, config: Optional[ConflictResolverConfig] = None):
        self.config = config or ConflictResolverConfig()

    def resolve(self, claim: str, external_result: Optional[Dict[str, Any]], self_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            ext_status = (external_result or {}).get("status")
            ext_conf = _safe_conf((external_result or {}).get("confidence", 0.0))
            self_status = (self_result or {}).get("status")
            self_conf = _safe_conf((self_result or {}).get("confidence", 0.0))

            # Presence flags for dynamic weighting
            ext_present = external_result is not None
            self_present = self_result is not None

            # Weighted score of outcomes with dynamic normalization
            ext_score = _status_to_score(ext_status)
            self_score = _status_to_score(self_status)
            if ext_present and self_present:
                w_ext = self.config.weight_external
                w_self = self.config.weight_self
            elif ext_present:
                w_ext = 1.0
                w_self = 0.0
            elif self_present:
                w_ext = 0.0
                w_self = 1.0
            else:
                w_ext = 0.0
                w_self = 0.0
            combined_score = (w_ext * ext_score) + (w_self * self_score)

            # Map combined score to final status
            if combined_score >= 0.4:
                final_status = "PASS"
            elif combined_score <= -0.4:
                final_status = "FAIL"
            else:
                final_status = "UNCERTAIN"

            # Confidence aggregation
            if (w_ext + w_self) > 0:
                base_conf = (w_ext * ext_conf) + (w_self * self_conf)
            else:
                base_conf = 0.1
            # Penalize if the signals disagree (e.g., PASS vs FAIL)
            disagree = (ext_present and self_present) and ((ext_score * self_score) < 0)
            if disagree:
                base_conf *= 0.85
            final_conf = max(0.0, min(1.0, base_conf))

            # Build reasoning and aggregate evidence/sources/contradictions
            reasons: List[str] = []
            if ext_status:
                reasons.append(f"External factuality: {ext_status} (conf {ext_conf:.2f})")
            if self_status:
                reasons.append(f"Self-consistency: {self_status} (conf {self_conf:.2f})")
            contradictions = (self_result or {}).get("contradictions", []) or []
            if contradictions and self_status == "FAIL":
                reasons.append(f"Detected {len(contradictions)} contradiction(s) against session knowledge.")
            if final_status == "UNCERTAIN" and not reasons:
                reasons.append("Insufficient evidence across checks.")

            evidence = []
            ev1 = (external_result or {}).get("evidence") or (external_result or {}).get("external_facts") or []
            ev2 = (self_result or {}).get("evidence", []) or []
            for e in ev1:
                if e not in evidence:
                    evidence.append(e)
            for e in ev2:
                if e not in evidence:
                    evidence.append(e)

            sources = (external_result or {}).get("sources", []) or []

            result = {
                "status": final_status,
                "confidence": final_conf,
                "reasoning": " ".join(reasons).strip(),
                "sources": sources,
                "contradictions": contradictions,
                "evidence": evidence,
                "should_add_to_kg": (final_status == "PASS" and final_conf >= self.config.add_to_kg_threshold and not contradictions),
            }
            logger.info(f"Conflict resolution: {result['status']} (conf {result['confidence']:.2f})")
            return result
        except Exception as e:
            logger.error(f"ConflictResolver error: {e}")
            return {
                "status": "UNCERTAIN",
                "confidence": 0.0,
                "reasoning": f"Exception in conflict resolution: {e}",
                "sources": [],
                "contradictions": [],
                "evidence": [],
                "should_add_to_kg": False,
            }
