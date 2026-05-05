"""
VerificationPipeline — the single conflict-resolution decision tree.

All checker dispatch and short-circuit logic lives here so that every agent
variant (full, simple, future) gets identical routing behaviour.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any
import logging

from ..modules.shared_structures import VerificationMethod

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    SKIP = "SKIP"


@dataclass
class VerificationResult:
    status: CheckStatus
    confidence: float
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    clarification_needed: Optional[str] = None
    # Set when AGLA rejects the claim; callers may convert this into a Socratic inquiry
    agla_context: Optional[dict] = None


class VerificationPipeline:
    """
    Owns the single authoritative decision tree:
      route → dispatch checkers → short-circuit on FAIL → final ambiguity pass.

    All checker arguments are optional; pass None to skip that check step.
    """

    def __init__(
        self,
        cross_alignment_checker=None,
        external_factuality_checker=None,
        self_contradiction_checker=None,
        ambiguity_checker=None,
        kg_manager=None,
        agla_client=None,
        clarification_handler=None,
    ):
        self.cross_alignment_checker = cross_alignment_checker
        self.external_factuality_checker = external_factuality_checker
        self.self_contradiction_checker = self_contradiction_checker
        self.ambiguity_checker = ambiguity_checker
        self.kg_manager = kg_manager
        self.agla_client = agla_client
        self.clarification_handler = clarification_handler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        claim_text: str,
        route=None,
        session_id: Optional[str] = None,
        image_path: Optional[str] = None,
        original_input: str = "",
    ) -> VerificationResult:
        """Run the conflict-resolution decision tree for one claim."""
        status = CheckStatus.PASS
        confidence = 1.0
        evidence: List[str] = []
        contradictions: List[str] = []
        clarification_needed: Optional[str] = None
        agla_context: Optional[dict] = None

        method = route.method if route else None

        if method == VerificationMethod.CROSS_MODAL:
            if not image_path:
                status = CheckStatus.SKIP
            else:
                used_remote = False
                if self.agla_client is not None:
                    try:
                        agla_out = self.agla_client.verify(
                            image=image_path,
                            claim=claim_text,
                            return_debug=False,
                        )
                        used_remote = True
                        verdict = agla_out.get("verdict", "Uncertain")
                        if verdict == "False":
                            status = CheckStatus.FAIL
                            confidence *= 0.85
                            truth = agla_out.get("truth") or ""
                            contradictions.append(
                                f"AGLA correction: {truth}" if truth
                                else "AGLA indicates the claim is false."
                            )
                            agla_context = {"agla_verdict": verdict}
                        else:
                            evidence.append(f"AGLA verdict: {verdict}")
                    except Exception as e:
                        logger.error(f"Remote AGLA API error: {e}")
                        used_remote = False

                if not used_remote:
                    r = self._cross_alignment(claim_text, image_path)
                    status, confidence, evidence, contradictions, clarification_needed = (
                        self._merge(r, claim_text, status, confidence, evidence, contradictions, clarification_needed, "alignment")
                    )

        elif method == VerificationMethod.EXTERNAL_SOURCE:
            if image_path:
                r = self._cross_alignment(claim_text, image_path)
                status, confidence, evidence, contradictions, clarification_needed = (
                    self._merge(r, claim_text, status, confidence, evidence, contradictions, clarification_needed, "alignment")
                )
            if status != CheckStatus.FAIL:
                r = self._external_factuality(claim_text)
                status, confidence, evidence, contradictions, clarification_needed = (
                    self._merge(r, claim_text, status, confidence, evidence, contradictions, clarification_needed, "contradiction")
                )

        elif method == VerificationMethod.KNOWLEDGE_GRAPH:
            r = self._self_contradiction(claim_text, session_id)
            status, confidence, evidence, contradictions, clarification_needed = (
                self._merge(r, claim_text, status, confidence, evidence, contradictions, clarification_needed, "contradiction")
            )

        elif method == VerificationMethod.EXPERT_VERIFICATION:
            r = self._ambiguity(claim_text, original_input)
            clarification_needed = r.get("clarification_questions") if r.get("needs_clarification") else None

        else:
            # Fallback: run the full sequence, short-circuiting on FAIL
            if image_path:
                r = self._cross_alignment(claim_text, image_path)
                status, confidence, evidence, contradictions, clarification_needed = (
                    self._merge(r, claim_text, status, confidence, evidence, contradictions, clarification_needed, "alignment")
                )
            if status != CheckStatus.FAIL:
                r = self._external_factuality(claim_text)
                status, confidence, evidence, contradictions, clarification_needed = (
                    self._merge(r, claim_text, status, confidence, evidence, contradictions, clarification_needed, "contradiction")
                )
            if status != CheckStatus.FAIL:
                r = self._self_contradiction(claim_text, session_id)
                status, confidence, evidence, contradictions, clarification_needed = (
                    self._merge(r, claim_text, status, confidence, evidence, contradictions, clarification_needed, "contradiction")
                )

        # Final ambiguity pass for all non-EXPERT routes
        if method != VerificationMethod.EXPERT_VERIFICATION:
            r = self._ambiguity(claim_text, original_input)
            if r.get("needs_clarification"):
                clarification_needed = clarification_needed or r.get("clarification_questions")

        return VerificationResult(
            status=status,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            clarification_needed=clarification_needed,
            agla_context=agla_context,
        )

    def persist(self, claim_text: str, evidence: List[str], confidence: float, session_id: str) -> None:
        """Add a verified claim to the knowledge graph if kg_manager is set."""
        if self.kg_manager is not None:
            self.kg_manager.add_claim(
                claim=claim_text,
                evidence=evidence,
                confidence=confidence,
                session_id=session_id,
            )

    # ------------------------------------------------------------------
    # Private helpers — each returns the raw checker dict or a safe default
    # ------------------------------------------------------------------

    def _cross_alignment(self, claim_text: str, image_path: str) -> dict:
        if not self.cross_alignment_checker:
            return {"status": "SKIP"}
        try:
            return self.cross_alignment_checker.check_alignment(claim_text, image_path)
        except Exception as e:
            logger.error(f"Cross-alignment check error: {e}")
            return {"status": "SKIP"}

    def _external_factuality(self, claim_text: str) -> dict:
        if not self.external_factuality_checker:
            return {"status": "SKIP"}
        try:
            return self.external_factuality_checker.verify_claim(claim_text)
        except Exception as e:
            logger.error(f"External factuality check error: {e}")
            return {"status": "SKIP"}

    def _self_contradiction(self, claim_text: str, session_id: Optional[str]) -> dict:
        if not self.self_contradiction_checker:
            return {"status": "SKIP"}
        try:
            return self.self_contradiction_checker.check_contradiction(claim_text, session_id)
        except Exception as e:
            logger.error(f"Self-contradiction check error: {e}")
            return {"status": "SKIP"}

    def _ambiguity(self, claim_text: str, original_input: str) -> dict:
        if not self.ambiguity_checker:
            return {"needs_clarification": False}
        try:
            return self.ambiguity_checker.check_ambiguity(claim_text, original_input)
        except Exception as e:
            logger.error(f"Ambiguity check error: {e}")
            return {"needs_clarification": False}

    def _merge(
        self,
        checker_result: dict,
        claim_text: str,
        status: CheckStatus,
        confidence: float,
        evidence: List[str],
        contradictions: List[str],
        clarification_needed: Optional[str],
        issue_type: str,
    ):
        """Apply one checker result into the running accumulators."""
        result_status = checker_result.get("status", "SKIP")
        if result_status == "FAIL":
            status = CheckStatus.FAIL
            confidence *= checker_result.get("confidence", 1.0)
            contradictions.extend(checker_result.get("contradictions", []))
            if clarification_needed is None and self.clarification_handler is not None:
                conflicting = (
                    checker_result.get("visual_description")
                    or checker_result.get("external_facts")
                    or checker_result.get("contradictions")
                )
                try:
                    clarification_needed = self.clarification_handler.generate_clarification(
                        claim_text, conflicting, issue_type
                    )
                except Exception as e:
                    logger.error(f"Clarification generation error: {e}")
        else:
            evidence.extend(checker_result.get("evidence", []))

        return status, confidence, evidence, contradictions, clarification_needed
