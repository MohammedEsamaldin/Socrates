"""
Tests for VerificationPipeline — proves the decision tree is structural.

Swapping checker stubs changes behaviour; no live modules are instantiated.
"""
import pytest
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from socrates_system.core.verification_pipeline import CheckStatus, VerificationPipeline
from socrates_system.modules.shared_structures import VerificationMethod, VerificationRoute


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------

def _route(method: VerificationMethod) -> VerificationRoute:
    return VerificationRoute(method=method, confidence=0.9, justification="test",
                             estimated_cost=0.0, estimated_latency=0.0)

def _alignment(status: str, confidence=0.8, contradictions=None, evidence=None):
    m = MagicMock()
    m.check_alignment.return_value = {
        "status": status,
        "confidence": confidence,
        "contradictions": contradictions or [],
        "evidence": evidence or ["visual match"],
        "visual_description": "test image",
    }
    return m

def _factuality(status: str, confidence=0.9, contradictions=None, evidence=None):
    m = MagicMock()
    m.verify_claim.return_value = {
        "status": status,
        "confidence": confidence,
        "contradictions": contradictions or [],
        "evidence": evidence or ["source A"],
        "external_facts": "some facts",
    }
    return m

def _contradiction(status: str, confidence=0.85, contradictions=None, evidence=None):
    m = MagicMock()
    m.check_contradiction.return_value = {
        "status": status,
        "confidence": confidence,
        "contradictions": contradictions or [],
        "evidence": evidence or [],
    }
    return m

def _no_ambiguity():
    m = MagicMock()
    m.check_ambiguity.return_value = {"needs_clarification": False}
    return m


# ---------------------------------------------------------------------------
# CROSS_MODAL route
# ---------------------------------------------------------------------------

class TestCrossModalRoute:
    def test_skip_when_no_image(self):
        pipeline = VerificationPipeline(
            cross_alignment_checker=_alignment("PASS"),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.CROSS_MODAL), image_path=None)
        assert result.status == CheckStatus.SKIP

    def test_pass_when_alignment_passes(self):
        pipeline = VerificationPipeline(
            cross_alignment_checker=_alignment("PASS"),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.CROSS_MODAL), image_path="/img.jpg")
        assert result.status == CheckStatus.PASS
        assert result.evidence  # has visual evidence

    def test_fail_when_alignment_fails(self):
        pipeline = VerificationPipeline(
            cross_alignment_checker=_alignment("FAIL", contradictions=["mismatch"]),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.CROSS_MODAL), image_path="/img.jpg")
        assert result.status == CheckStatus.FAIL
        assert "mismatch" in result.contradictions


# ---------------------------------------------------------------------------
# EXTERNAL_SOURCE route
# ---------------------------------------------------------------------------

class TestExternalSourceRoute:
    def test_pass_with_passing_factuality(self):
        pipeline = VerificationPipeline(
            cross_alignment_checker=_alignment("PASS"),
            external_factuality_checker=_factuality("PASS"),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.EXTERNAL_SOURCE))
        assert result.status == CheckStatus.PASS

    def test_alignment_fail_skips_factuality_check(self):
        factuality = _factuality("PASS")
        pipeline = VerificationPipeline(
            cross_alignment_checker=_alignment("FAIL", contradictions=["bad"]),
            external_factuality_checker=factuality,
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.EXTERNAL_SOURCE), image_path="/img.jpg")
        assert result.status == CheckStatus.FAIL
        factuality.verify_claim.assert_not_called()

    def test_fail_on_factuality_when_alignment_passes(self):
        pipeline = VerificationPipeline(
            external_factuality_checker=_factuality("FAIL", contradictions=["wrong fact"]),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.EXTERNAL_SOURCE))
        assert result.status == CheckStatus.FAIL
        assert "wrong fact" in result.contradictions


# ---------------------------------------------------------------------------
# KNOWLEDGE_GRAPH route
# ---------------------------------------------------------------------------

class TestKnowledgeGraphRoute:
    def test_fail_on_contradiction(self):
        pipeline = VerificationPipeline(
            self_contradiction_checker=_contradiction("FAIL", contradictions=["prior claim"]),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.KNOWLEDGE_GRAPH), session_id="s1")
        assert result.status == CheckStatus.FAIL
        assert "prior claim" in result.contradictions

    def test_pass_when_no_contradiction(self):
        pipeline = VerificationPipeline(
            self_contradiction_checker=_contradiction("PASS"),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=_route(VerificationMethod.KNOWLEDGE_GRAPH), session_id="s1")
        assert result.status == CheckStatus.PASS


# ---------------------------------------------------------------------------
# Fallback (no route)
# ---------------------------------------------------------------------------

class TestFallbackSequence:
    def test_all_pass(self):
        pipeline = VerificationPipeline(
            cross_alignment_checker=_alignment("PASS"),
            external_factuality_checker=_factuality("PASS"),
            self_contradiction_checker=_contradiction("PASS"),
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=None, image_path="/img.jpg")
        assert result.status == CheckStatus.PASS

    def test_alignment_fail_short_circuits_remaining_checks(self):
        factuality = _factuality("PASS")
        contradiction = _contradiction("PASS")
        pipeline = VerificationPipeline(
            cross_alignment_checker=_alignment("FAIL", contradictions=["x"]),
            external_factuality_checker=factuality,
            self_contradiction_checker=contradiction,
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=None, image_path="/img.jpg")
        assert result.status == CheckStatus.FAIL
        factuality.verify_claim.assert_not_called()
        contradiction.check_contradiction.assert_not_called()

    def test_factuality_fail_skips_contradiction_check(self):
        contradiction = _contradiction("PASS")
        pipeline = VerificationPipeline(
            external_factuality_checker=_factuality("FAIL", contradictions=["y"]),
            self_contradiction_checker=contradiction,
            ambiguity_checker=_no_ambiguity(),
        )
        result = pipeline.run("claim", route=None)
        assert result.status == CheckStatus.FAIL
        contradiction.check_contradiction.assert_not_called()


# ---------------------------------------------------------------------------
# persist
# ---------------------------------------------------------------------------

class TestPersist:
    def test_persist_calls_kg_manager(self):
        kg = MagicMock()
        pipeline = VerificationPipeline(kg_manager=kg)
        pipeline.persist("the claim", ["ev1"], 0.9, "session-1")
        kg.add_claim.assert_called_once_with(
            claim="the claim", evidence=["ev1"], confidence=0.9, session_id="session-1"
        )

    def test_persist_noop_without_kg_manager(self):
        pipeline = VerificationPipeline()
        pipeline.persist("the claim", [], 1.0, "s")  # must not raise
