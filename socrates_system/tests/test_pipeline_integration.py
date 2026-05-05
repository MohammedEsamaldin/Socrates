"""Integration test: SocratesPipeline with a deterministic stub LLM.

Verifies that the full extraction → categorization → routing path works
end-to-end without a live model server. The stub LLM returns hardcoded
JSON matching the format each module expects.
"""
import json
import unittest

from socrates_system.pipeline import SocratesPipeline
from socrates_system.modules.shared_structures import ClaimCategoryType


_EXTRACTION_RESPONSE = json.dumps([
    {
        "claim_text": "The Eiffel Tower is located in Berlin.",
        "confidence": 0.95,
        "entities": [
            {"text": "The Eiffel Tower", "label": "LOC"},
            {"text": "Berlin", "label": "GPE"},
        ],
        "relationships": [],
    }
])

_CATEGORIZATION_RESPONSE = json.dumps([
    {
        "categories": ["EXTERNAL_KNOWLEDGE_REQUIRED"],
        "confidence": 0.9,
        "justification": "Location claim verifiable via external sources.",
    }
])


class _StubLLM:
    """Deterministic LLM stub that returns canned JSON based on prompt content."""

    def generate_text(self, prompt: str, **kwargs) -> str:
        # Categorization prompt uniquely contains this header
        if "CLAIM TO ANALYZE" in prompt:
            return _CATEGORIZATION_RESPONSE
        return _EXTRACTION_RESPONSE

    def detect_contradiction_simple_sync(self, claim: str, session_facts: str):
        from socrates_system.modules.llm_manager import LLMResponse, LLMTaskType
        return LLMResponse(
            content='{"has_contradiction": false, "confidence": 0.1}',
            task_type=LLMTaskType.CONTRADICTION_DETECTION_SIMPLE,
            confidence=0.1,
        )


class TestPipelineIntegration(unittest.TestCase):
    """Full pipeline run with a stub LLM — no network, no model server required."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = SocratesPipeline(
            llm_manager=_StubLLM(),
            factuality_enabled=False,
            clarification_enabled=False,
            question_gen_enabled=False,
        )

    def test_run_returns_claims(self):
        results = self.pipeline.run("The Eiffel Tower is located in Berlin.")
        self.assertGreater(len(results), 0, "Pipeline should return at least one claim")

    def test_claims_have_text(self):
        results = self.pipeline.run("The Eiffel Tower is located in Berlin.")
        for claim in results:
            self.assertTrue(claim.text.strip(), "Each claim should have non-empty text")

    def test_claims_are_categorized(self):
        results = self.pipeline.run("The Eiffel Tower is located in Berlin.")
        for claim in results:
            self.assertIsNotNone(
                claim.categories,
                "Each claim should have categories assigned",
            )
            self.assertGreater(len(claim.categories), 0)

    def test_stub_llm_drives_categorization(self):
        """The stub returns EXTERNAL_KNOWLEDGE_REQUIRED; at least one claim should reflect that."""
        results = self.pipeline.run("The Eiffel Tower is located in Berlin.")
        all_category_names = {
            cat.name
            for claim in results
            for cat in (claim.categories or [])
        }
        self.assertIn(
            ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED,
            all_category_names,
            "Stub LLM should drive at least one claim to EXTERNAL_KNOWLEDGE_REQUIRED",
        )

    def test_different_stub_produces_different_categories(self):
        """Swapping the stub changes categorization — proves injection is real."""

        class _SubjectiveStub(_StubLLM):
            def generate_text(self, prompt: str, **kwargs) -> str:
                if "CLAIM TO ANALYZE" in prompt:
                    return json.dumps([{
                        "categories": ["SUBJECTIVE_OPINION"],
                        "confidence": 0.9,
                        "justification": "This is an opinion.",
                    }])
                return _EXTRACTION_RESPONSE

        pipeline2 = SocratesPipeline(
            llm_manager=_SubjectiveStub(),
            factuality_enabled=False,
            clarification_enabled=False,
            question_gen_enabled=False,
        )
        results = pipeline2.run("The Eiffel Tower is located in Berlin.")
        all_category_names = {
            cat.name
            for claim in results
            for cat in (claim.categories or [])
        }
        self.assertIn(
            ClaimCategoryType.SUBJECTIVE_OPINION,
            all_category_names,
            "Swapped stub should produce SUBJECTIVE_OPINION categories",
        )


if __name__ == "__main__":
    unittest.main()
