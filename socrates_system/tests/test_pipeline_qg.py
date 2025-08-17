import os
import types
import unittest
from unittest import mock

from socrates_system.pipeline import SocratesPipeline
from socrates_system.modules.shared_structures import ExtractedClaim, ClaimCategory, ClaimCategoryType


def make_dummy_claim():
    return ExtractedClaim(
        text="Paris is the capital of France.",
        start_char=0,
        end_char=31,
        confidence=0.99,
        source_text="Paris is the capital of France.",
        entities=[],
        relationships=[],
    )


def _set_categories(claim, names):
    claim.categories = [
        ClaimCategory(name=getattr(ClaimCategoryType, n), confidence=0.9, justification="test")
        for n in names
    ]
    return claim


class TestPipelineQG(unittest.TestCase):
    def test_pipeline_qg_cli_overrides_env_and_fallback_metrics(self):
        # Env suggests 4, CLI will pass 2 -> expect 2 per category
        with mock.patch.dict(os.environ, {"QG_QUESTIONS_PER_CATEGORY": "4"}, clear=False):
            lm = types.SimpleNamespace(generate_text=lambda *args, **kwargs: "1) Q")
            p = SocratesPipeline(
                llm_manager=lm,
                factuality_enabled=False,
                clarification_enabled=False,
                question_gen_enabled=True,
                questions_per_category=2,  # CLI override should win over env
                qg_min_threshold=0.99,     # make validation very strict so we trigger fallback
                qg_enable_fallback=True,
                qg_prioritize_visual=True,
            )

            claim = make_dummy_claim()
            # Stub extraction to return our single claim
            p.claim_extractor.extract_claims = lambda text: [claim]
            # Stub categorization to include a mix with filtered-out ones
            p.claim_categorizer.categorize_claim = lambda cl: _set_categories(
                cl,
                [
                    "VISUAL_GROUNDING_REQUIRED",
                    "EXTERNAL_KNOWLEDGE_REQUIRED",
                    "SUBJECTIVE_OPINION",        # should be filtered out for QG
                    "PROCEDURAL_DESCRIPTIVE",    # should be filtered out for QG
                ],
            )

            # Force primary generation to produce no valid items so fallback is used
            p.question_generator._generate_primary_questions = lambda claim_text, cat, n: []

            results = p.run("text")
            self.assertEqual(len(results), 1)
            c = results[0]

            # Only the two eligible categories should have questions
            self.assertEqual(set(c.socratic_questions.keys()), {"VISUAL_GROUNDING_REQUIRED", "EXTERNAL_KNOWLEDGE_REQUIRED"})

            # CLI questions_per_category should win over env
            self.assertEqual(len(c.socratic_questions["VISUAL_GROUNDING_REQUIRED"]), 2)
            self.assertEqual(len(c.socratic_questions["EXTERNAL_KNOWLEDGE_REQUIRED"]), 2)

            # All should be fallback and counted in stats
            self.assertEqual(p._qg_stats["total"], 4)
            self.assertEqual(p._qg_stats["fallback"], 4)


    def test_pipeline_qg_prioritization_orders_visual_first(self):
        lm = types.SimpleNamespace(generate_text=lambda *args, **kwargs: "1) Q")
        p = SocratesPipeline(
            llm_manager=lm,
            factuality_enabled=False,
            clarification_enabled=False,
            question_gen_enabled=True,
            questions_per_category=1,
            qg_min_threshold=0.2,
            qg_enable_fallback=True,
            qg_prioritize_visual=True,
        )

        claim = make_dummy_claim()
        p.claim_extractor.extract_claims = lambda text: [claim]
        p.claim_categorizer.categorize_claim = lambda cl: _set_categories(
            cl,
            ["EXTERNAL_KNOWLEDGE_REQUIRED", "VISUAL_GROUNDING_REQUIRED"],
        )

        # Let fallback handle generation to simplify; prioritization should reorder keys
        p.question_generator._generate_primary_questions = lambda claim_text, cat, n: []

        results = p.run("some text")
        c = results[0]
        keys = list(c.socratic_questions.keys())
        self.assertEqual(keys[0], "VISUAL_GROUNDING_REQUIRED")


    def test_pipeline_qg_env_applies_when_cli_missing(self):
        with mock.patch.dict(os.environ, {"QG_MIN_CONFIDENCE_THRESHOLD": "0.95"}, clear=False):
            lm = types.SimpleNamespace(generate_text=lambda *args, **kwargs: "1) Q")
            p = SocratesPipeline(
                llm_manager=lm,
                factuality_enabled=False,
                clarification_enabled=False,
                question_gen_enabled=True,
                # qg_min_threshold omitted -> env should apply
            )

            self.assertLess(abs(p.question_generator.config.min_confidence_threshold - 0.95), 1e-6)

            # Sanity run to ensure pipeline functions
            claim = make_dummy_claim()
            p.claim_extractor.extract_claims = lambda text: [claim]
            p.claim_categorizer.categorize_claim = lambda cl: _set_categories(
                cl,
                ["EXTERNAL_KNOWLEDGE_REQUIRED"],
            )
            # ensure we get something without depending on LLM
            p.question_generator._generate_primary_questions = lambda claim_text, cat, n: []

            res = p.run("text")
            self.assertEqual(len(res), 1)
            self.assertIn("EXTERNAL_KNOWLEDGE_REQUIRED", res[0].socratic_questions)


if __name__ == "__main__":
    unittest.main()
