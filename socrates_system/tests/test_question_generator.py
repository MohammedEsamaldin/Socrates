import unittest

from socrates_system.modules.question_generator import (
    SocraticQuestionGenerator,
    SocraticConfig,
    VerificationCapabilities,
    SocraticQuestion,
)


class DummyLLM:
    def __init__(self, mode="ok"):
        self.mode = mode

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        if self.mode == "error":
            raise RuntimeError("LLM failure")
        # Return a simple numbered list so _parse_questions_list picks them up
        return "\n".join([
            "1) What evidence supports the claim?",
            "2) How can we independently verify the claim?",
            "3) Which reliable sources confirm the statement?",
        ])


def make_gen(llm_mode="ok", cfg_overrides=None):
    caps = VerificationCapabilities(
        visual_grounding=["object_detection", "spatial_relationships"],
        external_knowledge=["wikipedia", "factcheck"],
        self_consistency=["knowledge_graph"],
    )
    cfg = SocraticConfig()
    if cfg_overrides:
        cfg.update(**cfg_overrides)
    # Pass our DummyLLM directly as the llm_interface (it only needs a .generate method)
    return SocraticQuestionGenerator(verification_capabilities=caps, llm_interface=DummyLLM(mode=llm_mode), config=cfg)

class TestSocraticQuestionGenerator(unittest.TestCase):
    def test_validator_respects_threshold_and_complexity(self):
        gen = make_gen(cfg_overrides={"min_confidence_threshold": 0.2, "max_question_complexity_ratio": 2.0})
        claim = "Paris is the capital of France"
        res = gen.generate_questions(claim, ["EXTERNAL_KNOWLEDGE_REQUIRED"], num_questions=2)
        self.assertIn("EXTERNAL_KNOWLEDGE_REQUIRED", res)
        qs = res["EXTERNAL_KNOWLEDGE_REQUIRED"]
        self.assertEqual(len(qs), 2)
        self.assertTrue(all(isinstance(q, SocraticQuestion) for q in qs))
        self.assertTrue(all(q.fallback is False for q in qs))
        self.assertTrue(all(0.0 <= q.confidence_score <= 1.0 for q in qs))

    def test_fallback_used_when_llm_errors(self):
        gen = make_gen(llm_mode="error", cfg_overrides={"min_confidence_threshold": 0.9, "enable_fallback": True})
        claim = "The Eiffel Tower is in Paris"
        res = gen.generate_questions(claim, ["EXTERNAL_KNOWLEDGE_REQUIRED"], num_questions=3)
        qs = res["EXTERNAL_KNOWLEDGE_REQUIRED"]
        self.assertEqual(len(qs), 3)
        self.assertTrue(all(q.fallback for q in qs))
        # Hints should be mapped for the category
        self.assertTrue(all("external" in q.verification_hint.lower() or "appropriate" in q.verification_hint.lower() for q in qs))

    def test_prioritization_places_visual_first_when_requested(self):
        gen = make_gen(cfg_overrides={"min_confidence_threshold": 0.2})
        claim = "The image shows a red car next to a tree"
        res = gen.generate_questions(
            claim,
            ["EXTERNAL_KNOWLEDGE_REQUIRED", "VISUAL_GROUNDING_REQUIRED"],
            num_questions=1,
            prioritize_category="VISUAL_GROUNDING_REQUIRED",
        )
        keys = list(res.keys())
        self.assertEqual(keys[0], "VISUAL_GROUNDING_REQUIRED")


if __name__ == "__main__":
    unittest.main()
