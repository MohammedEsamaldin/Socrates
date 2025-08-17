import unittest

from socrates_system.modules.question_generator import AmbiguityDetector


class DummyLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        return self.reply


class TestAmbiguityDetector(unittest.TestCase):
    def test_no_ambiguity_returns_empty(self):
        det = AmbiguityDetector(llm_interface=DummyLLM("NO_AMBIGUITY"))
        self.assertEqual(det.detect_ambiguous_terms("Apple released a new iPhone model in 2024."), {})

    def test_parses_ambiguous_terms_list(self):
        reply = """
AMBIGUOUS_TERMS:
apple: fruit company, tech company
bank: river edge, financial institution, data bank, power bank, the bank
""".strip()
        det = AmbiguityDetector(llm_interface=DummyLLM(reply))
        terms = det.detect_ambiguous_terms("Apple is near the bank")
        self.assertEqual(set(terms.keys()), {"apple", "bank"})
        # limited to first 4 interpretations
        self.assertEqual(len(terms["bank"]), 4)
        self.assertIn(terms["apple"][0], {"fruit company", "tech company"})


if __name__ == "__main__":
    unittest.main()
