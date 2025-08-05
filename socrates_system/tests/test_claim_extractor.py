import unittest
import spacy
import json
from unittest.mock import MagicMock, patch
from socrates_system.modules.claim_extractor import ClaimExtractor
from socrates_system.modules.shared_structures import ExtractedClaim, ExtractedEntity, ExtractedRelationship

class TestClaimExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the ClaimExtractor once for all tests."""
        cls.extractor = ClaimExtractor(llm_manager=None)
        cls.sample_text = "The Eiffel Tower, located in Paris, is 330 meters tall. It was completed in 1889. Its height is greater than the Washington Monument's, which stands at 169 meters."

    def setUp(self):
        """Set up for each test."""
        # Ensure a clean slate for each test
        self.extractor = ClaimExtractor(llm_manager=None)

    def test_rule_based_extraction(self):
        """Test the rule-based claim extraction logic."""
        # Ensure we are testing the rule-based path
        self.extractor.llm_manager = None
        extracted_claims = self.extractor.extract_claims(self.sample_text)

        self.assertIsNotNone(extracted_claims)
        self.assertEqual(len(extracted_claims), 3)

        # Check content of the first claim
        self.assertEqual(extracted_claims[0].text, "The Eiffel Tower, located in Paris, is 330 meters tall")
        self.assertEqual(len(extracted_claims[0].entities), 3)
        self.assertEqual(extracted_claims[0].entities[0].text, "The Eiffel Tower")
        self.assertEqual(extracted_claims[0].entities[0].label, "LOC")

        # Check content of the second claim
        self.assertEqual(extracted_claims[1].text, "It was completed in 1889")
        self.assertEqual(len(extracted_claims[1].entities), 1)
        self.assertEqual(extracted_claims[1].entities[0].text, "1889")
        self.assertEqual(extracted_claims[1].entities[0].label, "DATE")

        # Check content of the third claim
        self.assertEqual(extracted_claims[2].text, "Its height is greater than the Washington Monument's, which stands at 169 meters")
        self.assertEqual(len(extracted_claims[2].entities), 2)
        self.assertEqual(extracted_claims[2].entities[0].text, "the Washington Monument's")



    def test_llm_based_extraction_and_parsing(self):
        """Test the LLM-based extraction and response parsing logic."""
        mock_llm_response = {
            "claims": [
                {
                    "claim_text": "The Eiffel Tower was completed in 1889.",
                    "confidence": 0.98,
                    "entities": [
                        {"text": "The Eiffel Tower", "label": "FAC", "start_char": 0, "end_char": 16},
                        {"text": "1889", "label": "DATE", "start_char": 32, "end_char": 36}
                    ],
                    "relationships": [
                        {"subject": "The Eiffel Tower", "relation": "was completed in", "object": "1889"}
                    ]
                }
            ]
        }

        # Mock the LLMManager and its response
        mock_llm_manager = MagicMock()
        mock_llm_manager.generate_text.return_value = f"```json\n{json.dumps(mock_llm_response)}\n```"

        # Use the mock LLM manager for this test
        self.extractor.llm_manager = mock_llm_manager

        # The text passed here must contain the claim from the mock response
        claims = self.extractor.extract_claims("The Eiffel Tower was completed in 1889.")

        self.assertEqual(len(claims), 1)
        claim = claims[0]

        self.assertEqual(claim.text, "The Eiffel Tower was completed in 1889")
        self.assertAlmostEqual(claim.confidence, 0.98, places=2)
        self.assertEqual(len(claim.entities), 2)
        self.assertEqual(claim.entities[0].text, "The Eiffel Tower")
        self.assertEqual(claim.entities[0].label, "FAC")


    def test_complex_multi_claim_extraction(self):
        """Test claim extraction on a complex text with multiple claims, opinions, and rhetorical questions."""
        complex_text = (
            "While many consider the Mona Lisa to be the most beautiful painting ever created, "
            "it was painted by Leonardo da Vinci between 1503 and 1506. Isn't it amazing that "
            "it's housed at the Louvre Museum in Paris? The museum, which opened in 1793, "
            "also displays the Venus de Milo, but the Mona Lisa's enigmatic smile is surely "
            "its most famous feature. Some reports suggest the painting is valued at over "
            "$860 million, a figure that seems almost unbelievable for a single artwork."
        )

        mock_llm_response = {
            "claims": [
                {
                    "claim_text": "The Mona Lisa was painted by Leonardo da Vinci.",
                    "confidence": 0.99,
                    "entities": [{"text": "Mona Lisa", "label": "WORK_OF_ART"}, {"text": "Leonardo da Vinci", "label": "PERSON"}]
                },
                {
                    "claim_text": "The Mona Lisa was painted between 1503 and 1506.",
                    "confidence": 0.95,
                    "entities": [{"text": "Mona Lisa", "label": "WORK_OF_ART"}, {"text": "1503", "label": "DATE"}, {"text": "1506", "label": "DATE"}]
                },
                {
                    "claim_text": "The Mona Lisa is housed at the Louvre Museum in Paris.",
                    "confidence": 0.98,
                    "entities": [{"text": "Mona Lisa", "label": "WORK_OF_ART"}, {"text": "Louvre Museum", "label": "ORG"}, {"text": "Paris", "label": "GPE"}]
                },
                {
                    "claim_text": "The Louvre Museum opened in 1793.",
                    "confidence": 0.92,
                    "entities": [{"text": "Louvre Museum", "label": "ORG"}, {"text": "1793", "label": "DATE"}]
                },
                {
                    "claim_text": "The Louvre Museum displays the Venus de Milo.",
                    "confidence": 0.90,
                    "entities": [{"text": "Louvre Museum", "label": "ORG"}, {"text": "Venus de Milo", "label": "WORK_OF_ART"}]
                },
                {
                    "claim_text": "The Mona Lisa is valued at over $860 million.",
                    "confidence": 0.85,
                    "entities": [{"text": "Mona Lisa", "label": "WORK_OF_ART"}, {"text": "$860 million", "label": "MONEY"}]
                }
            ]
        }

        mock_llm_manager = MagicMock()
        mock_llm_manager.generate_text.return_value = f"```json\n{json.dumps(mock_llm_response)}\n```"
        self.extractor.llm_manager = mock_llm_manager

        extracted_claims = self.extractor.extract_claims(complex_text)

        self.assertEqual(len(extracted_claims), 6)
        
        # Check that opinions and questions were ignored and facts were extracted
        extracted_texts = {claim.text for claim in extracted_claims}
        expected_texts = {
            "The Mona Lisa was painted by Leonardo da Vinci",
            "The Mona Lisa was painted between 1503 and 1506",
            "The Mona Lisa is housed at the Louvre Museum in Paris",
            "The Louvre Museum opened in 1793",
            "The Louvre Museum displays the Venus de Milo",
            "The Mona Lisa is valued at over $860 million"
        }
        
        self.assertEqual(extracted_texts, expected_texts)

if __name__ == '__main__':
    unittest.main()
