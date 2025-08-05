import unittest
import logging
from socrates_system.modules.claim_categorizer import ClaimCategorizer, ClaimCategoryType
from socrates_system.modules.shared_structures import ExtractedClaim

class TestClaimCategorizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.categorizer = ClaimCategorizer(llm_manager=None)

    def test_rule_based_categorization(self):
        """Test the rule-based claim categorization logic."""
        claim = ExtractedClaim(
            text="The Eiffel Tower is 330 meters tall.",
            start_char=0,
            end_char=35,
            confidence=0.9,
            source_text="The Eiffel Tower is 330 meters tall."
        )

        categorized_claim = self.categorizer.categorize_claim(claim)

        self.assertIsNotNone(categorized_claim.categories)
        self.assertEqual(len(categorized_claim.categories), 1)
        self.assertEqual(categorized_claim.categories[0].name, ClaimCategoryType.QUANTITATIVE)

    def test_temporal_categorization(self):
        """Test categorization of a temporal claim."""
        claim = ExtractedClaim(
            text="The project was completed in 2023.",
            start_char=0,
            end_char=33,
            confidence=0.95,
            source_text="The project was completed in 2023."
        )

        categorized_claim = self.categorizer.categorize_claim(claim)
        category_names = {cat.name for cat in categorized_claim.categories}

        self.assertIn(ClaimCategoryType.TEMPORAL, category_names)
        self.assertIn(ClaimCategoryType.QUANTITATIVE, category_names)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
