import json
import unittest
import logging
from unittest.mock import MagicMock, patch
from socrates_system.modules.claim_categorizer import ClaimCategorizer
from socrates_system.modules.shared_structures import ExtractedClaim, ClaimCategory, ClaimCategoryType

class TestClaimCategorizer(unittest.TestCase):

    def setUp(self):
        # Create a fresh mock LLMManager for each test
        self.mock_llm_manager = MagicMock()
        self.categorizer = ClaimCategorizer(llm_manager=self.mock_llm_manager)
        
        # Set up the default mock response
        self.mock_response = {
            'categories': ['NUMERICAL_QUANTITATIVE'],
            'confidence': 0.9,
            'justification': 'The claim contains numerical data (330 meters)'
        }
        self.mock_llm_manager.generate.return_value = json.dumps([self.mock_response])
        
    def _setup_mock_response(self, categories, confidence=0.9, justification="Test justification"):
        """Helper to set up a mock response with specific categories."""
        if not isinstance(categories, list):
            categories = [categories]
        
        # Convert string category names to ClaimCategoryType enums if needed
        category_enums = []
        for cat in categories:
            if isinstance(cat, str):
                # Convert string to enum
                category_enums.append(ClaimCategoryType[cat])
            else:
                # Already an enum
                category_enums.append(cat)
            
        self.mock_response = {
            'categories': [cat.value for cat in category_enums],  # Store the string values
            'confidence': confidence,
            'justification': justification
        }
        
        # The LLM returns a JSON string with the category names as strings
        self.mock_llm_manager.generate.return_value = json.dumps([{
            'categories': [cat.name for cat in category_enums],
            'confidence': confidence,
            'justification': justification
        }])

    def test_numerical_quantitative_categorization(self):
        """Test categorization of a claim with numerical data."""
        # Setup mock response for numerical data
        self._setup_mock_response(
            categories=['NUMERICAL_QUANTITATIVE', 'FACTUAL'],
            justification="The claim contains numerical data (330 meters)"
        )
        
        claim = ExtractedClaim(
            text="The Eiffel Tower is 330 meters tall.",
            start_char=0,
            end_char=35,
            confidence=0.9,
            source_text="The Eiffel Tower is 330 meters tall."
        )

        categorized_claim = self.categorizer.categorize_claim(claim)

        self.assertIsNotNone(categorized_claim.categories)
        self.assertGreaterEqual(len(categorized_claim.categories), 1)
        
        # Check that the categories include NUMERICAL_QUANTITATIVE
        category_names = [cat.name[0] if isinstance(cat.name, list) else cat.name for cat in categorized_claim.categories]
        self.assertIn(ClaimCategoryType.NUMERICAL_QUANTITATIVE, category_names)

    def test_temporal_categorization(self):
        """Test categorization of a temporal claim."""
        # Setup mock response for temporal data
        self._setup_mock_response(
            categories=['TEMPORAL', 'FACTUAL'],
            justification="The claim contains a specific year (2023)"
        )
        
        claim = ExtractedClaim(
            text="The project was completed in 2023.",
            start_char=0,
            end_char=33,
            confidence=0.95,
            source_text="The project was completed in 2023."
        )

        categorized_claim = self.categorizer.categorize_claim(claim)
        
        # Extract category names, handling both list and single value cases
        category_names = []
        for cat in categorized_claim.categories:
            if isinstance(cat.name, list):
                category_names.extend(cat.name)
            else:
                category_names.append(cat.name)
        
        self.assertIn(ClaimCategoryType.TEMPORAL, category_names)
        self.assertIn(ClaimCategoryType.FACTUAL, category_names)
        
    def test_ambiguous_claim_handling(self):
        """Test handling of ambiguous claims."""
        # Setup mock response for ambiguous claim
        self._setup_mock_response(
            categories=['AMBIGUOUS_UNCLEAR'],
            confidence=0.8,
            justification="The claim is too vague to categorize definitively"
        )
        
        claim = ExtractedClaim(
            text="This thing is better than that thing.",
            start_char=0,
            end_char=35,
            confidence=0.9,
            source_text="This thing is better than that thing."
        )
        
        categorized_claim = self.categorizer.categorize_claim(claim)
        self.assertIsNotNone(categorized_claim.categories)
        self.assertGreaterEqual(len(categorized_claim.categories), 1)
        
        # Check that the categories include AMBIGUOUS_UNCLEAR
        category_names = [cat.name[0] if isinstance(cat.name, list) else cat.name for cat in categorized_claim.categories]
        self.assertIn(ClaimCategoryType.AMBIGUOUS_UNCLEAR, category_names)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
