import unittest
from socrates_system.modules.check_router import CheckRouter
from socrates_system.modules.shared_structures import (
    ExtractedClaim,
    ClaimCategory,
    ClaimCategoryType,
    VerificationMethod
)

class TestCheckRouter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.router = CheckRouter()

    def test_routing_to_calculation(self):
        """Test that a quantitative claim is routed to CALCULATION."""
        claim = ExtractedClaim(
            text="The tower is 330 meters tall.",
            start_char=0, end_char=29, confidence=0.9, source_text="",
            categories=[
                ClaimCategory(name=ClaimCategoryType.NUMERICAL_QUANTITATIVE, confidence=0.8, justification="...")
            ]
        )

        route = self.router.route_claim(claim)
        self.assertIsNotNone(route)
        self.assertEqual(route.method, VerificationMethod.CALCULATION)

    def test_routing_to_external_source(self):
        """Test that a temporal claim is routed to EXTERNAL_SOURCE."""
        claim = ExtractedClaim(
            text="It was completed in 1889.",
            start_char=0, end_char=25, confidence=0.9, source_text="",
            categories=[
                ClaimCategory(name=ClaimCategoryType.TEMPORAL, confidence=0.8, justification="...")
            ]
        )

        route = self.router.route_claim(claim)
        self.assertIsNotNone(route)
        self.assertEqual(route.method, VerificationMethod.EXTERNAL_SOURCE)

    def test_routing_to_knowledge_graph(self):
        """Test that a relational claim is routed to KNOWLEDGE_GRAPH."""
        claim = ExtractedClaim(
            text="Paris is the capital of France.",
            start_char=0, end_char=29, confidence=0.9, source_text="",
            categories=[
                ClaimCategory(name=ClaimCategoryType.RELATIONAL, confidence=0.8, justification="...")
            ]
        )

        route = self.router.route_claim(claim)
        self.assertIsNotNone(route)
        self.assertEqual(route.method, VerificationMethod.KNOWLEDGE_GRAPH)

if __name__ == '__main__':
    unittest.main()
