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

    def test_routing_to_cross_modal(self):
        """Test that a visual grounding claim is routed to CROSS_MODAL."""
        claim = ExtractedClaim(
            text="The person is wearing a red hat.",
            start_char=0, end_char=33, confidence=0.9, source_text="",
            categories=[
                ClaimCategory(name=ClaimCategoryType.VISUAL_GROUNDING_REQUIRED, confidence=0.8, justification="...")
            ]
        )

        route = self.router.route_claim(claim)
        self.assertIsNotNone(route)
        self.assertEqual(route.method, VerificationMethod.CROSS_MODAL)

    def test_routing_to_external_source(self):
        """Test that an external knowledge claim is routed to EXTERNAL_SOURCE."""
        claim = ExtractedClaim(
            text="It was completed in 1889.",
            start_char=0, end_char=25, confidence=0.9, source_text="",
            categories=[
                ClaimCategory(name=ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED, confidence=0.8, justification="...")
            ]
        )

        route = self.router.route_claim(claim)
        self.assertIsNotNone(route)
        self.assertEqual(route.method, VerificationMethod.EXTERNAL_SOURCE)

    def test_routing_to_knowledge_graph(self):
        """Test that a self-consistency claim is routed to KNOWLEDGE_GRAPH."""
        claim = ExtractedClaim(
            text="Paris is the capital of France.",
            start_char=0, end_char=29, confidence=0.9, source_text="",
            categories=[
                ClaimCategory(name=ClaimCategoryType.SELF_CONSISTENCY_REQUIRED, confidence=0.8, justification="...")
            ]
        )

        route = self.router.route_claim(claim)
        self.assertIsNotNone(route)
        self.assertEqual(route.method, VerificationMethod.KNOWLEDGE_GRAPH)

if __name__ == '__main__':
    unittest.main()
