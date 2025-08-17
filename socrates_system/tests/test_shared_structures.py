import unittest
from socrates_system.modules.shared_structures import (
    ExtractedEntity,
    ExtractedClaim,
    ClaimCategory,
    ClaimCategoryType,
    VerificationRoute,
    VerificationMethod
)

class TestSharedStructures(unittest.TestCase):

    def test_extracted_entity_creation(self):
        entity = ExtractedEntity(
            text="Eiffel Tower",
            label="LOC",
            start_char=4,
            end_char=16
        )
        self.assertEqual(entity.text, "Eiffel Tower")
        self.assertEqual(entity.label, "LOC")

    def test_extracted_claim_creation(self):
        claim = ExtractedClaim(
            text="The Eiffel Tower is 330 meters tall.",
            start_char=0,
            end_char=35,
            confidence=0.9,
            source_text="The Eiffel Tower is 330 meters tall and is in Paris."
        )
        self.assertEqual(claim.confidence, 0.9)
        self.assertEqual(len(claim.entities), 0)

    def test_claim_category_creation(self):
        category = ClaimCategory(
            name=ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED,
            confidence=0.95,
            justification="This is a factual statement."
        )
        self.assertEqual(category.name, ClaimCategoryType.EXTERNAL_KNOWLEDGE_REQUIRED)

    def test_verification_route_creation(self):
        route = VerificationRoute(
            method=VerificationMethod.EXTERNAL_SOURCE,
            confidence=0.88,
            justification="Requires external lookup.",
            estimated_cost=0.5,
            estimated_latency=1.0
        )
        self.assertEqual(route.method, VerificationMethod.EXTERNAL_SOURCE)

if __name__ == '__main__':
    unittest.main()
