"""
Integration pipeline for the Socrates Agent.

This script demonstrates how to use the claim extractor, categorizer, and router
modules together to process a piece of text and prepare claims for verification.
"""
import logging
from typing import List

from socrates_system.modules.claim_extractor import ClaimExtractor
from socrates_system.modules.claim_categorizer import ClaimCategorizer
from socrates_system.modules.check_router import CheckRouter
from socrates_system.modules.llm_manager import LLMManager
from socrates_system.modules.shared_structures import ExtractedClaim, VerificationMethod

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SocratesPipeline:
    """Orchestrates the claim processing pipeline."""

    def __init__(self, llm_manager: any = None):
        """Initializes the pipeline with all necessary components."""
        logging.info("Initializing Socrates Pipeline...")
        self.claim_extractor = ClaimExtractor(llm_manager=llm_manager)
        self.claim_categorizer = ClaimCategorizer(llm_manager=llm_manager)
        
        # Initialize with a set of available verification methods
        available_methods = {
            VerificationMethod.KNOWLEDGE_GRAPH,
            VerificationMethod.EXTERNAL_SOURCE,
            VerificationMethod.CALCULATION,
            VerificationMethod.EXPERT_VERIFICATION,
            VerificationMethod.DEFINITIONAL,
            VerificationMethod.CROSS_MODAL,
        }
        self.check_router = CheckRouter(available_methods=available_methods)
        logging.info("Socrates Pipeline initialized successfully.")

    def run(self, text: str) -> List[ExtractedClaim]:
        """
        Runs the full claim processing pipeline on a given text.

        Args:
            text: The input text to process.

        Returns:
            A list of processed claims, ready for verification.
        """
        logging.info("Starting claim processing pipeline...")

        # 1. Extract claims
        claims = self.claim_extractor.extract_claims(text)
        if not claims:
            logging.warning("No claims were extracted.")
            return []
        logging.info(f"Extracted {len(claims)} initial claims.")

        processed_claims = []
        for i, claim in enumerate(claims, 1):
            logging.info(f"--- Processing Claim {i}/{len(claims)}: '{claim.text[:80]}...' ---")
            
            # 2. Categorize each claim
            # The `categorize_claim` method now returns the updated claim object
            categorized_claim = self.claim_categorizer.categorize_claim(claim)
            logging.info(f"Categorized as: {[c.name for c in categorized_claim.categories]}")

            # 3. Route each claim for verification
            # 3. Route each claim for verification
            # The `route_claim` method returns a VerificationRoute object
            route = self.check_router.route_claim(categorized_claim)
            categorized_claim.verification_route = route
            logging.info(f"Routed to: {route.method.name}")
            
            processed_claims.append(categorized_claim)

        logging.info("Claim processing pipeline finished.")
        return processed_claims

if __name__ == '__main__':
    # To run with a real LLM, you would instantiate a configured LLMManager
    # For this example, we can use a mock or None to test rule-based fallbacks
    llm_manager = LLMManager()

    pipeline = SocratesPipeline(llm_manager=llm_manager)

    sample_text = """
    president of Sudan said "War is a good thing", did I read that correct?
    """

    final_claims = pipeline.run(sample_text)

    print("\n--- Final Processed Claims ---")
    for i, claim in enumerate(final_claims, 1):
        print(f"\nClaim {i}: {claim.text}")
        print(f"  - Confidence: {claim.confidence:.2f}")
        print(f"  - Context: {claim.context_window}")
        print("  - Entities:")
        for entity in claim.entities:
            print(f"    - '{entity.text}' ({entity.label})")
        if claim.relationships:
            print("  - Relationships:")
            for rel in claim.relationships:
                print(f"    - ({rel.subject}) -> [{rel.relation}] -> ({rel.object})")
        print("  - Categories:")
        for category in claim.categories:
            print(f"    - {category.name} (Confidence: {category.confidence:.2f}): {category.justification}")
        if claim.verification_route:
            route = claim.verification_route
            print(f"  - Verification Route: {route.method.name}")
            print(f"    - Justification: {route.justification}")
            print(f"    - Estimated Cost: {route.estimated_cost}, Latency: {route.estimated_latency}s")
