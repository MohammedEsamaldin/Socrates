"""
Integration pipeline for the Socrates Agent.

This script demonstrates how to use the claim extractor, categorizer, and router
modules together to process a piece of text and prepare claims for verification.
"""
import logging
from typing import List, Dict, Any

from socrates_system.modules.claim_extractor import ClaimExtractor
from socrates_system.modules.claim_categorizer import ClaimCategorizer
from socrates_system.modules.check_router import CheckRouter
from socrates_system.modules.llm_manager import LLMManager
from socrates_system.modules.shared_structures import ExtractedClaim, VerificationMethod, VerificationRoute
from socrates_system.modules.external_factuality_checker import ExternalFactualityChecker
import os
import argparse
from socrates_system.clarification_resolution import ClarificationResolutionModule
from socrates_system.clarification_resolution.data_models import (
    ClarificationContext as ClarContext,
    FactCheckResult as ClarFactCheckResult,
    IssueType,
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SocratesPipeline:
    """Orchestrates the claim processing pipeline."""

    def __init__(self, llm_manager: any = None, factuality_enabled: bool = None, clarification_enabled: bool = None, clarification_dev_mode: bool = None):
        """Initializes the pipeline with all necessary components."""
        logging.info("Initializing Socrates Pipeline...")
        self.claim_extractor = ClaimExtractor(llm_manager=llm_manager)
        self.claim_categorizer = ClaimCategorizer(llm_manager=llm_manager)
        self.llm_manager = llm_manager
        
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

        # Factuality check toggle from CLI/env
        if factuality_enabled is None:
            factuality_enabled = os.getenv("FACTUALITY_ENABLED", "true").lower() == "true"
        self.factuality_enabled = factuality_enabled
        self.external_checker = ExternalFactualityChecker() if self.factuality_enabled else None
        # Clarification module (optional)
        if clarification_enabled is None:
            clarification_enabled = os.getenv("CLARIFICATION_ENABLED", "true").lower() == "true"
        if clarification_dev_mode is None:
            clarification_dev_mode = os.getenv("CLARIFICATION_DEV_MODE", "false").lower() == "true"
        self.clarification_enabled = clarification_enabled
        self.clarification_dev_mode = clarification_dev_mode
        if self.clarification_enabled:
            try:
                self.clarifier = ClarificationResolutionModule(llm_manager=self.llm_manager, dev_mode=self.clarification_dev_mode)
            except Exception as e:
                logging.warning(f"Clarification module unavailable: {e}")
                self.clarifier = None
        else:
            self.clarifier = None
        self._clarification_results: Dict[int, Any] = {}
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

        processed_claims: List[ExtractedClaim] = []
        factuality_results = {}
        for i, claim in enumerate(claims, 1):
            logging.info(f"--- Processing Claim {i}/{len(claims)}: '{claim.text[:80]}...' ---")
            
            # 2. Categorize each claim
            # The `categorize_claim` method now returns the updated claim object
            categorized_claim = self.claim_categorizer.categorize_claim(claim)
            logging.info(f"Categorized as: {[c.name for c in categorized_claim.categories]}")

            # 2.1 Clarify ambiguous claims before routing (LLM-only per spec)
            try:
                if self.clarifier and any(getattr(c.name, 'name', str(c.name)) == 'AMBIGUOUS_RESOLUTION_REQUIRED' for c in categorized_claim.categories):
                    logging.info("Ambiguity detected; invoking Clarification Resolution Module (pre-routing)...")
                    # Build fact-check result placeholder
                    fc = ClarFactCheckResult(
                        verdict="UNCERTAIN",
                        confidence=0.0,
                        reasoning="Ambiguous claim - clarification required",
                        evidence=[],
                        sources=[],
                    )
                    # Choose ambiguous category or first
                    cat_enum = next((c.name for c in categorized_claim.categories if getattr(c.name, 'name', '') == 'AMBIGUOUS_RESOLUTION_REQUIRED'),
                                     (categorized_claim.categories[0].name if categorized_claim.categories else None))
                    ctx = ClarContext(
                        claim_text=categorized_claim.text,
                        category=cat_enum,
                        fact_check=fc,
                        failed_check_type="EXPERT_VERIFICATION",
                        issue_type=IssueType.AMBIGUITY,
                        claim_id=str(i),
                        metadata={"stage": "pre_route"},
                    )
                    clr_res = self.clarifier.resolve_claim(ctx)
                    self._clarification_results[i] = {"pre": clr_res}
                    if clr_res.corrected_claim and clr_res.corrected_claim.strip() and clr_res.corrected_claim.strip() != categorized_claim.text.strip():
                        logging.info("Applying corrected claim from clarification (pre-route)")
                        categorized_claim.text = clr_res.corrected_claim.strip()
                        # Re-categorize after correction
                        categorized_claim = self.claim_categorizer.categorize_claim(categorized_claim)
                        logging.info(f"Re-categorized as: {[c.name for c in categorized_claim.categories]}")
            except Exception as e:
                logging.warning(f"Pre-routing clarification failed: {e}")

            # 3. Route each claim for verification
            # 3. Route each claim for verification
            # The `route_claim` method returns a VerificationRoute object
            route = self.check_router.route_claim(categorized_claim)
            categorized_claim.verification_route = route
            logging.info(f"Routed to: {route.method.name}")
            # Apply pre-routing clarification next_action if it directs to KG
            pre_clr = self._clarification_results.get(i, {}).get("pre") if hasattr(self, "_clarification_results") else None
            if pre_clr:
                try:
                    next_action_name = getattr(pre_clr.next_action, 'name', str(pre_clr.next_action))
                    if next_action_name == 'DIRECT_TO_KG' and route.method != VerificationMethod.KNOWLEDGE_GRAPH:
                        categorized_claim.verification_route = VerificationRoute(
                            method=VerificationMethod.KNOWLEDGE_GRAPH,
                            confidence=getattr(pre_clr, 'resolution_confidence', 0.8) or 0.8,
                            justification="Clarification module directed claim to Knowledge Graph (pre-routing)",
                            estimated_cost=1.0,
                            estimated_latency=0.1,
                        )
                        logging.info("Route overridden to KNOWLEDGE_GRAPH by Clarification Module (pre)")
                except Exception as e:
                    logging.warning(f"Failed to apply pre-clarification routing override: {e}")

            # 4. External factuality check for routed claims (if enabled)
            if self.factuality_enabled and route.method == VerificationMethod.EXTERNAL_SOURCE:
                logging.info("Performing external factuality check for this claim...")
                try:
                    result = self.external_checker.verify_claim(categorized_claim.text)
                    factuality_results[i] = result
                    status = result.get("status")
                    conf = result.get("confidence", 0.0)
                    logging.info(f"Factuality: {status} (conf {conf:.2f})")
                    # Persist onto claim
                    categorized_claim.factuality_status = status
                    categorized_claim.factuality_confidence = conf
                    categorized_claim.factuality_verdict = True if status == "PASS" else (False if status == "FAIL" else None)
                    categorized_claim.factuality_evidence = result.get("evidence", [])
                    categorized_claim.factuality_sources = result.get("sources", [])
                    categorized_claim.factuality_reasoning = result.get("reasoning")
                    # 4.1 If conflict or uncertainty, run clarification and optionally rerun verification
                    if self.clarifier and status in ("FAIL", "UNCERTAIN"):
                        logging.info("Conflict/uncertainty detected; invoking Clarification Resolution Module (post-factuality)...")
                        # Map evidence strings to dicts with 'summary'
                        ev_list = []
                        for ev in result.get("evidence", []) or []:
                            if isinstance(ev, str):
                                ev_list.append({"summary": ev})
                            elif isinstance(ev, dict):
                                ev_list.append(ev)
                        for ev in result.get("external_facts", []) or []:
                            if isinstance(ev, str):
                                ev_list.append({"summary": ev})
                        fc = ClarFactCheckResult(
                            verdict=status,
                            confidence=float(conf) if not isinstance(conf, str) else float(conf or 0.0),
                            reasoning=result.get("reasoning"),
                            evidence=ev_list,
                            sources=result.get("sources", []),
                        )
                        # Use category if present, else EXTERNAL_KNOWLEDGE_REQUIRED
                        cat_enum = next((c.name for c in categorized_claim.categories), None)
                        try:
                            ctx = ClarContext(
                                claim_text=categorized_claim.text,
                                category=cat_enum,
                                fact_check=fc,
                                failed_check_type="EXTERNAL_SOURCE",
                                issue_type=IssueType.EXTERNAL_FACTUAL_CONFLICT,
                                claim_id=str(i),
                                metadata={"stage": "post_factuality"},
                            )
                        except Exception:
                            # If category missing, reuse ambiguous fallback
                            ctx = ClarContext(
                                claim_text=categorized_claim.text,
                                category=cat_enum,
                                fact_check=fc,
                                failed_check_type="EXTERNAL_SOURCE",
                                issue_type=IssueType.EXTERNAL_FACTUAL_CONFLICT,
                                claim_id=str(i),
                                metadata={"stage": "post_factuality"},
                            )
                        clr_res2 = self.clarifier.resolve_claim(ctx)
                        self._clarification_results.setdefault(i, {})["post"] = clr_res2
                        # Apply post-factuality next_action if direct-to-KG
                        try:
                            next_action_name2 = getattr(clr_res2.next_action, 'name', str(clr_res2.next_action))
                            if next_action_name2 == 'DIRECT_TO_KG':
                                categorized_claim.verification_route = VerificationRoute(
                                    method=VerificationMethod.KNOWLEDGE_GRAPH,
                                    confidence=getattr(clr_res2, 'resolution_confidence', 0.8) or 0.8,
                                    justification="Clarification module directed claim to Knowledge Graph (post-factuality)",
                                    estimated_cost=1.0,
                                    estimated_latency=0.1,
                                )
                                logging.info("Route overridden to KNOWLEDGE_GRAPH by Clarification Module (post)")
                        except Exception as e:
                            logging.warning(f"Failed to apply post-clarification routing override: {e}")
                        if clr_res2.corrected_claim and clr_res2.corrected_claim.strip() and clr_res2.corrected_claim.strip() != categorized_claim.text.strip():
                            logging.info("Applying corrected claim from clarification (post-factuality)")
                            categorized_claim.text = clr_res2.corrected_claim.strip()
                            # Optionally rerun external verification based on module decision
                            if getattr(clr_res2, 'rerun_verification', False):
                                try:
                                    logging.info("Re-running external factuality check on corrected claim...")
                                    result2 = self.external_checker.verify_claim(categorized_claim.text)
                                    factuality_results[i] = result2
                                    status2 = result2.get("status")
                                    conf2 = result2.get("confidence", 0.0)
                                    categorized_claim.factuality_status = status2
                                    categorized_claim.factuality_confidence = conf2
                                    categorized_claim.factuality_verdict = True if status2 == "PASS" else (False if status2 == "FAIL" else None)
                                    categorized_claim.factuality_evidence = result2.get("evidence", [])
                                    categorized_claim.factuality_sources = result2.get("sources", [])
                                    categorized_claim.factuality_reasoning = result2.get("reasoning")
                                except Exception as e:
                                    logging.warning(f"Rerun factuality check failed: {e}")
                except Exception as e:
                    logging.error(f"Factuality check error: {e}")
                    factuality_results[i] = {
                        "status": "ERROR",
                        "confidence": 0.0,
                        "external_facts": [],
                        "contradictions": [str(e)],
                        "evidence": [],
                        "sources": [],
                        "reasoning": "Exception during factuality check"
                    }
                    categorized_claim.factuality_status = "ERROR"
                    categorized_claim.factuality_confidence = 0.0
                    categorized_claim.factuality_verdict = None
                    categorized_claim.factuality_evidence = []
                    categorized_claim.factuality_sources = []
                    categorized_claim.factuality_reasoning = "Exception during factuality check"
            
            processed_claims.append(categorized_claim)

        # Summary metrics for factuality stage
        if self.factuality_enabled and factuality_results:
            total = len(factuality_results)
            pass_n = sum(1 for r in factuality_results.values() if r.get("status") == "PASS")
            fail_n = sum(1 for r in factuality_results.values() if r.get("status") == "FAIL")
            uncertain_n = sum(1 for r in factuality_results.values() if r.get("status") == "UNCERTAIN")
            avg_conf = sum(r.get("confidence", 0.0) for r in factuality_results.values()) / max(total, 1)
            logging.info(f"Factuality summary: total={total}, PASS={pass_n}, FAIL={fail_n}, UNCERTAIN={uncertain_n}, avg_conf={avg_conf:.2f}")

        logging.info("Claim processing pipeline finished.")
        # Note: keeping return type as list to avoid breaking external callers
        # Factuality results are logged; CLI mode prints them below.
        self._last_factuality_results = factuality_results
        return processed_claims

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Socrates claim processing pipeline")
    parser.add_argument("--enable-factuality", dest="enable_factuality", action="store_true", help="Enable external factuality checking")
    parser.add_argument("--disable-factuality", dest="disable_factuality", action="store_true", help="Disable external factuality checking")
    parser.add_argument("--text", type=str, default="in this image I was standing in front of a London Big Ben tower, which is in Germany.", help="Input text to process")
    # Clarification toggles
    parser.add_argument("--enable-clarification", dest="enable_clarification", action="store_true", help="Enable clarification module")
    parser.add_argument("--disable-clarification", dest="disable_clarification", action="store_true", help="Disable clarification module")
    parser.add_argument("--clar-dev", dest="clar_dev", action="store_true", help="Enable clarification dev mode")
    args = parser.parse_args()

    # Resolve factuality toggle from CLI overriding env
    env_enabled = os.getenv("FACTUALITY_ENABLED", "true").lower() == "true"
    if args.enable_factuality:
        factuality_enabled = True
    elif args.disable_factuality:
        factuality_enabled = False
    else:
        factuality_enabled = env_enabled
    # Resolve clarification toggles
    env_clar = os.getenv("CLARIFICATION_ENABLED", "true").lower() == "true"
    env_clar_dev = os.getenv("CLARIFICATION_DEV_MODE", "false").lower() == "true"
    if args.enable_clarification:
        clarification_enabled = True
    elif args.disable_clarification:
        clarification_enabled = False
    else:
        clarification_enabled = env_clar
    clarification_dev_mode = True if args.clar_dev else env_clar_dev

    # To run with a real LLM, instantiate a configured LLMManager
    llm_manager = LLMManager()
    pipeline = SocratesPipeline(
        llm_manager=llm_manager,
        factuality_enabled=factuality_enabled,
        clarification_enabled=clarification_enabled,
        clarification_dev_mode=clarification_dev_mode,
    )

    final_claims = pipeline.run(args.text)

    print("\n--- Final Processed Claims ---")
    for i, claim in enumerate(final_claims, 1):
        print(f"\nClaim {i}: {claim.text}")
        # Handle confidence conversion from string to float
        if isinstance(claim.confidence, str):
            confidence_map = {
                'Low': 0.2, 'low': 0.2,
                'Medium': 0.5, 'medium': 0.5, 
                'High': 0.8, 'high': 0.8
            }
            confidence = confidence_map.get(claim.confidence, 0.5)
        else:
            confidence = float(claim.confidence)
        print(f"  - Confidence: {confidence:.2f}")
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
        # Clarification results (if available)
        clr = getattr(pipeline, "_clarification_results", {}).get(i)
        if clr:
            print("  - Clarifications:")
            pre = clr.get("pre")
            post = clr.get("post")
            if pre:
                print("    - Pre-routing:")
                print(f"      corrected: {getattr(pre, 'corrected_claim', None)}")
                print(f"      confidence: {getattr(pre, 'resolution_confidence', 0.0):.2f}")
                print(f"      next_action: {getattr(getattr(pre, 'next_action', ''), 'name', str(getattr(pre, 'next_action', '')))}")
            if post:
                print("    - Post-factuality:")
                print(f"      corrected: {getattr(post, 'corrected_claim', None)}")
                print(f"      confidence: {getattr(post, 'resolution_confidence', 0.0):.2f}")
                print(f"      next_action: {getattr(getattr(post, 'next_action', ''), 'name', str(getattr(post, 'next_action', '')))}")
        # Factuality result (if available)
        fr = getattr(pipeline, "_last_factuality_results", {}).get(i)
        if fr:
            print("  - Factuality:")
            print(f"    - Status: {fr.get('status')}, Confidence: {fr.get('confidence', 0.0):.2f}")
            verdict = (True if fr.get('status') == 'PASS' else (False if fr.get('status') == 'FAIL' else None))
            print(f"    - Verdict: {verdict}")
            if fr.get("external_facts"):
                print(f"    - External Facts: {fr['external_facts'][:2]}")
            if fr.get("contradictions"):
                print(f"    - Contradictions: {fr['contradictions'][:2]}")
            if fr.get("sources"):
                print(f"    - Sources: {fr['sources'][:3]}")

    # Summary metrics (also logged above)
    if getattr(pipeline, "_last_factuality_results", None):
        frs = pipeline._last_factuality_results
        total = len(frs)
        pass_n = sum(1 for r in frs.values() if r.get("status") == "PASS")
        fail_n = sum(1 for r in frs.values() if r.get("status") == "FAIL")
        uncertain_n = sum(1 for r in frs.values() if r.get("status") == "UNCERTAIN")
        avg_conf = sum(r.get("confidence", 0.0) for r in frs.values()) / max(total, 1)
        print("\n--- Factuality Summary ---")
        print(f"Total checked: {total} | PASS: {pass_n} | FAIL: {fail_n} | UNCERTAIN: {uncertain_n} | Avg conf: {avg_conf:.2f}")
