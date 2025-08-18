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
from socrates_system.modules.knowledge_graph_manager import KnowledgeGraphManager
from socrates_system.modules.self_contradiction_checker import SelfContradictionChecker
from socrates_system.modules.conflict_resolver import ConflictResolver
import os
import argparse
import uuid
from socrates_system.clarification_resolution import ClarificationResolutionModule
from socrates_system.clarification_resolution.data_models import (
    ClarificationContext as ClarContext,
    FactCheckResult as ClarFactCheckResult,
    IssueType,
)

# Socratic Question Generator
from socrates_system.modules.question_generator import (
    SocraticQuestionGenerator,
    SocraticConfig,
    VerificationCapabilities,
    LLMInterfaceAdapter,
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Console Colors (centralized) ----------------
# Simple ANSI color helper with env overrides, e.g. SOC_COLOR_HEADING=cyan
import sys

class ConsoleColors:
    _ANSI = {
        'reset': "\033[0m",
        'bold': "\033[1m",
        'black': "\033[30m",
        'red': "\033[31m",
        'green': "\033[32m",
        'yellow': "\033[33m",
        'blue': "\033[34m",
        'magenta': "\033[35m",
        'cyan': "\033[36m",
        'white': "\033[37m",
        'bright_black': "\033[90m",
        'bright_red': "\033[91m",
        'bright_green': "\033[92m",
        'bright_yellow': "\033[93m",
        'bright_blue': "\033[94m",
        'bright_magenta': "\033[95m",
        'bright_cyan': "\033[96m",
        'bright_white': "\033[97m",
    }

    # Default role->color mapping; override via env SOC_COLOR_<ROLE>=<color>
    _ROLE_DEFAULTS = {
        'heading': 'bright_cyan',
        'claim': 'bright_white',
        'label': 'bright_black',
        'value': 'white',
        'entity': 'bright_blue',
        'category': 'bright_magenta',
        'question': 'bright_yellow',
        'route': 'bright_green',
        'clarification': 'bright_cyan',
        'factuality_pass': 'green',
        'factuality_fail': 'red',
        'factuality_uncertain': 'yellow',
        'summary': 'bright_cyan'
    }

    @staticmethod
    def _supports_color() -> bool:
        if os.getenv('NO_COLOR'):
            return False
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    @classmethod
    def c(cls, role: str, text: str) -> str:
        if not cls._supports_color():
            return text
        # Resolve color name for role (env override wins)
        env_key = f"SOC_COLOR_{role.upper()}"
        color_name = os.getenv(env_key, cls._ROLE_DEFAULTS.get(role, 'white'))
        code = cls._ANSI.get(color_name, '')
        reset = cls._ANSI['reset']
        return f"{code}{text}{reset}" if code else text


class SocratesPipeline:
    """Orchestrates the claim processing pipeline."""

    def __init__(self, llm_manager: any = None, factuality_enabled: bool = None, clarification_enabled: bool = None, clarification_dev_mode: bool = None, question_gen_enabled: bool = None, questions_per_category: int = None, qg_min_threshold: float = None, qg_max_complexity: float = None, qg_enable_fallback: bool = None, qg_prioritize_visual: bool = None):
        """Initializes the pipeline with all necessary components."""
        logging.info("Initializing Socrates Pipeline...")
        self.claim_extractor = ClaimExtractor(llm_manager=llm_manager)
        self.claim_categorizer = ClaimCategorizer(llm_manager=llm_manager)
        self.llm_manager = llm_manager
        
        # Session management for Knowledge Graph / self-consistency
        try:
            self.session_id = os.getenv("SOC_SESSION_ID") or str(uuid.uuid4())
            logging.info(f"Using session_id={self.session_id}")
        except Exception:
            self.session_id = str(uuid.uuid4())
            logging.info(f"Using generated session_id={self.session_id}")
        
        # Initialize Knowledge Graph Manager and session
        try:
            self.kg_manager = KnowledgeGraphManager()
            try:
                self.kg_manager.initialize_session(self.session_id)
            except Exception as e:
                logging.warning(f"Failed to initialize KG session: {e}")
        except Exception as e:
            logging.warning(f"KnowledgeGraphManager unavailable: {e}")
            self.kg_manager = None
        
        # Initialize Self-Contradiction Checker and attach KG
        try:
            self.self_checker = SelfContradictionChecker()
        except Exception as e:
            logging.warning(f"SelfContradictionChecker unavailable: {e}")
            self.self_checker = None
        if getattr(self, "self_checker", None) and getattr(self, "kg_manager", None):
            try:
                self.self_checker.set_kg_manager(self.kg_manager)
            except Exception as e:
                logging.warning(f"Failed attaching KG to SelfContradictionChecker: {e}")
        
        # Initialize Conflict Resolver
        try:
            self.conflict_resolver = ConflictResolver()
        except Exception as e:
            logging.warning(f"ConflictResolver unavailable: {e}")
            self.conflict_resolver = None
        
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

        # Socratic question generation toggle from CLI/env
        if question_gen_enabled is None:
            question_gen_enabled = os.getenv("QUESTION_GEN_ENABLED", "true").lower() == "true"
        self.question_gen_enabled = question_gen_enabled

        # Initialize Socratic Question Generator if enabled
        if self.question_gen_enabled:
            try:
                # Configure capabilities based on available downstream modules
                capabilities = VerificationCapabilities(
                    visual_grounding=[
                        "object_detection",
                        "text_recognition",
                        "scene_description",
                        "spatial_relationships",
                    ],
                    external_knowledge=[
                        "wikipedia",
                        "google_fact_check",
                        "wikidata",
                        "news_articles",
                    ],
                    self_consistency=[
                        "knowledge_graph",
                        "previous_session_claims",
                    ],
                )
                qg_config = SocraticConfig()
                # Apply CLI overrides where provided
                if isinstance(questions_per_category, int) and questions_per_category > 0:
                    qg_config.update(questions_per_category=questions_per_category)
                if isinstance(qg_min_threshold, (int, float)) and qg_min_threshold is not None:
                    qg_config.update(min_confidence_threshold=float(qg_min_threshold))
                if isinstance(qg_max_complexity, (int, float)) and qg_max_complexity is not None:
                    qg_config.update(max_question_complexity_ratio=float(qg_max_complexity))
                if isinstance(qg_enable_fallback, bool):
                    qg_config.update(enable_fallback=bool(qg_enable_fallback))
                if isinstance(qg_prioritize_visual, bool):
                    qg_config.update(prioritize_visual_grounding=bool(qg_prioritize_visual))

                # Apply environment variable overrides (if present)
                env_min = os.getenv("QG_MIN_CONFIDENCE_THRESHOLD")
                if qg_min_threshold is None and env_min is not None:
                    try:
                        qg_config.update(min_confidence_threshold=float(env_min))
                    except Exception:
                        logging.warning("Invalid QG_MIN_CONFIDENCE_THRESHOLD env; ignoring")
                env_max = os.getenv("QG_MAX_COMPLEXITY_RATIO")
                if qg_max_complexity is None and env_max is not None:
                    try:
                        qg_config.update(max_question_complexity_ratio=float(env_max))
                    except Exception:
                        logging.warning("Invalid QG_MAX_COMPLEXITY_RATIO env; ignoring")
                env_fb = os.getenv("QG_ENABLE_FALLBACK")
                if qg_enable_fallback is None and env_fb is not None:
                    qg_config.update(enable_fallback=(env_fb.lower() == "true"))
                env_pv = os.getenv("QG_PRIORITIZE_VISUAL")
                if qg_prioritize_visual is None and env_pv is not None:
                    qg_config.update(prioritize_visual_grounding=(env_pv.lower() == "true"))
                # Questions per category via env (only if CLI not provided)
                if not (isinstance(questions_per_category, int) and questions_per_category > 0):
                    env_qpc = os.getenv("QG_QUESTIONS_PER_CATEGORY")
                    if env_qpc is not None:
                        try:
                            qg_config.update(questions_per_category=int(env_qpc))
                        except Exception:
                            logging.warning("Invalid QG_QUESTIONS_PER_CATEGORY env; ignoring")

                # Use shared LLM manager via adapter
                llm_adapter = LLMInterfaceAdapter(llm_manager=self.llm_manager)
                self.question_generator = SocraticQuestionGenerator(
                    verification_capabilities=capabilities,
                    llm_interface=llm_adapter,
                    config=qg_config,
                )
            except Exception as e:
                logging.warning(f"Question generator unavailable: {e}")
                self.question_generator = None
        else:
            self.question_generator = None
        # Initialize QG stats
        self._qg_stats: Dict[str, Any] = {"total": 0, "fallback": 0}
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
                # Determine if clarification is needed
                amb_cat = any(
                    getattr(c.name, 'name', str(c.name)) == 'AMBIGUOUS_RESOLUTION_REQUIRED'
                    for c in (categorized_claim.categories or [])
                )
                amb_reason = bool(getattr(categorized_claim, 'ambiguity_reason', None))
                no_cats = not bool(categorized_claim.categories)
                should_clarify = amb_cat or amb_reason or no_cats
                if self.clarifier and should_clarify:
                    logging.info(
                        "Pre-routing clarification triggered (reasons: %s)",
                        ", ".join([
                            r for r, ok in [
                                ("AMBIGUOUS_CATEGORY", amb_cat),
                                ("AMBIGUITY_REASON", amb_reason),
                                ("NO_CATEGORIES", no_cats),
                            ] if ok
                        ]) or "UNKNOWN"
                    )
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

            # 2.5 Generate Socratic questions (after clarification/re-categorization, before routing)
            if getattr(self, "question_generator", None):
                try:
                    logging.info("Generating Socratic questions after clarification and before routing...")
                    # Select relevant categories (exclude unverifiable and ambiguity)
                    cat_names = []
                    for c in categorized_claim.categories:
                        try:
                            cname = getattr(c.name, 'name', str(c.name))
                        except Exception:
                            cname = str(getattr(c, 'name', ''))
                        if not cname:
                            continue
                        if cname in ("SUBJECTIVE_OPINION", "PROCEDURAL_DESCRIPTIVE", "AMBIGUOUS_RESOLUTION_REQUIRED"):
                            continue
                        if cname not in cat_names:
                            cat_names.append(cname)
                    if cat_names:
                        # Prioritize visual grounding if configured and present
                        prioritize = None
                        try:
                            if self.question_generator.config.prioritize_visual_grounding and "VISUAL_GROUNDING_REQUIRED" in cat_names:
                                prioritize = "VISUAL_GROUNDING_REQUIRED"
                        except Exception:
                            prioritize = None
                        num_q = getattr(self.question_generator.config, 'questions_per_category', 1) or 1
                        q_results = self.question_generator.generate_questions(
                            categorized_claim.text,
                            categories=cat_names,
                            num_questions=num_q,
                            prioritize_category=prioritize,
                        )
                        # Store as plain dicts for portability
                        sq: Dict[str, List[Dict[str, Any]]] = {}
                        for cat, qs in q_results.items():
                            sq[cat] = [
                                {
                                    "question": q.question,
                                    "category": q.category,
                                    "verification_hint": q.verification_hint,
                                    "confidence_score": q.confidence_score,
                                    "fallback": q.fallback,
                                }
                                for q in qs
                            ]
                        categorized_claim.socratic_questions = sq
                        # Update QG stats
                        try:
                            for _cat, _qs in sq.items():
                                self._qg_stats["total"] += len(_qs)
                                self._qg_stats["fallback"] += sum(1 for _q in _qs if _q.get("fallback"))
                        except Exception:
                            pass
                        logging.info(
                            "Generated %d Socratic question groups for categories: %s",
                            sum(len(v) for v in sq.values()),
                            ", ".join(sq.keys()),
                        )
                    else:
                        logging.info("No eligible categories for question generation; skipping.")
                except Exception as e:
                    logging.warning(f"Question generation failed: {e}")

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
            
            # 5. Self-Consistency (KG) check and evidence-weighted conflict resolution
            if getattr(self, "self_checker", None) and getattr(self, "conflict_resolver", None):
                try:
                    sc_result = self.self_checker.check_contradiction(categorized_claim.text, self.session_id)
                    ext_result = factuality_results.get(i)
                    final_result = self.conflict_resolver.resolve(categorized_claim.text, ext_result, sc_result)
                    # Merge into factuality_results for unified CLI display
                    factuality_results[i] = final_result
                    # Persist onto claim for downstream consumers
                    categorized_claim.factuality_status = final_result.get("status")
                    categorized_claim.factuality_confidence = final_result.get("confidence", 0.0)
                    categorized_claim.factuality_verdict = True if final_result.get("status") == "PASS" else (False if final_result.get("status") == "FAIL" else None)
                    categorized_claim.factuality_evidence = final_result.get("evidence", [])
                    categorized_claim.factuality_sources = final_result.get("sources", [])
                    categorized_claim.factuality_reasoning = final_result.get("reasoning")
                    # Add to KG if recommended
                    if getattr(self, "kg_manager", None) and final_result.get("should_add_to_kg"):
                        try:
                            self.kg_manager.add_claim(
                                claim=categorized_claim.text,
                                evidence=final_result.get("evidence", []),
                                confidence=float(final_result.get("confidence", 0.8)),
                                session_id=self.session_id,
                            )
                            logging.info("Claim added to Knowledge Graph (session)")
                        except Exception as e:
                            logging.warning(f"Failed to add claim to Knowledge Graph: {e}")
                except Exception as e:
                    logging.warning(f"Self-consistency/conflict resolution failed: {e}")

            processed_claims.append(categorized_claim)

        # Summary metrics for factuality stage
        if self.factuality_enabled and factuality_results:
            total = len(factuality_results)
            pass_n = sum(1 for r in factuality_results.values() if r.get("status") == "PASS")
            fail_n = sum(1 for r in factuality_results.values() if r.get("status") == "FAIL")
            uncertain_n = sum(1 for r in factuality_results.values() if r.get("status") == "UNCERTAIN")
            avg_conf = sum(r.get("confidence", 0.0) for r in factuality_results.values()) / max(total, 1)
            logging.info(f"Factuality summary: total={total}, PASS={pass_n}, FAIL={fail_n}, UNCERTAIN={uncertain_n}, avg_conf={avg_conf:.2f}")

        # Summary metrics for question generation stage
        if getattr(self, "question_generator", None):
            try:
                q_total = int(self._qg_stats.get("total", 0))
                q_fallback = int(self._qg_stats.get("fallback", 0))
                if q_total > 0:
                    fb_rate = (q_fallback / q_total) * 100.0
                    logging.info(f"Socratic QG summary: total_questions={q_total}, fallback_used={q_fallback} ({fb_rate:.1f}%)")
            except Exception:
                pass

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
    # Socratic question generation toggles
    parser.add_argument("--enable-question-gen", dest="enable_qg", action="store_true", help="Enable Socratic question generation")
    parser.add_argument("--disable-question-gen", dest="disable_qg", action="store_true", help="Disable Socratic question generation")
    parser.add_argument("--questions-per-category", dest="qg_per_cat", type=int, default=None, help="Number of Socratic questions to generate per relevant category")
    # Socratic question validation tuning
    parser.add_argument("--qg-min-threshold", dest="qg_min_threshold", type=float, default=None, help="Minimum confidence threshold for accepting generated questions (0-1)")
    parser.add_argument("--qg-max-complexity", dest="qg_max_complexity", type=float, default=None, help="Maximum allowed question/claim length ratio before penalization")
    parser.add_argument("--qg-enable-fallback", dest="qg_enable_fallback", action="store_true", help="Enable fallback question templates if validation fails")
    parser.add_argument("--qg-disable-fallback", dest="qg_disable_fallback", action="store_true", help="Disable fallback question templates")
    parser.add_argument("--qg-prioritize-visual", dest="qg_prioritize_visual", action="store_true", help="Prioritize VISUAL_GROUNDING_REQUIRED category when present")
    parser.add_argument("--qg-deprioritize-visual", dest="qg_deprioritize_visual", action="store_true", help="Do not prioritize VISUAL_GROUNDING_REQUIRED category")
    # Knowledge Graph display toggles
    parser.add_argument("--show-kg", dest="show_kg", action="store_true", help="Display session knowledge graph after processing")
    parser.add_argument("--kg-max-items", dest="kg_max_items", type=int, default=None, help="Maximum entities/relations to display for KG output")
    parser.add_argument("--kg-query", dest="kg_query", type=str, default=None, help="Optional free-text query to match entities/relations in the session KG")
    # LLM selection
    parser.add_argument("--llm-provider", dest="llm_provider", type=str, choices=["ollama", "openai", "claude"], default=None, help="LLM provider to use (overrides SOC_LLM_PROVIDER)")
    parser.add_argument("--llm-model", dest="llm_model", type=str, default=None, help="Model name for the selected provider (overrides SOC_LLM_MODEL)")
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

    # Resolve question generation toggle
    env_qg = os.getenv("QUESTION_GEN_ENABLED", "true").lower() == "true"
    if args.enable_qg:
        question_gen_enabled = True
    elif args.disable_qg:
        question_gen_enabled = False
    else:
        question_gen_enabled = env_qg

    # Resolve LLM selection (CLI overrides env)
    selected_provider = args.llm_provider or os.getenv("SOC_LLM_PROVIDER")
    selected_model = args.llm_model or os.getenv("SOC_LLM_MODEL")
    # Instantiate LLMManager with selected provider/model
    llm_manager = LLMManager(model_name=selected_model, provider=selected_provider)
    pipeline = SocratesPipeline(
        llm_manager=llm_manager,
        factuality_enabled=factuality_enabled,
        clarification_enabled=clarification_enabled,
        clarification_dev_mode=clarification_dev_mode,
        question_gen_enabled=question_gen_enabled,
        questions_per_category=args.qg_per_cat,
        qg_min_threshold=args.qg_min_threshold,
        qg_max_complexity=args.qg_max_complexity,
        qg_enable_fallback=True if args.qg_enable_fallback else (False if args.qg_disable_fallback else None),
        qg_prioritize_visual=True if args.qg_prioritize_visual else (False if args.qg_deprioritize_visual else None),
    )

    # Inform about selected LLM
    try:
        print(ConsoleColors.c('label', 'Using LLM: ') + ConsoleColors.c('value', f"{getattr(llm_manager, 'provider').value}:{getattr(llm_manager, 'model_name')}"))
    except Exception:
        pass

    final_claims = pipeline.run(args.text)

    print("\n" + ConsoleColors.c('heading', '--- Final Processed Claims ---'))
    for i, claim in enumerate(final_claims, 1):
        print("\n" + ConsoleColors.c('label', f"Claim {i}: ") + ConsoleColors.c('claim', f"{claim.text}"))
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
        print("  - " + ConsoleColors.c('label', 'Confidence: ') + ConsoleColors.c('value', f"{confidence:.2f}"))
        print("  - " + ConsoleColors.c('label', 'Context: ') + ConsoleColors.c('value', f"{claim.context_window}"))
        print("  - " + ConsoleColors.c('label', 'Entities:'))
        for entity in claim.entities:
            print("    - " + ConsoleColors.c('entity', f"'{entity.text}'") + ConsoleColors.c('label', f" ({entity.label})"))
        if claim.relationships:
            print("  - " + ConsoleColors.c('label', 'Relationships:'))
            for rel in claim.relationships:
                print(ConsoleColors.c('value', f"    - ({rel.subject}) -> ") + ConsoleColors.c('label', f"[{rel.relation}]") + ConsoleColors.c('value', f" -> ({rel.object})"))
        print("  - " + ConsoleColors.c('label', 'Categories:'))
        for category in claim.categories:
            print("    - " + ConsoleColors.c('category', f"{getattr(category.name, 'name', str(category.name))}") +
                  ConsoleColors.c('label', f" (Confidence: ") + ConsoleColors.c('value', f"{category.confidence:.2f}") + ConsoleColors.c('label', "):") +
                  " " + ConsoleColors.c('value', f"{category.justification}"))
        if getattr(claim, 'socratic_questions', None):
            print("  - " + ConsoleColors.c('label', 'Socratic Questions:'))
            try:
                for cat, qs in claim.socratic_questions.items():
                    print("    - " + ConsoleColors.c('category', f"{cat}") + ConsoleColors.c('label', ":"))
                    for q in qs:
                        print("      • " + ConsoleColors.c('question', f"{q.get('question')}") +
                              ConsoleColors.c('label', " (conf ") + ConsoleColors.c('value', f"{q.get('confidence_score', 0.0):.2f}") + ConsoleColors.c('label', ")"))
                        if q.get('verification_hint'):
                            print("        " + ConsoleColors.c('label', 'hint: ') + ConsoleColors.c('value', f"{q.get('verification_hint')}"))
            except Exception:
                pass
        if claim.verification_route:
            route = claim.verification_route
            print("  - " + ConsoleColors.c('label', 'Verification Route: ') + ConsoleColors.c('route', f"{route.method.name}"))
            print("    - " + ConsoleColors.c('label', 'Justification: ') + ConsoleColors.c('value', f"{route.justification}"))
            print("    - " + ConsoleColors.c('label', 'Estimated Cost: ') + ConsoleColors.c('value', f"{route.estimated_cost}") +
                  ConsoleColors.c('label', ', Latency: ') + ConsoleColors.c('value', f"{route.estimated_latency}s"))
        # Clarification results (if available)
        clr = getattr(pipeline, "_clarification_results", {}).get(i)
        if clr:
            print("  - " + ConsoleColors.c('clarification', 'Clarifications:'))
            pre = clr.get("pre")
            post = clr.get("post")
            if pre:
                print("    - " + ConsoleColors.c('label', 'Pre-routing:'))
                print("      " + ConsoleColors.c('label', 'corrected: ') + ConsoleColors.c('value', f"{getattr(pre, 'corrected_claim', None)}"))
                print("      " + ConsoleColors.c('label', 'confidence: ') + ConsoleColors.c('value', f"{getattr(pre, 'resolution_confidence', 0.0):.2f}"))
                print("      " + ConsoleColors.c('label', 'next_action: ') + ConsoleColors.c('clarification', f"{getattr(getattr(pre, 'next_action', ''), 'name', str(getattr(pre, 'next_action', '')))}"))
            if post:
                print("    - " + ConsoleColors.c('label', 'Post-factuality:'))
                print("      " + ConsoleColors.c('label', 'corrected: ') + ConsoleColors.c('value', f"{getattr(post, 'corrected_claim', None)}"))
                print("      " + ConsoleColors.c('label', 'confidence: ') + ConsoleColors.c('value', f"{getattr(post, 'resolution_confidence', 0.0):.2f}"))
                print("      " + ConsoleColors.c('label', 'next_action: ') + ConsoleColors.c('clarification', f"{getattr(getattr(post, 'next_action', ''), 'name', str(getattr(post, 'next_action', '')))}"))
        # Factuality result (if available)
        fr = getattr(pipeline, "_last_factuality_results", {}).get(i)
        if fr:
            print("  - " + ConsoleColors.c('label', 'Factuality:'))
            status = fr.get('status')
            status_role = 'factuality_pass' if status == 'PASS' else ('factuality_fail' if status == 'FAIL' else 'factuality_uncertain')
            print("    - " + ConsoleColors.c('label', 'Status: ') + ConsoleColors.c(status_role, f"{status}") +
                  ConsoleColors.c('label', ', Confidence: ') + ConsoleColors.c('value', f"{fr.get('confidence', 0.0):.2f}"))
            verdict = (True if fr.get('status') == 'PASS' else (False if fr.get('status') == 'FAIL' else None))
            print("    - " + ConsoleColors.c('label', 'Verdict: ') + ConsoleColors.c('value', f"{verdict}"))
            if fr.get('reasoning'):
                print("    - " + ConsoleColors.c('label', 'Reasoning: ') + ConsoleColors.c('value', f"{fr.get('reasoning')}"))
            if fr.get("external_facts"):
                print("    - " + ConsoleColors.c('label', 'External Facts: ') + ConsoleColors.c('value', f"{fr['external_facts'][:2]}"))
            elif fr.get("evidence"):
                print("    - " + ConsoleColors.c('label', 'Evidence: ') + ConsoleColors.c('value', f"{fr['evidence'][:2]}"))
            if fr.get("contradictions"):
                print("    - " + ConsoleColors.c('label', 'Contradictions: ') + ConsoleColors.c('value', f"{fr['contradictions'][:2]}"))
            if fr.get("sources"):
                print("    - " + ConsoleColors.c('label', 'Sources: ') + ConsoleColors.c('value', f"{fr['sources'][:3]}"))

    # Summary metrics (also logged above)
    if getattr(pipeline, "_last_factuality_results", None):
        frs = pipeline._last_factuality_results
        total = len(frs)
        pass_n = sum(1 for r in frs.values() if r.get("status") == "PASS")
        fail_n = sum(1 for r in frs.values() if r.get("status") == "FAIL")
        uncertain_n = sum(1 for r in frs.values() if r.get("status") == "UNCERTAIN")
        avg_conf = sum(r.get("confidence", 0.0) for r in frs.values()) / max(total, 1)
        print("\n" + ConsoleColors.c('summary', '--- Factuality Summary ---'))
        print(ConsoleColors.c('label', 'Total checked: ') + ConsoleColors.c('value', f"{total}") +
              ConsoleColors.c('label', ' | PASS: ') + ConsoleColors.c('factuality_pass', f"{pass_n}") +
              ConsoleColors.c('label', ' | FAIL: ') + ConsoleColors.c('factuality_fail', f"{fail_n}") +
              ConsoleColors.c('label', ' | UNCERTAIN: ') + ConsoleColors.c('factuality_uncertain', f"{uncertain_n}") +
              ConsoleColors.c('label', ' | Avg conf: ') + ConsoleColors.c('value', f"{avg_conf:.2f}"))
    # Socratic QG Summary
    if getattr(pipeline, "_qg_stats", None):
        q_total = int(pipeline._qg_stats.get("total", 0))
        q_fallback = int(pipeline._qg_stats.get("fallback", 0))
        if q_total > 0:
            fb_rate = (q_fallback / q_total) * 100.0
            print("\n" + ConsoleColors.c('summary', '--- Socratic Question Generation Summary ---'))
            print(ConsoleColors.c('label', 'Total questions: ') + ConsoleColors.c('value', f"{q_total}") +
                  ConsoleColors.c('label', ' | Fallback used: ') + ConsoleColors.c('value', f"{q_fallback}") +
                  ConsoleColors.c('label', ' (') + ConsoleColors.c('value', f"{fb_rate:.1f}") + ConsoleColors.c('label', '%)'))

    # Knowledge Graph display (toggle via CLI or env SOC_SHOW_KG=true)
    env_show_kg = os.getenv("SOC_SHOW_KG", "false").lower() == "true"
    show_kg = True if args.show_kg else env_show_kg
    if show_kg and getattr(pipeline, "kg_manager", None):
        try:
            print("\n" + ConsoleColors.c('summary', '--- Session Knowledge Graph ---'))
            print(ConsoleColors.c('label', 'Session: ') + ConsoleColors.c('value', f"{pipeline.session_id}"))
            kg_export = pipeline.kg_manager.export_session_graph(pipeline.session_id) or {}
            stats = kg_export.get('statistics', {"nodes": 0, "edges": 0})
            print(ConsoleColors.c('label', 'Entities: ') + ConsoleColors.c('value', f"{stats.get('nodes', 0)}") +
                  ConsoleColors.c('label', ' | Relations: ') + ConsoleColors.c('value', f"{stats.get('edges', 0)}"))

            # Optional structured query over KG
            if args.kg_query:
                qres = pipeline.kg_manager.query_knowledge_graph(args.kg_query, pipeline.session_id)
                print(ConsoleColors.c('label', 'Query: ') + ConsoleColors.c('value', f"{args.kg_query}"))
                print(ConsoleColors.c('label', 'Matched entities: ') + ConsoleColors.c('value', f"{', '.join(qres.get('query_entities', []) or [])}"))
                if qres.get('results'):
                    max_items = args.kg_max_items or int(os.getenv("SOC_KG_MAX_ITEMS", "10"))
                    for idx, item in enumerate(qres['results'][:max_items], 1):
                        ent = item.get('entity', {})
                        print("  - " + ConsoleColors.c('entity', f"{ent.get('text', ent.get('id', ''))}") +
                              ConsoleColors.c('label', f" ({ent.get('label', '')}) ") +
                              ConsoleColors.c('label', 'connections: ') + ConsoleColors.c('value', f"{item.get('connections', 0)}"))
                        rels = item.get('relations', [])
                        for r in rels[:max_items]:
                            print("      • " + ConsoleColors.c('value', f"{r.get('target')} ") + ConsoleColors.c('label', f"[{r.get('relation')}]"))
                else:
                    print(ConsoleColors.c('label', 'Query results: ') + ConsoleColors.c('value', 'none'))
            else:
                # Compact dump of nodes and edges
                nodes = kg_export.get('nodes', {})
                edges = kg_export.get('edges', [])
                id_to_text = {nid: (ndata.get('text') or nid) for nid, ndata in nodes.items()}
                max_items = args.kg_max_items or int(os.getenv("SOC_KG_MAX_ITEMS", "10"))
                if nodes:
                    print(ConsoleColors.c('label', f"Entities (showing up to {max_items}):"))
                    for i, (nid, ndata) in enumerate(list(nodes.items())[:max_items], 1):
                        txt = ndata.get('text', nid)
                        lbl = ndata.get('label', '')
                        conf = ndata.get('confidence', 0.0)
                        print(f"  - {ConsoleColors.c('entity', txt)}{ConsoleColors.c('label', f' ({lbl}) ')}" +
                              ConsoleColors.c('label', 'conf: ') + ConsoleColors.c('value', f"{conf:.2f}") +
                              ConsoleColors.c('label', ' | id: ') + ConsoleColors.c('value', f"{nid}"))
                if edges:
                    print(ConsoleColors.c('label', f"Relations (showing up to {max_items}):"))
                    for e in edges[:max_items]:
                        try:
                            src, dst, data = e
                            pred = data.get('predicate', 'related_to')
                            print("  - " + ConsoleColors.c('entity', f"{id_to_text.get(src, src)}") +
                                  ConsoleColors.c('label', f" -> [{pred}] -> ") +
                                  ConsoleColors.c('entity', f"{id_to_text.get(dst, dst)}"))
                        except Exception:
                            pass
        except Exception as e:
            print(ConsoleColors.c('label', 'KG display error: ') + ConsoleColors.c('value', f"{e}"))
