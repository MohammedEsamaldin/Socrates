"""
Integration pipeline for the Socrates Agent.

This script demonstrates how to use the claim extractor, categorizer, and router
modules together to process a piece of text and prepare claims for verification.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

_IMPORT_EXCEPTION = None  # defer hard failure until runtime (except for --help)
try:
    from socrates_system.modules.claim_extractor import ClaimExtractor
    from socrates_system.modules.claim_categorizer import ClaimCategorizer
    from socrates_system.modules.check_router import CheckRouter
    from socrates_system.modules.deterministic_router import DeterministicRouter
    from socrates_system.modules.llm_manager import LLMManager
    from socrates_system.modules.shared_structures import ExtractedClaim, VerificationMethod, VerificationRoute
    from socrates_system.modules.external_factuality_checker import ExternalFactualityChecker
    from socrates_system.modules.knowledge_graph_manager import KnowledgeGraphManager
    from socrates_system.modules.self_contradiction_checker import SelfContradictionChecker
    from socrates_system.modules.conflict_resolver import ConflictResolver
except Exception as _e:
    _IMPORT_EXCEPTION = _e
import os
import argparse
import uuid
try:
    from socrates_system.clarification_resolution import ClarificationResolutionModule
    from socrates_system.clarification_resolution.data_models import (
        ClarificationContext as ClarContext,
        FactCheckResult as ClarFactCheckResult,
        IssueType,
    )
except Exception:
    ClarificationResolutionModule = None
    ClarContext = None
    ClarFactCheckResult = None
    IssueType = None

# Socratic Question Generator
try:
    from socrates_system.modules.question_generator import (
        SocraticQuestionGenerator,
        SocraticConfig,
        VerificationCapabilities,
        LLMInterfaceAdapter,
    )
except Exception:
    SocraticQuestionGenerator = None
    SocraticConfig = None
    VerificationCapabilities = None
    LLMInterfaceAdapter = None

# Cross-Alignment Checker (advanced preferred, fallback to simplified)
try:
    from socrates_system.modules.cross_alignment_checker import (
        CrossAlignmentChecker as AdvancedCrossAlignmentChecker,
    )
except Exception:
    AdvancedCrossAlignmentChecker = None
try:
    from socrates_system.modules.cross_alignment_checker_simple import (
        CrossAlignmentChecker as SimpleCrossAlignmentChecker,
    )
except Exception:
    SimpleCrossAlignmentChecker = None

# Remote AGLA API client (preferred for cross-modal when configured)
try:
    from socrates_system.modules.agla_client import AGLAClient
    from socrates_system.config import (
        AGLA_API_URL,
        AGLA_API_VERIFY_PATH,
        AGLA_API_TIMEOUT,
    )
except Exception:
    AGLAClient = None
    AGLA_API_URL = None
    AGLA_API_VERIFY_PATH = None
    AGLA_API_TIMEOUT = None

# --- Utility: conservative negation detection to guard polarity flips ---
def _has_negation(text: Optional[str]) -> bool:
    if not text:
        return False
    s = f" {str(text).strip().lower()} "
    neg_terms = [
        " not ", " no ", " never ", " none ", " without ", " nothing ", " nowhere ",
        "n't ", "n't,", "n't.", "n't?",
    ]
    return any(t in s for t in neg_terms)

# Setup basic logging for SocratesPipeline; honor SOC_LOG_LEVEL (e.g., DEBUG, INFO)
_lvl_name = os.getenv('SOC_LOG_LEVEL', 'INFO').upper()
_lvl = getattr(logging, _lvl_name, logging.INFO)
logging.basicConfig(level=_lvl, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def __init__(self, llm_manager: any = None, factuality_enabled: bool = None, clarification_enabled: bool = None, clarification_dev_mode: bool = None, question_gen_enabled: bool = None, questions_per_category: int = None, qg_min_threshold: float = None, qg_max_complexity: float = None, qg_enable_fallback: bool = None, qg_prioritize_visual: bool = None, conflict_resolution_mode: str = None, factuality_context_mode: Optional[str] = None, factuality_context_max_items: Optional[int] = None, router_mode: Optional[str] = None, post_factuality_clarification_enabled: Optional[bool] = None):
        """Initializes the pipeline with all necessary components."""
        logging.info("Initializing Socrates Pipeline...")
        self.claim_extractor = ClaimExtractor(llm_manager=llm_manager)
        self.claim_categorizer = ClaimCategorizer(llm_manager=llm_manager)
        self.llm_manager = llm_manager
        # Prepare per-module LLMs (env overrides -> new instance, else fallback to shared)
        self.llm_factual = self._make_module_llm(
            os.getenv("FACTUALITY_LLM_PROVIDER"),
            os.getenv("FACTUALITY_LLM_MODEL"),
            llm_manager,
        )
        self.llm_self = self._make_module_llm(
            os.getenv("SELF_LLM_PROVIDER"),
            os.getenv("SELF_LLM_MODEL"),
            llm_manager,
        )
        
        # Session management for Knowledge Graph / self-consistency
        try:
            self.session_id = os.getenv("SOC_SESSION_ID") or str(uuid.uuid4())
            logging.info(f"Using session_id={self.session_id}")
        except Exception:
            self.session_id = str(uuid.uuid4())
            logging.info(f"Using generated session_id={self.session_id}")
        
        # Initialize Knowledge Graph Manager and session
        try:
            self.kg_manager = KnowledgeGraphManager(llm_manager=llm_manager)
            try:
                self.kg_manager.initialize_session(self.session_id)
            except Exception as e:
                logging.warning(f"Failed to initialize KG session: {e}")
        except Exception as e:
            logging.warning(f"KnowledgeGraphManager unavailable: {e}")
            self.kg_manager = None
        
        # Attach KG manager to ClaimExtractor for canonical ID resolution
        if getattr(self, "kg_manager", None) and getattr(self, "claim_extractor", None):
            try:
                self.claim_extractor.set_kg_manager(self.kg_manager, self.session_id)
            except Exception as e:
                logging.warning(f"Failed attaching KG to ClaimExtractor: {e}")
        
        # Initialize Self-Contradiction Checker and attach KG
        try:
            self.self_checker = SelfContradictionChecker(llm_manager=self.llm_self)
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
        # Resolve conflict resolution mode (CLI/env)
        try:
            crm_raw = (conflict_resolution_mode or os.getenv("SOC_CONFLICT_MODE") or "auto")
            crm_l = str(crm_raw).strip().lower()
            if crm_l in ("manual", "man", "interactive"):
                self.conflict_resolution_mode = "manual"
            else:
                self.conflict_resolution_mode = "auto"
        except Exception:
            self.conflict_resolution_mode = "auto"
        
        # Initialize with a set of available verification methods
        available_methods = {
            VerificationMethod.KNOWLEDGE_GRAPH,
            VerificationMethod.EXTERNAL_SOURCE,
            VerificationMethod.CALCULATION,
            VerificationMethod.EXPERT_VERIFICATION,
            VerificationMethod.DEFINITIONAL,
            VerificationMethod.CROSS_MODAL,
        }
        # Resolve router mode (CLI/env) and initialize routers
        try:
            rm_raw = (router_mode or os.getenv("SOC_ROUTER_MODE") or "llm")
            rm_l = str(rm_raw).strip().lower()
            if rm_l in ("det", "deterministic"):
                self.router_mode = "deterministic"
            elif rm_l in ("hybrid", "mix", "mixed"):
                self.router_mode = "hybrid"
            else:
                self.router_mode = "llm"
        except Exception:
            self.router_mode = "llm"
        self.check_router = CheckRouter(available_methods=available_methods)
        try:
            self.det_router = DeterministicRouter(
                available_methods=available_methods,
                kg_manager=self.kg_manager,
                session_id=self.session_id,
            )
        except Exception as e:
            logging.warning(f"DeterministicRouter unavailable: {e}")
            self.det_router = None
        try:
            logging.info(f"Router mode selected: {self.router_mode}")
        except Exception:
            pass

        # Initialize Cross-Alignment checker
        self.cross_checker = None
        try:
            if AdvancedCrossAlignmentChecker is not None:
                self.cross_checker = AdvancedCrossAlignmentChecker()
                logging.info("Using Advanced Cross-Alignment Checker")
            else:
                self.cross_checker = SimpleCrossAlignmentChecker()
                logging.info("Using Simplified Cross-Alignment Checker")
        except Exception as e:
            logging.warning(f"CrossAlignmentChecker unavailable: {e}")
            self.cross_checker = None

        # Initialize remote AGLA API client if configured (preferred for cross-modal)
        self.agla_client = None
        try:
            if AGLA_API_URL:
                self.agla_client = AGLAClient(AGLA_API_URL, AGLA_API_VERIFY_PATH, AGLA_API_TIMEOUT)
                logging.info(f"AGLA remote API configured: {AGLA_API_URL}{AGLA_API_VERIFY_PATH}")
        except Exception as e:
            logging.warning(f"AGLAClient initialization failed: {e}")
            self.agla_client = None

        # Factuality check toggle from CLI/env
        if factuality_enabled is None:
            factuality_enabled = os.getenv("FACTUALITY_ENABLED", "true").lower() == "true"
        self.factuality_enabled = factuality_enabled
        self.external_checker = ExternalFactualityChecker(llm_manager=self.llm_factual) if self.factuality_enabled else None
        # Configure input context for external factuality LLM aggregation
        try:
            mode_raw = (factuality_context_mode or os.getenv("FACTUALITY_CONTEXT_MODE") or "SOCRATIC_QUESTIONS")
            mode_up = str(mode_raw).strip().upper()
            if mode_up in ("SOCRATIC", "SOCRATIC_QUESTIONS"):
                norm = "SOCRATIC_QUESTIONS"
            elif mode_up in ("CLAIMS", "EXTRACTED_CLAIMS"):
                norm = "EXTRACTED_CLAIMS"
            elif mode_up in ("NONE", "DISABLED"):
                norm = "NONE"
            else:
                norm = "SOCRATIC_QUESTIONS"
            self.factuality_context_mode = norm
        except Exception:
            self.factuality_context_mode = "SOCRATIC_QUESTIONS"
        try:
            self.factuality_context_max_items = int(
                factuality_context_max_items if factuality_context_max_items is not None else os.getenv("FACTUALITY_CONTEXT_MAX_ITEMS", 6)
            )
        except Exception:
            self.factuality_context_max_items = 6
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
        # Resolve post-factuality clarifications toggle (defaults to enabled)
        try:
            if post_factuality_clarification_enabled is None:
                pf_raw = os.getenv("SOC_POST_FACTUALITY_CLAR", "true")
                self.post_factuality_clarification_enabled = str(pf_raw).strip().lower() == "true"
            else:
                self.post_factuality_clarification_enabled = bool(post_factuality_clarification_enabled)
            logging.info(f"Post-factuality clarifications: {'ENABLED' if self.post_factuality_clarification_enabled else 'DISABLED'}")
        except Exception:
            self.post_factuality_clarification_enabled = True
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
                if qg_enable_fallback is None and env_fb is not None:
                    try:
                        qg_config.update(enable_fallback=(str(env_fb).strip().lower() == 'true'))
                    except Exception:
                        logging.warning("Invalid QG_ENABLE_FALLBACK env; ignoring")
                # Build adapter and generator
                adapter = LLMInterfaceAdapter(self.llm_manager)
                self.question_generator = SocraticQuestionGenerator(qg_config, capabilities, adapter)
                self._qg_stats: Dict[str, int] = {"total": 0, "fallback": 0}
            except Exception as e:
                logging.warning(f"SocraticQuestionGenerator unavailable: {e}")
                self.question_generator = None
                self._qg_stats = {"total": 0, "fallback": 0}
        else:
            self.question_generator = None
            self._qg_stats = {"total": 0, "fallback": 0}

    def _route_claim(self, claim: ExtractedClaim) -> VerificationRoute:
        """
        Route a claim according to the configured router mode.
        Modes:
          - 'llm': use CheckRouter
          - 'deterministic': use DeterministicRouter
          - 'hybrid': run both and pick best by heuristic scoring
        """
        try:
            mode = getattr(self, "router_mode", "llm")
        except Exception:
            mode = "llm"

        llm_route: Optional[VerificationRoute] = None
        det_route: Optional[VerificationRoute] = None

        # Route according to mode (avoid unnecessary invocations)
        if mode == "llm":
            try:
                llm_route = self.check_router.route_claim(claim)
            except Exception as e:
                logging.warning(f"LLM router failed: {e}")
                if getattr(self, "det_router", None):
                    try:
                        det_route = self.det_router.route_claim(claim)
                    except Exception as e2:
                        logging.warning(f"Deterministic router (fallback) failed: {e2}")
            route = llm_route or det_route
        elif mode == "deterministic":
            if getattr(self, "det_router", None):
                try:
                    det_route = self.det_router.route_claim(claim)
                except Exception as e:
                    logging.warning(f"Deterministic router failed: {e}")
            if det_route is None:
                try:
                    llm_route = self.check_router.route_claim(claim)
                except Exception as e:
                    logging.warning(f"LLM router (fallback) failed: {e}")
            route = det_route or llm_route
        else:
            # Hybrid mode: run both and choose
            try:
                llm_route = self.check_router.route_claim(claim)
            except Exception as e:
                logging.warning(f"LLM router failed: {e}")
            if getattr(self, "det_router", None):
                try:
                    det_route = self.det_router.route_claim(claim)
                except Exception as e:
                    logging.warning(f"Deterministic router failed: {e}")

            def score(r: Optional[VerificationRoute]) -> float:
                if r is None:
                    return 0.0
                base = float(getattr(r, "confidence", 0.0) or 0.0)
                bonus = 0.0
                md = getattr(r, "metadata", {}) or {}
                # Prefer KG when contradictions detected or good coverage
                try:
                    if md.get("kg_contradiction_detected"):
                        bonus += 0.25
                    elif md.get("contradiction_status") == "FAIL" or int(md.get("contradictions_count", 0) or 0) > 0:
                        bonus += 0.25
                except Exception:
                    pass
                try:
                    cov = float(md.get("kg_coverage_ratio", md.get("kg_coverage", 0.0)) or 0.0)
                    if cov >= 0.5:
                        bonus += 0.15
                except Exception:
                    pass
                # Respect route hints and vision flags
                try:
                    hint = getattr(claim, "route_hint", None)
                    if hint and isinstance(hint, str):
                        h = hint.strip().upper()
                        if h.startswith("KG") and r.method == VerificationMethod.KNOWLEDGE_GRAPH:
                            bonus += 0.2
                        elif h.startswith("EXTERNAL") and r.method == VerificationMethod.EXTERNAL_SOURCE:
                            bonus += 0.2
                        elif h.startswith("VISUAL") and r.method == VerificationMethod.CROSS_MODAL:
                            bonus += 0.2
                except Exception:
                    pass
                try:
                    if getattr(claim, "vision_flag", False) and r.method == VerificationMethod.CROSS_MODAL:
                        bonus += 0.15
                except Exception:
                    pass
                # Category-based nudges
                try:
                    cat_names = [getattr(c.name, 'name', str(c.name)) for c in (claim.categories or [])]
                    if "SELF_CONSISTENCY_REQUIRED" in cat_names and r.method == VerificationMethod.KNOWLEDGE_GRAPH:
                        bonus += 0.2
                    if "EXTERNAL_KNOWLEDGE_REQUIRED" in cat_names and r.method == VerificationMethod.EXTERNAL_SOURCE:
                        bonus += 0.2
                    if "VISUAL_GROUNDING_REQUIRED" in cat_names and r.method == VerificationMethod.CROSS_MODAL:
                        bonus += 0.2
                except Exception:
                    pass
                return max(0.0, min(1.0, base + bonus))

            s_llm = score(llm_route)
            s_det = score(det_route)
            route = det_route if s_det >= s_llm else llm_route

        # Fallback if both routers failed
        if route is None:
            route = VerificationRoute(
                method=VerificationMethod.EXTERNAL_SOURCE,
                confidence=0.5,
                justification="Defaulted due to router failure",
                estimated_cost=1.0,
                estimated_latency=1.0,
                metadata={"router_mode": mode, "fallback": "default"},
            )

        # Annotate metadata with selection info
        try:
            md = dict(getattr(route, "metadata", {}) or {})
            md.setdefault("router_mode", mode)
            if mode == "hybrid":
                md.setdefault("selected_by", "hybrid")
            route.metadata = md
        except Exception:
            pass
        return route

    def _make_module_llm(self, provider_env: Optional[str], model_env: Optional[str], fallback_llm: Optional[LLMManager]) -> Optional[LLMManager]:
        """Create a module-specific LLMManager if env overrides are provided; else return fallback.
        provider_env: e.g., 'ollama' | 'openai' | 'claude' (case-insensitive)
        model_env: model name string for the selected provider
        """
        try:
            prov = (provider_env or "").strip().lower()
            mdl = (model_env or "").strip()
            if prov or mdl:
                # New instance dedicated to the module
                return LLMManager(provider=prov or None, model_name=mdl or None)
        except Exception as e:
            logging.warning(f"Failed to create module-specific LLMManager (provider={provider_env}, model={model_env}): {e}")
        return fallback_llm
    

    def _manual_resolve_conflict(self, claim: str, external_result: Optional[Dict[str, Any]], self_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Interactive manual conflict resolution in CLI.

        Falls back to auto resolution if stdin is not a TTY or on error.
        """
        try:
            # Use programmatic resolver to suggest defaults if available
            suggested = None
            if getattr(self, "conflict_resolver", None):
                try:
                    suggested = self.conflict_resolver.resolve(claim, external_result, self_result)
                except Exception:
                    suggested = None
            if suggested is None:
                # naive fallback suggestion
                base = external_result or self_result or {}
                suggested = {
                    "status": base.get("status", "UNCERTAIN"),
                    "confidence": float(base.get("confidence", 0.1) or 0.1),
                    "reasoning": base.get("reasoning", "Manual decision (no suggestion)"),
                    "sources": base.get("sources", []),
                    "contradictions": (self_result or {}).get("contradictions", []),
                    "evidence": (external_result or {}).get("evidence") or (external_result or {}).get("external_facts") or (self_result or {}).get("evidence") or [],
                    "should_add_to_kg": False,
                }

            try:
                is_tty = sys.stdin.isatty()
            except Exception:
                is_tty = False
            if not is_tty:
                logging.warning("Manual conflict mode requested, but no interactive TTY detected; using auto suggestion.")
                return suggested

            print("\n" + ConsoleColors.c('heading', '--- Manual Conflict Resolution ---'))
            print(ConsoleColors.c('label', 'Claim: ') + ConsoleColors.c('claim', f"{claim}"))
            # External factuality summary
            if external_result:
                print(ConsoleColors.c('label', 'External factuality: ') + ConsoleColors.c('value', f"{external_result.get('status')} (conf {float(external_result.get('confidence', 0.0)):.2f})"))
                if external_result.get('reasoning'):
                    print("  - " + ConsoleColors.c('label', 'reasoning: ') + ConsoleColors.c('value', f"{external_result.get('reasoning')}"))
                if external_result.get('evidence') or external_result.get('external_facts'):
                    ev = external_result.get('evidence') or external_result.get('external_facts')
                    try:
                        print("  - " + ConsoleColors.c('label', 'evidence: ') + ConsoleColors.c('value', f"{ev[:2]}"))
                    except Exception:
                        pass
            else:
                print(ConsoleColors.c('label', 'External factuality: ') + ConsoleColors.c('value', 'None'))
            # Self-consistency summary
            if self_result:
                print(ConsoleColors.c('label', 'Self-consistency: ') + ConsoleColors.c('value', f"{self_result.get('status')} (conf {float(self_result.get('confidence', 0.0)):.2f})"))
                if self_result.get('contradictions'):
                    print("  - " + ConsoleColors.c('label', 'contradictions: ') + ConsoleColors.c('value', f"{self_result.get('contradictions')[:2]}"))
                if self_result.get('reasoning'):
                    print("  - " + ConsoleColors.c('label', 'reasoning: ') + ConsoleColors.c('value', f"{self_result.get('reasoning')}"))
            else:
                print(ConsoleColors.c('label', 'Self-consistency: ') + ConsoleColors.c('value', 'None'))

            # Prompt user for final decision
            default_status = (suggested.get('status') or 'UNCERTAIN').upper()
            default_conf = float(suggested.get('confidence', 0.5) or 0.5)
            default_add = bool(suggested.get('should_add_to_kg', False))
            try:
                raw_status = input(ConsoleColors.c('label', f"Final status [PASS/FAIL/UNCERTAIN] (default {default_status}): "))
            except EOFError:
                raw_status = ''
            status = (raw_status or default_status).strip().upper()
            if status not in ("PASS", "FAIL", "UNCERTAIN"):
                status = default_status
            try:
                raw_conf = input(ConsoleColors.c('label', f"Confidence 0-1 (default {default_conf:.2f}): "))
            except EOFError:
                raw_conf = ''
            try:
                conf = float(raw_conf.strip()) if raw_conf.strip() else default_conf
            except Exception:
                conf = default_conf
            conf = max(0.0, min(1.0, conf))
            try:
                raw_add = input(ConsoleColors.c('label', f"Add to KG? [y/N] (default {'y' if default_add else 'n'}): "))
            except EOFError:
                raw_add = ''
            yn = (raw_add or ("y" if default_add else "n")).strip().lower()
            should_add = yn.startswith('y')
            try:
                raw_reason = input(ConsoleColors.c('label', "Optional reasoning note (press Enter to skip): "))
            except EOFError:
                raw_reason = ''
            reason = raw_reason.strip() or (suggested.get('reasoning') or 'Manual decision based on evidence review.')

            # Aggregate fields
            evidence = []
            for e in ((external_result or {}).get('evidence') or (external_result or {}).get('external_facts') or []):
                if e not in evidence:
                    evidence.append(e)
            for e in ((self_result or {}).get('evidence') or []):
                if e not in evidence:
                    evidence.append(e)
            sources = (external_result or {}).get('sources', []) or []
            contradictions = (self_result or {}).get('contradictions', []) or []

            return {
                "status": status,
                "confidence": conf,
                "reasoning": f"Manual: {reason}",
                "sources": sources,
                "contradictions": contradictions,
                "evidence": evidence,
                "should_add_to_kg": (status == "PASS" and should_add and not contradictions),
            }
        except Exception as e:
            logging.warning(f"Manual conflict resolution failed, using auto suggestion: {e}")
            if getattr(self, "conflict_resolver", None):
                try:
                    return self.conflict_resolver.resolve(claim, external_result, self_result)
                except Exception:
                    pass
            return {
                "status": "UNCERTAIN",
                "confidence": 0.0,
                "reasoning": f"Exception during manual resolution: {e}",
                "sources": [],
                "contradictions": [],
                "evidence": [],
                "should_add_to_kg": False,
            }

    def run(self, text: str, image_path: Optional[str] = None) -> List[ExtractedClaim]:
        """
        Runs the full claim processing pipeline on a given text.

        Args:
            text: The input text to process.
            image_path: Optional path to an image for multimodal verification.

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
                        proposed = clr_res.corrected_claim.strip()
                        allow_flip_env = os.getenv("SOC_ALLOW_POLARITY_FLIP", "false").lower() == "true"
                        if (not allow_flip_env) and (_has_negation(categorized_claim.text) != _has_negation(proposed)):
                            logging.info("Skipping pre-route correction due to polarity flip guard")
                        else:
                            logging.info("Applying corrected claim from clarification (pre-route)")
                            categorized_claim.text = proposed
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

            # 3. Route each claim for verification (mode-aware)
            route = self._route_claim(categorized_claim)
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

            # 4. Cross-modal alignment check when visual grounding is required
            if route.method == VerificationMethod.CROSS_MODAL:
                stage_name = getattr(self, "_verification_stage", "UNKNOWN")
                logging.debug(f"[XMOD {stage_name}] Starting cross-modal check | claim: {categorized_claim.text}")
                logging.info("Performing cross-modal alignment check for this claim...")
                # Colored, explicit log of the claim being sent to cross-modal
                try:
                    logging.info(
                        ConsoleColors.c('heading', f"[XMOD {stage_name}] ")
                        + ConsoleColors.c('label', 'Claim: ')
                        + ConsoleColors.c('value', f"{categorized_claim.text}")
                    )
                except Exception:
                    logging.info(f"[XMOD {stage_name}] Claim: {categorized_claim.text}")
                try:
                    if not getattr(self, "cross_checker", None) and not getattr(self, "agla_client", None):
                        raise RuntimeError("No cross-modal verifier available (AGLA/local)")
                    if not image_path:
                        result = {
                            "status": "UNCERTAIN",
                            "confidence": 0.0,
                            "evidence": [],
                            "sources": [],
                            "contradictions": [],
                            "reasoning": "No image provided for cross-modal verification",
                        }
                    else:
                        # Prefer remote AGLA API when available; fallback to local checker
                        if getattr(self, "agla_client", None) is not None:
                            try:
                                # Pick the highest-confidence visual Socratic question, if available
                                soc_q = None
                                try:
                                    sq_map = getattr(categorized_claim, "socratic_questions", {}) or {}
                                    vis_list = sq_map.get("VISUAL_GROUNDING_REQUIRED") or []
                                    if vis_list:
                                        soc_q = max(
                                            vis_list,
                                            key=lambda d: float(d.get("confidence_score", 0.0) or 0.0),
                                        ).get("question")
                                except Exception:
                                    pass
                                # Log the chosen verifier path and Socratic question
                                try:
                                    logging.info(ConsoleColors.c('label', 'Verifier: ') + ConsoleColors.c('value', 'AGLA'))
                                    if soc_q:
                                        logging.info(
                                            ConsoleColors.c('label', 'Socratic Q: ')
                                            + ConsoleColors.c('question', f"{soc_q}")
                                        )
                                except Exception:
                                    logging.info("Verifier: AGLA")
                                out = self.agla_client.verify(
                                    image=image_path,
                                    claim=categorized_claim.text,
                                    socratic_question=soc_q,
                                    return_debug=False,
                                )
                                verdict_str = (out or {}).get("verdict", "Uncertain")
                                if verdict_str == "True":
                                    _status, _conf = "PASS", 0.85
                                elif verdict_str == "False":
                                    _status, _conf = "FAIL", 0.85
                                else:
                                    _status, _conf = "UNCERTAIN", 0.5
                                _truth = (out or {}).get("truth") or ""
                                ev = []
                                # if _status == "FAIL" and _truth:
                                if _truth:
                                    ev.append(f"AGLA evidence: {_truth}")
                                ev.append(f"AGLA verdict: {verdict_str}")
                                srcs = []
                                try:
                                    if AGLA_API_URL:
                                        srcs.append(f"{AGLA_API_URL}{AGLA_API_VERIFY_PATH}")
                                except Exception:
                                    pass
                                result = {
                                    "status": _status,
                                    "confidence": _conf,
                                    "evidence": ev,
                                    "sources": srcs,
                                    "agla_truth": _truth,
                                    "contradictions": [] if _status != "FAIL" else ([_truth] if _truth else ["Image contradicts claim"]),
                                    "reasoning": "Remote AGLA verification",
                                }
                            except Exception as ae:
                                logging.warning(f"AGLA API error, falling back to local cross-checker: {ae}")
                                if getattr(self, "cross_checker", None) is not None:
                                    result = self.cross_checker.check_alignment(categorized_claim.text, image_path)
                                else:
                                    raise
                        else:
                            try:
                                logging.info(ConsoleColors.c('label', 'Verifier: ') + ConsoleColors.c('value', 'LOCAL'))
                            except Exception:
                                logging.info("Verifier: LOCAL")
                            result = self.cross_checker.check_alignment(categorized_claim.text, image_path)
                    # Robustness: ensure result is a dict; coerce None/invalid returns
                    if not isinstance(result, dict):
                        logging.warning("Cross-modal verifier returned non-dict/None; coercing to UNCERTAIN result")
                        result = {
                            "status": "UNCERTAIN",
                            "confidence": 0.0,
                            "evidence": [],
                            "sources": [],
                            "contradictions": [],
                            "reasoning": "Invalid verifier return",
                        }
                    # Evidence-based override for inconsistent third-party API results
                    original_status = result.get("status")
                    original_conf = result.get("confidence", 0.0)
                    
                    # Check if evidence contradicts the verdict (common with AGLA API)
                    if original_status == "FAIL" and result.get("evidence"):
                        evidence_supports_claim = self._analyze_evidence_support(
                            categorized_claim.text, result.get("evidence", [])
                        )
                        if evidence_supports_claim:
                            logging.info(f"Evidence-based override: FAIL → PASS (evidence supports claim)")
                            # Override the result
                            result = dict(result)  # Make a copy
                            result["status"] = "PASS"
                            result["confidence"] = min(0.75, original_conf)  # Moderate confidence for override
                            result["reasoning"] = f"Evidence-based override of {result.get('reasoning', 'API')} verdict"
                            if "evidence" in result:
                                result["evidence"].append(f"Override reason: Evidence text supports claim despite API verdict")
                    
                    factuality_results[i] = result
                    status = result.get("status")
                    conf = result.get("confidence", 0.0)
                    used = "AGLA" if "AGLA" in str(result.get("reasoning", "")) else "LOCAL"
                    logging.debug(f"[XMOD {stage_name}] Result via {used}: {status} (conf {conf:.2f}) | claim: {categorized_claim.text}")
                    logging.info(f"Cross-modal: {status} (conf {conf:.2f})")
                    # Evidence dump (capped)
                    try:
                        try:
                            _max_ev = int(os.getenv("SOC_MAX_EVIDENCE_LOG", "2") or 2)
                        except Exception:
                            _max_ev = 2
                        ev_list = result.get("evidence") or []
                        if ev_list:
                            ev_lines = []
                            if isinstance(ev_list, (list, tuple)):
                                for _e in list(ev_list)[: max(0, _max_ev)]:
                                    if isinstance(_e, dict):
                                        for k in ("text", "description", "desc", "explanation", "reason"):
                                            if isinstance(_e.get(k), str) and _e.get(k).strip():
                                                ev_lines.append(_e.get(k).strip())
                                                break
                                        if not ev_lines or len(ev_lines) < 1:
                                            try:
                                                ev_lines.append(", ".join(f"{kk}={vv}" for kk, vv in list(_e.items())[:3]))
                                            except Exception:
                                                ev_lines.append(str(_e))
                                    else:
                                        ev_lines.append(str(_e))
                            else:
                                ev_lines.append(str(ev_list))
                            # Trim lines to reasonable length
                            ev_lines = [ (s[:200] + "…") if isinstance(s, str) and len(s) > 200 else s for s in ev_lines ]
                            for j, line in enumerate(ev_lines, 1):
                                logging.info(f"Cross-modal evidence [{j}]: {line}")
                    except Exception:
                        pass
                    # Persist onto claim
                    categorized_claim.factuality_status = status
                    categorized_claim.factuality_confidence = conf
                    categorized_claim.factuality_verdict = True if status == "PASS" else (False if status == "FAIL" else None)
                    categorized_claim.factuality_evidence = result.get("evidence", [])
                    categorized_claim.factuality_sources = result.get("sources", [])
                    categorized_claim.factuality_reasoning = result.get("reasoning")
                    # Attribute-only KG updates on cross-modal PASS (e.g., colors, ordinals)
                    try:
                        if getattr(self, "kg_manager", None) and status == "PASS":
                            self.kg_manager.add_attribute_facts_from_claim(categorized_claim.text, self.session_id)
                            logging.info("Persisted attribute facts to KG after cross-modal PASS")
                    except Exception as e:
                        logging.warning(f"Failed to persist attribute facts from cross-modal PASS: {e}")
                except Exception as e:
                    logging.error(f"Cross-modal check error: {e}")
                    factuality_results[i] = {
                        "status": "ERROR",
                        "confidence": 0.0,
                        "external_facts": [],
                        "contradictions": [str(e)],
                        "evidence": [],
                        "sources": [],
                        "reasoning": "Exception during cross-modal check",
                    }
                    categorized_claim.factuality_status = "ERROR"
                    categorized_claim.factuality_confidence = 0.0
                    categorized_claim.factuality_verdict = None
                    categorized_claim.factuality_evidence = []
                    categorized_claim.factuality_sources = []
                    categorized_claim.factuality_reasoning = "Exception during cross-modal check"
                else:
                    # Post-factuality clarification for cross-modal uncertainties/conflicts
                    try:
                        if self.post_factuality_clarification_enabled and self.clarifier and status in ("FAIL", "UNCERTAIN"):
                            logging.info("Invoking Clarification Module for cross-modal conflict/uncertainty (post-factuality)...")
                            ev_list = []
                            for ev in (result.get("evidence", []) or []):
                                if isinstance(ev, str):
                                    ev_list.append({"summary": ev})
                                elif isinstance(ev, dict):
                                    ev_list.append(ev)
                            fc = ClarFactCheckResult(
                                verdict=status,
                                confidence=float(conf) if not isinstance(conf, str) else float(conf or 0.0),
                                reasoning=result.get("reasoning"),
                                evidence=ev_list,
                                sources=result.get("sources", []),
                            )
                            cat_enum = next((c.name for c in categorized_claim.categories if getattr(c.name, 'name', '') == 'AMBIGUOUS_RESOLUTION_REQUIRED'),
                                             (categorized_claim.categories[0].name if categorized_claim.categories else None))
                            ctx = ClarContext(
                                claim_text=categorized_claim.text,
                                category=cat_enum,
                                fact_check=fc,
                                failed_check_type="CROSS_MODAL",
                                issue_type=IssueType.VISUAL_CONFLICT,
                                claim_id=str(i),
                                metadata={"stage": "post_factuality", "verification_stage": getattr(self, "_verification_stage", "UNKNOWN"), "agla_truth": result.get("agla_truth")},
                            )
                            clr_res_cm = self.clarifier.resolve_claim(ctx)
                            self._clarification_results.setdefault(i, {})["post"] = clr_res_cm
                            if clr_res_cm.corrected_claim and clr_res_cm.corrected_claim.strip() and clr_res_cm.corrected_claim.strip() != categorized_claim.text.strip():
                                proposed2 = clr_res_cm.corrected_claim.strip()
                                allow_flip_env2 = os.getenv("SOC_ALLOW_POLARITY_FLIP", "false").lower() == "true"
                                if (not allow_flip_env2) and (_has_negation(categorized_claim.text) != _has_negation(proposed2)):
                                    logging.info("Skipping post-factuality correction (cross-modal) due to polarity flip guard")
                                else:
                                    categorized_claim.text = proposed2
                                    logging.info("Applying corrected claim from cross-modal clarification (post-factuality)")
                                # Optionally re-run cross-modal check if requested
                                if getattr(clr_res_cm, 'rerun_verification', False):
                                    try:
                                        if image_path:
                                            logging.info("Re-running cross-modal check on corrected claim...")
                                            # Prefer AGLA again on rerun
                                            if getattr(self, "agla_client", None) is not None:
                                                try:
                                                    # Reuse highest-confidence visual Socratic question on rerun
                                                    soc_q2 = None
                                                    try:
                                                        sq_map = getattr(categorized_claim, "socratic_questions", {}) or {}
                                                        vis_list = sq_map.get("VISUAL_GROUNDING_REQUIRED") or []
                                                        if vis_list:
                                                            soc_q2 = max(
                                                                vis_list,
                                                                key=lambda d: float(d.get("confidence_score", 0.0) or 0.0),
                                                            ).get("question")
                                                    except Exception:
                                                        pass
                                                    out2 = self.agla_client.verify(
                                                        image=image_path,
                                                        claim=categorized_claim.text,
                                                        socratic_question=soc_q2,
                                                        return_debug=False,
                                                    )
                                                    verdict2 = (out2 or {}).get("verdict", "Uncertain")
                                                    if verdict2 == "True":
                                                        _status2, _conf2 = "PASS", 0.85
                                                    elif verdict2 == "False":
                                                        _status2, _conf2 = "FAIL", 0.85
                                                    else:
                                                        _status2, _conf2 = "UNCERTAIN", 0.5
                                                    _truth2 = (out2 or {}).get("truth") or ""
                                                    ev2 = []
                                                    # if _status2 == "FAIL" and _truth2:
                                                    if _truth2:
                                                        ev2.append(f"AGLA evidence: {_truth2}")
                                                    ev2.append(f"AGLA verdict: {verdict2}")
                                                    srcs2 = []
                                                    try:
                                                        if AGLA_API_URL:
                                                            srcs2.append(f"{AGLA_API_URL}{AGLA_API_VERIFY_PATH}")
                                                    except Exception:
                                                        pass
                                                    result2 = {
                                                        "status": _status2,
                                                        "confidence": _conf2,
                                                        "evidence": ev2,
                                                        "sources": srcs2,
                                                        "contradictions": [] if _status2 != "FAIL" else ([_truth2] if _truth2 else ["Image contradicts claim"]),
                                                        "reasoning": "Remote AGLA verification (rerun)",
                                                    }
                                                except Exception as ae2:
                                                    logging.warning(f"AGLA API rerun error, falling back to local cross-checker: {ae2}")
                                                    if getattr(self, "cross_checker", None) is not None:
                                                        result2 = self.cross_checker.check_alignment(categorized_claim.text, image_path)
                                                    else:
                                                        raise
                                            else:
                                                result2 = self.cross_checker.check_alignment(categorized_claim.text, image_path)
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
                                        logging.warning(f"Rerun cross-modal check failed: {e}")
                    except Exception as e:
                        logging.warning(f"Cross-modal post-factuality clarification failed: {e}")

            # 4. External factuality check for routed claims (if enabled)
            if self.factuality_enabled and route.method == VerificationMethod.EXTERNAL_SOURCE:
                logging.info("Performing external factuality check for this claim...")
                try:
                    # Build optional input context for external factuality LLM aggregation
                    input_ctx = None
                    try:
                        mode = str(getattr(self, 'factuality_context_mode', 'SOCRATIC_QUESTIONS') or 'SOCRATIC_QUESTIONS').upper()
                        max_items = int(getattr(self, 'factuality_context_max_items', 6) or 6)
                        if mode == 'SOCRATIC_QUESTIONS':
                            items = []
                            sq_map = getattr(categorized_claim, "socratic_questions", {}) or {}
                            ek_list = sq_map.get("EXTERNAL_KNOWLEDGE_REQUIRED") or []
                            if ek_list:
                                sorted_ek = sorted(ek_list, key=lambda d: float(d.get("confidence_score", 0.0) or 0.0), reverse=True)
                                for q in sorted_ek:
                                    qtext = q.get("question")
                                    if qtext:
                                        items.append(qtext)
                            else:
                                for cat, qs in sq_map.items():
                                    if cat in ("SUBJECTIVE_OPINION", "PROCEDURAL_DESCRIPTIVE", "AMBIGUOUS_RESOLUTION_REQUIRED", "VISUAL_GROUNDING_REQUIRED", "SELF_CONSISTENCY_REQUIRED"):
                                        continue
                                    for q in qs:
                                        qtext = q.get("question")
                                        if qtext:
                                            items.append(qtext)
                            if items:
                                input_ctx = {"type": "SOCRATIC_QUESTIONS", "items": items[:max_items]}
                        elif mode == 'EXTRACTED_CLAIMS':
                            try:
                                other_claims = [c.text for c in claims if getattr(c, 'text', None) and str(c.text).strip() and str(c.text).strip() != str(categorized_claim.text).strip()]
                            except Exception:
                                other_claims = []
                            if other_claims:
                                input_ctx = {"type": "EXTRACTED_CLAIMS", "items": other_claims[:max_items]}
                        else:
                            input_ctx = None
                    except Exception as _e_ctx:
                        logging.debug(f"Failed to build input context: {_e_ctx}")
                        input_ctx = None

                    result = self.external_checker.verify_claim(categorized_claim.text, input_context=input_ctx)
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
                    # Attribute-only KG updates on external PASS (e.g., colors, ordinals)
                    try:
                        if getattr(self, "kg_manager", None) and status == "PASS":
                            self.kg_manager.add_attribute_facts_from_claim(categorized_claim.text, self.session_id)
                            logging.info("Persisted attribute facts to KG after external PASS")
                    except Exception as e:
                        logging.warning(f"Failed to persist attribute facts from external PASS: {e}")
                    # 4.1 If conflict or uncertainty, run clarification and optionally rerun verification
                    if self.post_factuality_clarification_enabled and self.clarifier and status in ("FAIL", "UNCERTAIN"):
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
                                metadata={"stage": "post_factuality", "verification_stage": getattr(self, "_verification_stage", "UNKNOWN")},
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
                                metadata={"stage": "post_factuality", "verification_stage": getattr(self, "_verification_stage", "UNKNOWN")},
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
                            proposed3 = clr_res2.corrected_claim.strip()
                            allow_flip_env3 = os.getenv("SOC_ALLOW_POLARITY_FLIP", "false").lower() == "true"
                            if (not allow_flip_env3) and (_has_negation(categorized_claim.text) != _has_negation(proposed3)):
                                logging.info("Skipping post-factuality correction (external) due to polarity flip guard")
                            else:
                                categorized_claim.text = proposed3
                                logging.info("Applying corrected claim from clarification (post-factuality)")
                            # Optionally rerun external verification based on module decision
                            if getattr(clr_res2, 'rerun_verification', False):
                                try:
                                    logging.info("Re-running external factuality check on corrected claim...")
                                    # Rebuild input context on corrected claim
                                    input_ctx2 = None
                                    try:
                                        mode2 = str(getattr(self, 'factuality_context_mode', 'SOCRATIC_QUESTIONS') or 'SOCRATIC_QUESTIONS').upper()
                                        max_items2 = int(getattr(self, 'factuality_context_max_items', 6) or 6)
                                        if mode2 == 'SOCRATIC_QUESTIONS':
                                            items2 = []
                                            sq_map2 = getattr(categorized_claim, "socratic_questions", {}) or {}
                                            ek_list2 = sq_map2.get("EXTERNAL_KNOWLEDGE_REQUIRED") or []
                                            if ek_list2:
                                                sorted_ek2 = sorted(ek_list2, key=lambda d: float(d.get("confidence_score", 0.0) or 0.0), reverse=True)
                                                for q in sorted_ek2:
                                                    qtext = q.get("question")
                                                    if qtext:
                                                        items2.append(qtext)
                                            else:
                                                for cat, qs in sq_map2.items():
                                                    if cat in ("SUBJECTIVE_OPINION", "PROCEDURAL_DESCRIPTIVE", "AMBIGUOUS_RESOLUTION_REQUIRED", "VISUAL_GROUNDING_REQUIRED", "SELF_CONSISTENCY_REQUIRED"):
                                                        continue
                                                    for q in qs:
                                                        qtext = q.get("question")
                                                        if qtext:
                                                            items2.append(qtext)
                                            if items2:
                                                input_ctx2 = {"type": "SOCRATIC_QUESTIONS", "items": items2[:max_items2]}
                                        elif mode2 == 'EXTRACTED_CLAIMS':
                                            try:
                                                other_claims2 = [c.text for c in claims if getattr(c, 'text', None) and str(c.text).strip() and str(c.text).strip() != str(categorized_claim.text).strip()]
                                            except Exception:
                                                other_claims2 = []
                                            if other_claims2:
                                                input_ctx2 = {"type": "EXTRACTED_CLAIMS", "items": other_claims2[:max_items2]}
                                        else:
                                            input_ctx2 = None
                                    except Exception as _e_ctx2:
                                        logging.debug(f"Failed to build input context (rerun): {_e_ctx2}")
                                        input_ctx2 = None

                                    result2 = self.external_checker.verify_claim(categorized_claim.text, input_context=input_ctx2)
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
                    # Choose between auto vs manual conflict resolution
                    try:
                        mode = getattr(self, 'conflict_resolution_mode', 'auto')
                    except Exception:
                        mode = 'auto'
                    if mode == 'manual':
                        final_result = self._manual_resolve_conflict(categorized_claim.text, ext_result, sc_result)
                    else:
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
                    # Final aggregated clarification stage: forward combined evidence for hallucination correction
                    try:
                        if self.post_factuality_clarification_enabled and self.clarifier and str(final_result.get("status", "")).upper() in ("FAIL", "UNCERTAIN"):
                            ev_all = []
                            # External evidence
                            for ev in ((ext_result or {}).get("evidence") or (ext_result or {}).get("external_facts") or []):
                                if isinstance(ev, str):
                                    ev_all.append({"summary": ev})
                                elif isinstance(ev, dict):
                                    ev_all.append(ev)
                            # Self-consistency evidence/contradictions
                            for ev in ((sc_result or {}).get("evidence") or []):
                                if isinstance(ev, str):
                                    ev_all.append({"summary": ev})
                                elif isinstance(ev, dict):
                                    ev_all.append(ev)
                            for con in ((sc_result or {}).get("contradictions") or []):
                                if isinstance(con, str):
                                    ev_all.append({"summary": con})
                                elif isinstance(con, dict):
                                    ev_all.append(con)
                            fc_final = ClarFactCheckResult(
                                verdict=final_result.get("status"),
                                confidence=float(final_result.get("confidence", 0.0)),
                                reasoning=final_result.get("reasoning"),
                                evidence=ev_all,
                                sources=(ext_result or {}).get("sources", []) or [],
                            )
                            # Determine issue type based on underlying failures
                            issue = IssueType.EXTERNAL_FACTUAL_CONFLICT
                            try:
                                sc_status = str((sc_result or {}).get("status", "")).upper()
                                if sc_status == "FAIL" or ((sc_result or {}).get("contradictions")):
                                    issue = IssueType.KNOWLEDGE_CONTRADICTION
                            except Exception:
                                pass
                            cat_enum2 = next((c.name for c in categorized_claim.categories), None)
                            ctx_final = ClarContext(
                                claim_text=categorized_claim.text,
                                category=cat_enum2,
                                fact_check=fc_final,
                                failed_check_type="MERGED",
                                issue_type=issue,
                                claim_id=str(i),
                                metadata={"stage": "final_post_merge", "verification_stage": getattr(self, "_verification_stage", "UNKNOWN")},
                            )
                            clr_res_final = self.clarifier.resolve_claim(ctx_final)
                            self._clarification_results.setdefault(i, {})["final"] = clr_res_final
                            # Optionally apply corrected claim text (no auto-rerun here to avoid loops)
                            if getattr(clr_res_final, 'corrected_claim', None) and clr_res_final.corrected_claim.strip() and clr_res_final.corrected_claim.strip() != categorized_claim.text.strip():
                                proposed4 = clr_res_final.corrected_claim.strip()
                                allow_flip_env4 = os.getenv("SOC_ALLOW_POLARITY_FLIP", "false").lower() == "true"
                                if (not allow_flip_env4) and (_has_negation(categorized_claim.text) != _has_negation(proposed4)):
                                    logging.info("Skipping final post-merge correction due to polarity flip guard")
                                else:
                                    categorized_claim.text = proposed4
                    except Exception as e:
                        logging.warning(f"Final aggregated clarification failed: {e}")
                    # Add to KG if recommended (exclude meta/ambiguous categories)
                    is_meta_claim = False
                    try:
                        META_CATS = {
                            "AMBIGUOUS_RESOLUTION_REQUIRED",
                            "CLARIFICATION_REQUIRED",
                            "META_REASONING",
                            "PROCEDURAL_DESCRIPTIVE",
                            "SUBJECTIVE_OPINION",
                        }
                        for c in getattr(categorized_claim, "categories", []) or []:
                            nm = getattr(c, "name", None)
                            # Support Enum.name or nested name types
                            try:
                                nm_str = getattr(nm, "name", None) or str(nm)
                            except Exception:
                                nm_str = str(nm)
                            if nm_str in META_CATS:
                                is_meta_claim = True
                                break
                    except Exception:
                        pass

                    if (
                        getattr(self, "kg_manager", None)
                        and final_result.get("should_add_to_kg")
                        and str(final_result.get("status", "")).upper() == "PASS"
                        and not is_meta_claim
                    ):
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
    
    def _analyze_evidence_support(self, claim_text: str, evidence_list: List[Any]) -> bool:
        """
        LLM-powered analysis of whether evidence text actually supports a claim.
        
        This handles cases where third-party APIs (like AGLA) return FAIL verdicts
        but their evidence text actually supports the claim, using sophisticated
        semantic understanding via LLM.
        
        Args:
            claim_text: The original claim being verified
            evidence_list: List of evidence items from the verifier
            
        Returns:
            True if evidence supports the claim, False otherwise
        """
        if not claim_text or not evidence_list:
            return False
            
        try:
            # Extract evidence text from various formats
            evidence_texts = []
            for ev in evidence_list:
                if isinstance(ev, str):
                    s = ev.strip()
                    sl = s.lower()
                    # Drop meta/label lines that bias analysis
                    if sl.startswith("agla verdict:") or sl.startswith("socratic question:") or sl.startswith("override reason:"):
                        continue
                    # Normalize AGLA prefixes to keep only descriptive content
                    for pref in ("AGLA evidence:", "AGLA correction:"):
                        if s.startswith(pref):
                            s = s[len(pref):].strip()
                            break
                    if s:
                        evidence_texts.append(s)
                elif isinstance(ev, dict):
                    # Try common evidence text keys
                    for key in ["text", "description", "correction", "summary", "explanation"]:
                        if key in ev and isinstance(ev[key], str):
                            evidence_texts.append(ev[key])
                            break
                    # Also handle AGLA-style evidence dicts if any
                    _s = str(ev)
                    if _s.startswith("AGLA correction:"):
                        evidence_texts.append(_s[len("AGLA correction:"):].strip())
            
            if not evidence_texts:
                return False
            
            # Combine all evidence into a single text for analysis
            combined_evidence = " ".join(evidence_texts)
            
            # Use LLM to analyze semantic relationship between claim and evidence
            return self._llm_analyze_claim_evidence_alignment(claim_text, combined_evidence)
            
        except Exception as e:
            logging.debug(f"Error in LLM evidence analysis: {e}")
            # Fallback to simple heuristic if LLM fails
            return self._fallback_evidence_analysis(claim_text, evidence_list)
    
    def _llm_analyze_claim_evidence_alignment(self, claim: str, evidence: str) -> bool:
        """
        Use LLM to determine if evidence supports a claim with semantic understanding.
        
        Args:
            claim: The claim to verify
            evidence: The evidence text to analyze
            
        Returns:
            True if evidence supports the claim, False otherwise
        """
        try:
            # Get the pipeline LLM (fallback to main LLM if not available)
            llm = getattr(self, 'llm_manager', None)
            if not llm:
                logging.debug("No LLM available for evidence analysis")
                return False
            
            # Construct a focused prompt for claim-evidence alignment
            prompt = f"""You are an expert fact-checker analyzing whether EVIDENCE supports a CLAIM.

CLAIM: "{claim}"

EVIDENCE: "{evidence}"

Task: Determine if the evidence semantically supports the claim, even if the wording differs.

Consider:
- Semantic equivalence (e.g., "holding a bat" supports "bat is present")
- Logical implications (e.g., "player with bat" implies "bat exists")
- Context and relationships (e.g., "pizza on a white plate" supports "there is a white plate")

Ignore:
- Any metadata or labels such as "AGLA verdict:", "Socratic question:", or "Override reason:" (these lines are not descriptive evidence)
- Minor wording differences
- Exact phrase matching requirements
- Overly strict interpretations

Respond with ONLY one word:
- "SUPPORTS" if the evidence supports or is consistent with the claim
- "CONTRADICTS" if the evidence contradicts or refutes the claim
- "UNCLEAR" if the relationship is ambiguous or insufficient

Response:"""
            
            # Generate LLM response with minimal tokens
            try:
                response = llm.generate_text(
                    prompt,
                    max_tokens=10,  # Very short response
                    temperature=0.1  # Low temperature for consistency
                )
                
                if response:
                    response_clean = response.strip().upper()
                    if "SUPPORTS" in response_clean:
                        logging.debug(f"LLM evidence analysis: SUPPORTS - '{evidence[:100]}...' supports '{claim}'")
                        return True
                    elif "CONTRADICTS" in response_clean:
                        logging.debug(f"LLM evidence analysis: CONTRADICTS - '{evidence[:100]}...' contradicts '{claim}'")
                        return False
                    else:
                        logging.debug(f"LLM evidence analysis: UNCLEAR - '{evidence[:100]}...' unclear for '{claim}'")
                        return False
                else:
                    logging.debug("LLM returned empty response for evidence analysis")
                    return False
                    
            except Exception as e:
                logging.debug(f"LLM generation failed in evidence analysis: {e}")
                return False
                
        except Exception as e:
            logging.debug(f"Error in LLM evidence analysis: {e}")
            return False
    
    def _fallback_evidence_analysis(self, claim_text: str, evidence_list: List[Any]) -> bool:
        """
        Fallback rule-based evidence analysis when LLM is unavailable.
        
        Args:
            claim_text: The original claim being verified
            evidence_list: List of evidence items from the verifier
            
        Returns:
            True if evidence supports the claim, False otherwise
        """
        try:
            # Extract evidence text
            evidence_texts = []
            for ev in evidence_list:
                if isinstance(ev, str):
                    evidence_texts.append(ev.lower())
                elif isinstance(ev, dict):
                    for key in ["text", "description", "correction", "summary", "explanation"]:
                        if key in ev and isinstance(ev[key], str):
                            evidence_texts.append(ev[key].lower())
                            break
            
            if not evidence_texts:
                return False
            
            claim_lower = claim_text.lower().strip()
            
            # Simple keyword-based analysis for existence claims
            if any(phrase in claim_lower for phrase in ["is present", "is in", "there is", "contains", "has a"]):
                # Extract main entity
                for phrase in ["there is a ", "is a ", "contains a ", "has a "]:
                    if phrase in claim_lower:
                        parts = claim_lower.split(phrase, 1)
                        if len(parts) > 1:
                            entity = parts[1].split()[0].strip(".?!,")
                            # Check if entity appears positively in evidence
                            for ev_text in evidence_texts:
                                if entity in ev_text and not any(neg in ev_text for neg in ["no ", "not ", "without "]):
                                    return True
            
            return False
            
        except Exception as e:
            logging.debug(f"Error in fallback evidence analysis: {e}")
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Socrates claim processing pipeline")
    parser.add_argument("--enable-factuality", dest="enable_factuality", action="store_true", help="Enable external factuality checking")
    parser.add_argument("--disable-factuality", dest="disable_factuality", action="store_true", help="Disable external factuality checking")
    parser.add_argument("--text", type=str, default="in this image I was standing in front of a London Big Ben tower, which is in Germany.", help="Input text to process")
    parser.add_argument("--image", dest="image", type=str, default=None, help="Optional path to an image for cross-modal verification")
    # Clarification toggles
    parser.add_argument("--enable-clarification", dest="enable_clarification", action="store_true", help="Enable clarification module")
    parser.add_argument("--disable-clarification", dest="disable_clarification", action="store_true", help="Disable clarification module")
    parser.add_argument("--clar-dev", dest="clar_dev", action="store_true", help="Enable clarification dev mode")
    # Socratic question generation toggles
    parser.add_argument("--enable-question-gen", dest="enable_qg", action="store_true", help="Enable Socratic question generation")
    parser.add_argument("--disable-question-gen", dest="disable_qg", action="store_true", help="Disable Socratic question generation")
    parser.add_argument("--questions-per-category", dest="qg_per_cat", type=int, default=None, help="Number of Socratic questions to generate per relevant category")
    # External factuality context selection
    parser.add_argument("--factuality-context", dest="factuality_context", type=str, choices=["socratic", "claims", "none"], default=None, help="Context used to guide external factuality verdict aggregation")
    parser.add_argument("--fact-context-max-items", dest="fact_ctx_max_items", type=int, default=None, help="Max number of context items to include in external factuality prompt")
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
    # Conflict resolution mode
    parser.add_argument("--conflict-mode", dest="conflict_mode", type=str, choices=["auto", "manual"], default=None, help="Conflict resolution: auto uses resolver; manual prompts user decision")
    # Router mode
    parser.add_argument("--router-mode", dest="router_mode", type=str, choices=["llm", "deterministic", "hybrid"], default=None, help="Routing mode: 'llm', 'deterministic', or 'hybrid'")
    # LLM selection
    parser.add_argument("--llm-provider", dest="llm_provider", type=str, choices=["ollama", "openai", "claude", "llava_hf", "llava_original"], default=None, help="LLM provider to use (overrides SOC_LLM_PROVIDER). Supports: ollama | openai | claude | llava_hf | llava_original")
    parser.add_argument("--llm-model", dest="llm_model", type=str, default=None, help="Model name for the selected provider (overrides SOC_LLM_MODEL)")
    # Original LLaVA provider advanced options (mapped to env toggles used by LLMManager)
    parser.add_argument("--llava-orig-use-cli", dest="llava_orig_use_cli", action="store_true", help="Force original LLaVA CLI fallback (sets SOC_LLAVA_ORIG_USE_CLI=true)")
    parser.add_argument("--llava-conv-template", dest="llava_conv_template", type=str, default=None, help="Conversation template for original LLaVA (sets SOC_LLAVA_CONV_TEMPLATE)")
    parser.add_argument("--llava-timeout-sec", dest="llava_timeout_sec", type=int, default=None, help="Timeout seconds for LLaVA CLI calls (sets SOC_LLAVA_TIMEOUT_SEC)")
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
    # Apply LLaVA original toggles from CLI flags via env for LLMManager consumption
    try:
        if getattr(args, "llava_orig_use_cli", False):
            os.environ["SOC_LLAVA_ORIG_USE_CLI"] = "true"
        if getattr(args, "llava_conv_template", None):
            os.environ["SOC_LLAVA_CONV_TEMPLATE"] = str(args.llava_conv_template)
        if getattr(args, "llava_timeout_sec", None) is not None:
            os.environ["SOC_LLAVA_TIMEOUT_SEC"] = str(int(args.llava_timeout_sec))
    except Exception as _e:
        logging.warning(f"Failed to apply LLaVA CLI/env overrides: {_e}")
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
        conflict_resolution_mode=(args.conflict_mode or os.getenv("SOC_CONFLICT_MODE")),
        factuality_context_mode=args.factuality_context,
        factuality_context_max_items=args.fact_ctx_max_items,
        router_mode=args.router_mode,
    )

    # Inform about selected LLM
    try:
        print(ConsoleColors.c('label', 'Using LLM: ') + ConsoleColors.c('value', f"{getattr(llm_manager, 'provider').value}:{getattr(llm_manager, 'model_name')}") )
    except Exception:
        pass
    # Inform about conflict resolution mode
    try:
        print(ConsoleColors.c('label', 'Conflict mode: ') + ConsoleColors.c('value', f"{getattr(pipeline, 'conflict_resolution_mode', 'auto')}") )
    except Exception:
        pass
    # Inform about router mode
    try:
        print(ConsoleColors.c('label', 'Router mode: ') + ConsoleColors.c('value', f"{getattr(pipeline, 'router_mode', 'llm')}") )
    except Exception:
        pass

    final_claims = pipeline.run(args.text, image_path=args.image)

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
            # Route confidence
            try:
                print("    - " + ConsoleColors.c('label', 'Confidence: ') + ConsoleColors.c('value', f"{float(getattr(route, 'confidence', 0.0) or 0.0):.2f}"))
            except Exception:
                pass
            # Secondary actions (if any)
            if getattr(route, 'secondary_actions', None):
                print("    - " + ConsoleColors.c('label', 'Secondary actions:'))
                for act in route.secondary_actions:
                    try:
                        name = act.get('name') or act.get('action') or act.get('type') or str(act)
                    except Exception:
                        name = str(act)
                    print("      • " + ConsoleColors.c('value', f"{name}"))
            # Route metadata summary (compact)
            try:
                md = getattr(route, 'metadata', {}) or {}
                md_items = []
                if md.get('router_mode'):
                    md_items.append(f"mode={md.get('router_mode')}")
                if md.get('selected_by'):
                    md_items.append(f"selected_by={md.get('selected_by')}")
                if md.get('kg_coverage_ratio') is not None:
                    try:
                        md_items.append(f"kg_cov={float(md.get('kg_coverage_ratio', 0.0)):.2f}")
                    except Exception:
                        md_items.append(f"kg_cov={md.get('kg_coverage_ratio')}")
                if md.get('kg_contradiction_detected'):
                    md_items.append("kg_contra=True")
                if md_items:
                    print("    - " + ConsoleColors.c('label', 'Route meta: ') + ConsoleColors.c('value', ", ".join(md_items)))
            except Exception:
                pass
            # Claim-level hints from extraction/categorization
            if getattr(claim, 'route_hint', None) or getattr(claim, 'vision_flag', None):
                print("    - " + ConsoleColors.c('label', 'Hints: ') + ConsoleColors.c('value', f"route_hint={getattr(claim, 'route_hint', None)}, vision={getattr(claim, 'vision_flag', None)}"))
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

    # Summary metrics for factuality stage
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
            rate_str = f"{fb_rate:.1f}"
            line = (
                ConsoleColors.c('label', 'Total questions: ') + ConsoleColors.c('value', str(q_total)) +
                ConsoleColors.c('label', ' | Fallback used: ') + ConsoleColors.c('value', str(q_fallback)) +
                ConsoleColors.c('label', ' (') + ConsoleColors.c('value', rate_str) + ConsoleColors.c('label', '%)')
            )
            print(line)

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
