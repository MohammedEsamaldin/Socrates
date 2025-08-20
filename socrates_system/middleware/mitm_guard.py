"""
Hallucination MitM (Man-in-the-Middle) middleware

Intercepts input (text/image) and output (model response) to detect and mitigate
hallucinations. It minimally edits only hallucinating tokens/claims, using existing
Socrates modules for verification and an LLM for correction/clarification.

Usage example:

    from socrates_system.middleware.mitm_guard import HallucinationMitM, LLMMainModelAdapter

    mitm = HallucinationMitM(
        main_model=LLMMainModelAdapter(provider="ollama", model_name="llama3.1:8b"),
    )
    result = mitm.run(text="The Eiffel Tower is in Berlin.")
    print(result.corrected_output)

Design notes:
- Uses ClaimExtractor + ClaimCategorizer + CheckRouter to identify verification routes.
- Applies ExternalFactualityChecker (text), CrossAlignmentChecker/AGLA (image),
  and SelfContradictionChecker (session KG), then ConflictResolver to merge results.
- When a claim is FAIL/UNCERTAIN or ambiguous, calls LLMManager to propose a corrected
  version of the claim. Only the differing tokens within the claim span are replaced.
- Performs the same pipeline pre-model (input) and post-model (output).
- Allows selecting/plugging the main model via a simple adapter interface.
"""
from __future__ import annotations

import os
import uuid
import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..modules.claim_extractor import ClaimExtractor
from ..modules.claim_categorizer import ClaimCategorizer
from ..modules.check_router import CheckRouter
from ..modules.shared_structures import ExtractedClaim, VerificationMethod
from ..modules.external_factuality_checker import ExternalFactualityChecker
from ..modules.knowledge_graph_manager import KnowledgeGraphManager
from ..modules.self_contradiction_checker import SelfContradictionChecker
from ..modules.conflict_resolver import ConflictResolver
from ..modules.llm_manager import LLMManager, LLMResponse

# Cross-modal options (prefer remote AGLA if configured)
try:
    from ..modules.agla_client import AGLAClient
    from ..config import (
        AGLA_API_URL,
        AGLA_API_VERIFY_PATH,
        AGLA_API_TIMEOUT,
    )
except Exception:  # pragma: no cover
    AGLAClient = None  # type: ignore
    AGLA_API_URL = None  # type: ignore
    AGLA_API_VERIFY_PATH = None  # type: ignore
    AGLA_API_TIMEOUT = None  # type: ignore

try:  # advanced cross-modal (fallback to simple inside checker)
    from ..modules.cross_alignment_checker import CrossAlignmentChecker as AdvancedCrossAlignmentChecker
except Exception:  # pragma: no cover
    AdvancedCrossAlignmentChecker = None  # type: ignore
from ..modules.cross_alignment_checker_simple import (
    CrossAlignmentChecker as SimpleCrossAlignmentChecker,
)

logger = logging.getLogger(__name__)


class MainModelAdapter(Protocol):
    """Protocol for the main model to be wrapped by the MitM.

    Implementations should accept optional image_path for multimodal prompts.
    """

    def generate(self, text: str, image_path: Optional[str] = None, **kwargs) -> str:  # pragma: no cover - interface only
        ...


class LLMMainModelAdapter:
    """A simple main-model adapter backed by the shared LLMManager.

    Useful for testing or when your main model is an LLM supported by LLMManager.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        anthropic_base_url: Optional[str] = None,
    ) -> None:
        self.llm = LLMManager(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            anthropic_api_key=anthropic_api_key,
            anthropic_base_url=anthropic_base_url,
        )

    def generate(self, text: str, image_path: Optional[str] = None, **kwargs) -> str:
        # For simplicity, ignore image here; LLMManager is text-first.
        return self.llm.generate_text(prompt=text, max_tokens=512)


@dataclass
class Correction:
    start: int
    end: int
    original: str
    replacement: str
    reason: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)


@dataclass
class MitMRunResult:
    corrected_input: str
    input_corrections: List[Correction]
    raw_output: str
    corrected_output: str
    output_corrections: List[Correction]
    session_id: str


class HallucinationMitM:
    def __init__(
        self,
        main_model: Optional[MainModelAdapter] = None,
        llm_manager: Optional[LLMManager] = None,
        session_id: Optional[str] = None,
    ) -> None:
        # LLM used for verification/correction prompts
        self.llm = llm_manager or LLMManager()

        # Main model adapter (can be any implementation of MainModelAdapter)
        self.main_model = main_model or LLMMainModelAdapter()

        # Core modules
        self.extractor = ClaimExtractor(llm_manager=self.llm)
        self.categorizer = ClaimCategorizer(llm_manager=self.llm)
        self.router = CheckRouter()
        self.kg_manager = KnowledgeGraphManager()
        self.self_checker = SelfContradictionChecker()
        self.self_checker.set_kg_manager(self.kg_manager)
        self.external_checker = ExternalFactualityChecker(llm_manager=self.llm)
        self.conflict_resolver = ConflictResolver(llm_manager=self.llm)

        # Cross-modal
        self.agla = None
        if AGLAClient and (AGLA_API_URL or os.getenv("AGLA_API_URL")):
            try:
                self.agla = AGLAClient(
                    base_url=os.getenv("AGLA_API_URL", AGLA_API_URL),
                    verify_path=os.getenv("AGLA_API_VERIFY_PATH", AGLA_API_VERIFY_PATH or "/verify"),
                    timeout=float(os.getenv("AGLA_API_TIMEOUT", str(AGLA_API_TIMEOUT or 20))),
                )
                logger.info("MitM: Using remote AGLA for cross-modal verification")
            except Exception as e:  # pragma: no cover
                logger.warning(f"MitM: Failed to init AGLA client, falling back to local checker: {e}")
                self.agla = None
        self.cross_modal = AdvancedCrossAlignmentChecker() if AdvancedCrossAlignmentChecker else SimpleCrossAlignmentChecker()

        # Session
        self.session_id = session_id or os.getenv("SOC_SESSION_ID") or str(uuid.uuid4())
        self.kg_manager.initialize_session(self.session_id)

    # ---------------------- Public API ----------------------
    def run(self, text: Optional[str] = None, image_path: Optional[str] = None, **gen_kwargs) -> MitMRunResult:
        """Run full MitM pipeline: preprocess input -> main model -> postprocess output."""
        original_text = text or ""
        corrected_input, input_corrs = self._process_text(text=original_text, image_path=image_path)

        # Generate with the main model
        raw_output = self.main_model.generate(corrected_input, image_path=image_path, **gen_kwargs)

        # Post-process output
        corrected_output, output_corrs = self._process_text(text=raw_output, image_path=image_path)

        return MitMRunResult(
            corrected_input=corrected_input,
            input_corrections=input_corrs,
            raw_output=raw_output,
            corrected_output=corrected_output,
            output_corrections=output_corrs,
            session_id=self.session_id,
        )

    # ---------------------- Core logic ----------------------
    def _process_text(self, text: str, image_path: Optional[str]) -> Tuple[str, List[Correction]]:
        if not text:
            return text, []

        claims = self.extractor.extract_claims(text)
        if not claims:
            return text, []

        # Categorize and route claims
        categorized = self.categorizer.categorize_claims([c.text for c in claims])
        routes = self.router.route_claims(categorized)

        # Attach categories/routes back to claim objects (best-effort)
        for i, claim in enumerate(claims):
            try:
                claim.categories = categorized[i].categories  # type: ignore[attr-defined]
                claim.verification_route = routes[i]
            except Exception:
                pass

        # Verify each claim and produce corrections when needed
        corrections: List[Correction] = []
        for c in claims:
            route = getattr(c, "verification_route", None)
            if not route:
                continue

            verdict = self._verify_claim_route(c, route, image_path=image_path)

            # Update KG on PASS (attribute facts)
            try:
                if verdict.get("status") == "PASS":
                    self.kg_manager.add_attribute_facts_from_claim(c.text, self.session_id)
            except Exception:
                pass

            # Decide whether to correct
            need_fix = verdict.get("status") in ("FAIL", "UNCERTAIN") or (verdict.get("ambiguity_reason") is not None)
            if need_fix:
                corr = self._propose_correction(claim=c, verdict=verdict, image_path=image_path)
                if corr and corr.replacement and corr.replacement.strip() and corr.replacement.strip() != c.text.strip():
                    corrections.append(corr)

        # Apply minimal token-level edits
        corrected_text = self._apply_corrections_minimal(text, claims, corrections)
        return corrected_text, corrections

    def _verify_claim_route(self, claim: ExtractedClaim, route: Any, image_path: Optional[str]) -> Dict[str, Any]:
        method = getattr(route, "method", None)
        verdict: Dict[str, Any] = {}

        try:
            if method == VerificationMethod.CROSS_MODAL and image_path:
                verdict = self._cross_modal_verify(claim.text, image_path)
            elif method == VerificationMethod.EXTERNAL_SOURCE:
                verdict = self.external_checker.verify_claim(claim.text, input_context=None)
            elif method == VerificationMethod.KNOWLEDGE_GRAPH:
                verdict = self.self_checker.check_contradiction(claim.text, self.session_id)
            else:
                verdict = {"status": "UNCERTAIN", "confidence": 0.5, "reasoning": "No verification route applicable"}
        except Exception as e:
            logger.warning(f"Verification failed for claim '{claim.text[:40]}...': {e}")
            verdict = {"status": "UNCERTAIN", "confidence": 0.3, "reasoning": f"Verification error: {e}"}

        # If both external and self checks are available in future, merge with resolver
        # Here, resolver expects structures similar to pipeline, so we pass best-effort
        return verdict

    def _cross_modal_verify(self, claim_text: str, image_path: str) -> Dict[str, Any]:
        # Prefer remote AGLA
        if self.agla is not None:
            try:
                res = self.agla.verify(claim_text=claim_text, image_path=image_path)
                return res or {"status": "UNCERTAIN", "confidence": 0.5}
            except Exception as e:
                logger.warning(f"AGLA verify failed; falling back to local: {e}")
        # Local checker
        try:
            res = self.cross_modal.check_alignment(text_claim=claim_text, image_path=image_path)
            # Map to standardized fields (ensure keys)
            return {
                "status": res.get("status", "UNCERTAIN"),
                "confidence": res.get("confidence", 0.5),
                "evidence": res.get("evidence", []),
                "contradictions": res.get("contradictions", []),
                "reasoning": res.get("reasoning", ""),
                "sources": [],
            }
        except Exception as e:
            logger.warning(f"Cross-modal local check failed: {e}")
            return {"status": "UNCERTAIN", "confidence": 0.5, "reasoning": f"Cross-modal error: {e}"}

    # ---------------------- Correction logic ----------------------
    def _propose_correction(self, claim: ExtractedClaim, verdict: Dict[str, Any], image_path: Optional[str]) -> Optional[Correction]:
        # Build evidence/context for the LLM to rewrite the claim minimally
        evidence_lines: List[str] = []
        sources: List[str] = []
        if verdict.get("evidence"):
            ev = verdict.get("evidence")
            if isinstance(ev, list):
                evidence_lines.extend([str(x) for x in ev])
        if verdict.get("external_facts"):
            ev2 = verdict.get("external_facts")
            if isinstance(ev2, list):
                evidence_lines.extend([str(x) for x in ev2])
        if verdict.get("sources"):
            src = verdict.get("sources")
            if isinstance(src, list):
                sources.extend([str(x) for x in src])

        context = "\n".join([f"- {line}" for line in evidence_lines]) or "(no evidence available)"
        system_prompt = (
            "You are a precise factual editor. Given a possibly hallucinated claim and evidence, "
            "rewrite ONLY what is necessary to make the claim true and unambiguous. Keep style and length similar. "
            "Return ONLY the corrected claim text, no explanations."
        )
        user_prompt = (
            f"Original claim: {claim.text}\n"
            f"Evidence/context:\n{context}\n"
            f"If evidence is insufficient, produce a conservative, non-committal correction that avoids unsupported specifics."
        )
        try:
            corrected = self.llm.generate_text(prompt=user_prompt, system_prompt=system_prompt, max_tokens=128, temperature=0.2) or ""
            corrected = corrected.strip().strip('`').strip()
        except Exception as e:
            logger.warning(f"Correction LLM failed: {e}")
            return None

        # Minimal-diff replacement confined to the claim span
        start, end = claim.start_char, claim.end_char
        original_span = claim.source_text[start:end]
        replacement_span = self._minimal_token_rewrite(original_span, corrected)

        return Correction(
            start=start,
            end=end,
            original=original_span,
            replacement=replacement_span,
            reason=verdict.get("reasoning", "hallucination mitigation"),
            confidence=float(verdict.get("confidence", 0.5) or 0.5),
            sources=list(set(sources)),
        )

    def _minimal_token_rewrite(self, original: str, corrected: str) -> str:
        # If the LLM returned something wildly different, prefer corrected entirely
        if not original or not corrected:
            return corrected or original
        o_tokens = original.split()
        c_tokens = corrected.split()
        sm = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)
        out: List[str] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                out.extend(o_tokens[i1:i2])
            elif tag in ('replace', 'delete', 'insert'):
                out.extend(c_tokens[j1:j2])
        return " ".join(out).strip()

    def _apply_corrections_minimal(self, text: str, claims: List[ExtractedClaim], corrections: List[Correction]) -> str:
        if not corrections:
            return text
        # Sort by start index to apply left-to-right; adjust offsets as we go
        corr_sorted = sorted(corrections, key=lambda c: c.start)
        result_chars: List[str] = []
        last_index = 0
        offset_shift = 0
        for corr in corr_sorted:
            start, end = corr.start, corr.end
            # Clamp to text bounds
            start = max(0, min(len(text), start))
            end = max(start, min(len(text), end))
            result_chars.append(text[last_index:start])
            result_chars.append(corr.replacement)
            last_index = end
        result_chars.append(text[last_index:])
        return "".join(result_chars)


__all__ = [
    "HallucinationMitM",
    "LLMMainModelAdapter",
    "MainModelAdapter",
    "Correction",
    "MitMRunResult",
]
