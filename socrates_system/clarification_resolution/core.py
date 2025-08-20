from __future__ import annotations

import json
import re
import difflib
from typing import Any, Callable, Dict, List, Optional

from .data_models import (
    ClarificationContext,
    ClarificationResult,
    FactCheckResult,
    IssueType,
    ResolutionAction,
    SocraticQuestion,
)
from .question_generators import GENERATOR_BY_ISSUE
from . import config as clar_cfg

# Reuse project's logger
try:
    from socrates_system.utils.logger import setup_logger
except Exception:  # Fallback
    import logging
    def setup_logger(name: str):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

logger = setup_logger(__name__)

# LLM Manager
try:
    from socrates_system.modules.llm_manager import get_llm_manager
except Exception:
    get_llm_manager = None  # type: ignore

ResponseProvider = Callable[[SocraticQuestion], Any]


class ClarificationResolutionModule:
    """LLM-driven clarification module to resolve problematic claims via Socratic dialogue.

    Workflow:
      1) Receive context (claim, category, fact-check result, failed check type, issue type)
      2) Generate issue-specific Socratic questions
      3) Optionally refine/style questions with LLM
      4) Collect user responses via provided callback or interactive prompt
      5) Produce a corrected claim using LLM + validate
      6) Compute resolution confidence and decide next action (reverify vs direct to KG)
    """

    def __init__(
        self,
        llm_manager=None,
        dev_mode: bool = clar_cfg.DEV_MODE_DEFAULT,
        route_policy: Optional[Dict[str, str]] = None,
    ) -> None:
        self.llm = llm_manager or (get_llm_manager() if get_llm_manager else None)
        self.dev_mode = dev_mode
        # Default routing policy per issue type (string keys for easier config)
        self.route_policy: Dict[str, str] = route_policy or dict(clar_cfg.DEFAULT_NEXT_ACTION)
        logger.info("ClarificationResolutionModule initialized (dev_mode=%s)", dev_mode)

    def resolve_claim(
        self,
        ctx: ClarificationContext,
        responses: Optional[Dict[str, Any]] = None,
        response_provider: Optional[ResponseProvider] = None,
        max_questions: int = clar_cfg.MAX_QUESTIONS_PER_SESSION,
    ) -> ClarificationResult:
        # 1) Generate questions
        questions = self._generate_questions(ctx, max_questions=max_questions)
        # 2) Collect responses if not provided
        if responses is None:
            responses = self._collect_responses(questions, response_provider)
        # 3) Correct claim based on responses
        corrected_claim, reasoning = self._process_responses_and_correct_claim(ctx, questions, responses)
        # 4) Confidence and next action
        resolution_conf = self._calculate_resolution_confidence(ctx, responses, corrected_claim)
        next_action, rerun = self._determine_next_action(ctx, corrected_claim, resolution_conf)
        # 5) Build result
        result = ClarificationResult(
            original_claim=ctx.claim_text,
            corrected_claim=corrected_claim,
            questions=questions,
            responses=responses or {},
            resolution_confidence=resolution_conf,
            next_action=next_action,
            reasoning=reasoning,
            issue_type=ctx.issue_type,
            rerun_verification=rerun,
        )
        return result

    # ------------------------ Internal helpers ------------------------

    def _generate_questions(self, ctx: ClarificationContext, max_questions: int) -> List[SocraticQuestion]:
        gen = GENERATOR_BY_ISSUE.get(ctx.issue_type)
        if gen is None:
            raise ValueError(f"Unsupported issue type: {ctx.issue_type}")
        questions = gen.generate_questions(ctx, max_questions=max_questions)
        questions = self._validate_questions(questions)
        if clar_cfg.REFINE_QUESTIONS_WITH_LLM and self.llm is not None:
            try:
                questions = self._refine_questions_with_llm(ctx, questions)
            except Exception as e:
                logger.warning("LLM refinement failed; using base questions. Error: %s", e)
        return questions

    def _refine_questions_with_llm(self, ctx: ClarificationContext, questions: List[SocraticQuestion]) -> List[SocraticQuestion]:
        """Use LLM to sharpen questions into concise, Socratic prompts targeted to resolve the issue."""
        raw_questions = [q.text for q in questions]
        system_prompt = (
            "You are Socrates. Rewrite the given questions so that each is precise, concise (<=25 words), "
            "and directly targeted to resolve the described issue. Avoid fluff. Do not add hints. "
            "Return JSON with field 'questions' as an array of strings of the same length as input."
        )
        payload = {
            "claim": ctx.claim_text,
            "issue_type": ctx.issue_type.value,
            "category": ctx.category.value if hasattr(ctx.category, 'value') else str(ctx.category),
            "failed_check_type": ctx.failed_check_type,
            "fact_check": {
                "verdict": ctx.fact_check.verdict,
                "confidence": ctx.fact_check.confidence,
                "reasoning": ctx.fact_check.reasoning,
            },
            "questions": raw_questions,
        }
        prompt = (
            "Task: Refine Socratic questions for claim clarification\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Output only valid JSON: {\"questions\": [...]}"
        )
        text = self.llm.generate_text(prompt=prompt, system_prompt=system_prompt, max_tokens=512, temperature=0.2)  # type: ignore
        try:
            obj = self._parse_json_safely(text)
            # Allow direct list as output
            refined = obj.get("questions", obj if isinstance(obj, list) else [])
            out: List[SocraticQuestion] = []
            # Align lengths safely
            pair_count = min(len(refined), len(questions))
            for i in range(pair_count):
                q = refined[i]
                if isinstance(q, str) and q.strip():
                    base = questions[i]
                    out.append(
                        SocraticQuestion(
                            id=base.id,
                            text=q.strip(),
                            qtype=base.qtype,
                            choices=base.choices,
                            expects=base.expects,
                            metadata=base.metadata,
                        )
                    )
            # If nothing valid parsed, fall back
            return self._validate_questions(out) if out else questions
        except Exception as e:
            logger.warning("Failed to parse refined questions JSON: %s", e)
            return questions

    def _validate_questions(self, questions: List[SocraticQuestion]) -> List[SocraticQuestion]:
        seen = set()
        cleaned: List[SocraticQuestion] = []
        for q in questions:
            t = (q.text or "").strip()
            if not t:
                continue
            if len(t) > 200:
                t = t[:200].rstrip() + "…"
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            cleaned.append(SocraticQuestion(id=q.id, text=t, qtype=q.qtype, choices=q.choices, expects=q.expects, metadata=q.metadata))
        return cleaned

    def _collect_responses(
        self,
        questions: List[SocraticQuestion],
        response_provider: Optional[ResponseProvider],
    ) -> Dict[str, Any]:
        # If a programmatic response provider is supplied, use it
        if response_provider is not None:
            responses = {}
            for q in questions:
                try:
                    responses[q.id] = response_provider(q)
                except Exception as e:
                    logger.warning("Response provider failed for %s: %s", q.id, e)
                    responses[q.id] = None
            return responses
        # If user rewrite is NOT required, avoid any interactive prompts (auto mode)
        if not clar_cfg.REQUIRE_USER_REWRITE:
            return {q.id: None for q in questions}
        # Otherwise, fall back to interactive CLI if available
        try:
            import sys
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                from .user_interface import present_questions_interactive
                return present_questions_interactive(questions)
        except Exception:
            pass
        # Non-interactive default (no TTY or UI failure)
        return {q.id: None for q in questions}

    def _summarize_evidence(self, ctx: ClarificationContext, limit: int = 3) -> List[str]:
        ev_summaries: List[str] = []
        try:
            for e in ctx.fact_check.evidence[:limit]:
                if isinstance(e, dict):
                    s = (e.get("summary") or e.get("title") or e.get("text") or "").strip()
                    if s:
                        ev_summaries.append(s)
                elif isinstance(e, str):
                    if e.strip():
                        ev_summaries.append(e.strip())
        except Exception:
            pass
        return ev_summaries

    # ------------------------ Parsing & text utils ------------------------
    def _strip_code_fences(self, text: str) -> str:
        t = (text or "").strip()
        # Remove backticks and leading language hints
        # ```json ... ``` or ``` ... ```
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
            t = re.sub(r"```$", "", t).strip()
        # Remove markdown inline code fences
        t = t.strip("`")
        # Remove leading `json\n`
        if t.lower().startswith("json\n"):
            t = t[5:]
        return t.strip()

    def _parse_json_safely(self, text: str) -> Any:
        """Extract and parse JSON even if surrounded by prose or fences.
        Tries direct parse, then searches for the first well-formed object/array.
        Raises ValueError if parsing fails.
        """
        t = self._strip_code_fences(text)
        # Direct attempt
        try:
            return json.loads(t)
        except Exception:
            pass

        # Search for JSON objects/arrays by tracking brackets
        candidates: List[str] = []
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            stack = 0
            start_idx = None
            for i, ch in enumerate(t):
                if ch == open_ch:
                    if stack == 0:
                        start_idx = i
                    stack += 1
                elif ch == close_ch and stack > 0:
                    stack -= 1
                    if stack == 0 and start_idx is not None:
                        candidates.append(t[start_idx : i + 1])
                        start_idx = None
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                continue
        raise ValueError("No valid JSON object/array found in LLM output")

    def _tokenize(self, text: str) -> List[str]:
        # Split into words and punctuation tokens
        return re.findall(r"\w+|[^\w\s]", text or "")

    def _detokenize(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        s = " ".join(tokens)
        # Fix spaces before punctuation
        s = re.sub(r"\s+([,.;:!?])", r" \1", s)
        s = re.sub(r"\(\s+", "(", s)
        s = re.sub(r"\s+\)", ")", s)
        s = s.replace(" 's", "'s")
        s = re.sub(r"\s+\-\s+", "-", s)
        return s.strip()

    def _selective_replace_tokens(self, original: str, proposed: str) -> str:
        """Perform conservative token-level replacement from proposed into original.
        Honors config thresholds; may ignore insertions unless allowed.
        """
        if not original or not proposed:
            return proposed or original

        orig_tokens = self._tokenize(original)
        prop_tokens = self._tokenize(proposed)
        if len(orig_tokens) < clar_cfg.SELECTIVE_MIN_TOKENS:
            return proposed

        sm_char = difflib.SequenceMatcher(None, original, proposed)
        char_diff_ratio = 1.0 - sm_char.ratio()
        if char_diff_ratio <= clar_cfg.SELECTIVE_MAX_CHAR_DIFF_RATIO:
            # Small change overall; accept proposed
            out = proposed
        else:
            sm = difflib.SequenceMatcher(a=orig_tokens, b=prop_tokens)
            changed = 0
            out_tokens: List[str] = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    out_tokens.extend(orig_tokens[i1:i2])
                elif tag == 'replace':
                    changed += (i2 - i1)
                    # Replace with proposed span
                    out_tokens.extend(prop_tokens[j1:j2])
                elif tag == 'delete':
                    changed += (i2 - i1)
                    if clar_cfg.SELECTIVE_ALLOW_INSERTIONS:
                        # deletion from original => drop tokens
                        pass
                    else:
                        # keep original tokens (reject deletion)
                        out_tokens.extend(orig_tokens[i1:i2])
                elif tag == 'insert':
                    ins_count = (j2 - j1)
                    if clar_cfg.SELECTIVE_ALLOW_INSERTIONS:
                        out_tokens.extend(prop_tokens[j1:j2])
                    else:
                        # ignore insertions
                        pass
            change_ratio = changed / max(1, len(orig_tokens))
            if change_ratio > clar_cfg.SELECTIVE_MAX_TOKEN_CHANGE_RATIO:
                # Too many changes; prefer original with minimal replacements allowed above
                out_tokens = orig_tokens if not clar_cfg.SELECTIVE_ALLOW_INSERTIONS else out_tokens
            out = self._detokenize(out_tokens)

        # Enforce token cap
        tok_cap = clar_cfg.MAX_CORRECTED_CLAIM_TOKENS
        final_tokens = self._tokenize(out)
        if len(final_tokens) > tok_cap:
            out = self._detokenize(final_tokens[:tok_cap])
        return out.strip()

    def _evaluate_user_answer_with_llm(
        self,
        ctx: ClarificationContext,
        qa_pairs: List[Dict[str, Any]],
        user_corrected_claim: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to evaluate the user's answer/rewrite against the evidence.

        Returns a dict like {consistency, confidence, reasoning}. Does NOT change the claim.
        """
        if self.llm is None:
            return None
        payload = {
            "original_claim": ctx.claim_text,
            "user_corrected_claim": user_corrected_claim,
            "issue_type": getattr(ctx.issue_type, "value", str(ctx.issue_type)),
            "fact_check": {
                "verdict": getattr(ctx.fact_check, "verdict", None),
                "confidence": getattr(ctx.fact_check, "confidence", None),
                "reasoning": getattr(ctx.fact_check, "reasoning", None),
            },
            "evidence_summaries": self._summarize_evidence(ctx),
            "qa": qa_pairs,
        }
        system = (
            "You are an evaluator. Compare the user's clarified/corrected claim to the evidence summaries. "
            "Determine if it is SUPPORTED, CONFLICTING, or INSUFFICIENT relative to the evidence. "
            "Output ONLY JSON with: {consistency: SUPPORTED|CONFLICTING|INSUFFICIENT, confidence: 0..1, reasoning: string}."
        )
        prompt = f"Input JSON to evaluate (be strict, concise):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        try:
            text = self.llm.generate_text(prompt=prompt, system_prompt=system, max_tokens=256, temperature=0.1)  # type: ignore
            obj = self._parse_json_safely(text)
            consistency = (obj.get("consistency") or "").strip().upper()
            confidence = float(obj.get("confidence", 0.0) or 0.0)
            reasoning = (obj.get("reasoning") or "").strip()
            if consistency not in {"SUPPORTED", "CONFLICTING", "INSUFFICIENT"}:
                consistency = "INSUFFICIENT"
            confidence = max(0.0, min(1.0, confidence))
            return {"consistency": consistency, "confidence": confidence, "reasoning": reasoning}
        except Exception:
            return None

    def _process_responses_and_correct_claim(
        self,
        ctx: ClarificationContext,
        questions: List[SocraticQuestion],
        responses: Dict[str, Any],
    ) -> (Optional[str], str):
        """Use LLM to produce a corrected claim based on user responses.
        Returns (corrected_claim, reasoning).
        """
        # Compose compact Q/A list
        qa_pairs = []
        for q in questions:
            ans = responses.get(q.id, None)
            qa_pairs.append({"question": q.text, "answer": ans})

        # Enforce user-only correction when configured
        if clar_cfg.REQUIRE_USER_REWRITE:
            corrected_claim = self._fallback_correction_from_responses(questions, responses, default=None)
            reasoning = "User-provided rewrite required; no model auto-correction."
            # Evaluate user's answer vs evidence with LLM (for reasoning/confidence), without altering claim
            try:
                eval_res = self._evaluate_user_answer_with_llm(ctx, qa_pairs, corrected_claim)
                if eval_res:
                    reasoning = (
                        f"{reasoning} Evaluation: {eval_res['consistency']} (conf {eval_res['confidence']:.2f}). "
                        f"{eval_res['reasoning']}"
                    )
            except Exception:
                pass
            return corrected_claim, reasoning

        system = (
            "You are a precise claim correction assistant. Given a claim, its issue type, and user answers to Socratic questions, "
            "produce a single corrected claim that is specific, verifiable, and resolves the issue. "
            "If the original claim is already correct after clarification, return it unchanged. "
            "Output ONLY JSON with fields: corrected_claim (string), reasoning (string)."
        )
        prompt = (
            f"Original claim: {ctx.claim_text}\n"
            f"Issue type: {ctx.issue_type.value}\n"
            f"Category: {getattr(ctx.category, 'name', str(ctx.category))}\n"
            f"Failed check: {ctx.failed_check_type}\n"
            f"Fact-check verdict: {ctx.fact_check.verdict} (conf {ctx.fact_check.confidence:.2f})\n"
            f"LLM opinion: {ctx.fact_check.reasoning or ''}\n"
            f"Q/A: {json.dumps(qa_pairs, ensure_ascii=False)}\n\n"
            "Return JSON now."
        )
        corrected_claim = None
        reasoning = ""
        if self.llm is None or not clar_cfg.CORRECT_CLAIM_WITH_LLM:
            # Fallback: try to use user's rewrite if provided
            corrected_claim = self._fallback_correction_from_responses(questions, responses, default=None)
            reasoning = "LLM disabled; used user-provided clarification."
            return corrected_claim, reasoning

        text = self.llm.generate_text(prompt=prompt, system_prompt=system, max_tokens=256, temperature=0.2)  # type: ignore
        try:
            obj = self._parse_json_safely(text)
            corrected_claim = obj.get("corrected_claim")
            reasoning = obj.get("reasoning", "")
            if not corrected_claim or not isinstance(corrected_claim, str):
                corrected_claim = self._fallback_correction_from_responses(questions, responses, default=ctx.claim_text)
                reasoning = reasoning or "Fallback applied due to invalid LLM output."
        except Exception as e:
            logger.warning("Failed to parse corrected claim JSON: %s", e)
            corrected_claim = self._fallback_correction_from_responses(questions, responses, default=ctx.claim_text)
            reasoning = "Fallback applied due to JSON parsing error."

        # Apply selective replacement when enabled and appropriate
        try:
            if corrected_claim and isinstance(corrected_claim, str) and corrected_claim.strip():
                if clar_cfg.SELECTIVE_TOKEN_REPLACEMENT:
                    merged = self._selective_replace_tokens(ctx.claim_text, corrected_claim)
                    if merged != corrected_claim:
                        reasoning = (reasoning + " | Applied selective token replacement").strip()
                    corrected_claim = merged
        except Exception as e:
            logger.warning("Selective replacement failed; using raw corrected claim: %s", e)

        return corrected_claim, reasoning

    def _fallback_correction_from_responses(
        self,
        questions: List[SocraticQuestion],
        responses: Dict[str, Any],
        default: Optional[str] = None,
    ) -> Optional[str]:
        # If user provided a rewritten precise claim, prefer it
        for q in questions:
            expects = (q.expects or "").lower()
            if (
                expects.startswith("rewrite_precise_claim")
                or "rewrite" in expects
                or "propose_correction" in expects
                or "revise" in expects
                or "correction" in expects
            ):
                ans = responses.get(q.id)
                if isinstance(ans, str) and len(ans.split()) >= 3:
                    return ans.strip()
        # When strict mode is on, never auto-use defaults
        if clar_cfg.REQUIRE_USER_REWRITE:
            return None
        return default

    def _calculate_resolution_confidence(
        self,
        ctx: ClarificationContext,
        responses: Dict[str, Any],
        corrected_claim: Optional[str],
    ) -> float:
        score = 0.5  # base
        # Response quality heuristics
        answered = sum(1 for v in responses.values() if v not in (None, ""))
        score += 0.1 * min(answered, 3)
        # If corrected claim is significantly more specific than original
        if corrected_claim and corrected_claim.strip() and corrected_claim.strip() != ctx.claim_text.strip():
            score += 0.15
        # Issue-specific adjustments
        if ctx.issue_type == IssueType.AMBIGUITY:
            score += 0.05
        elif ctx.issue_type == IssueType.EXTERNAL_FACTUAL_CONFLICT:
            score -= 0.05
        # Clamp and thresholds
        score = max(0.0, min(1.0, score))
        return score

    def _determine_next_action(
        self,
        ctx: ClarificationContext,
        corrected_claim: Optional[str],
        confidence: float,
    ) -> (ResolutionAction, bool):
        # If no user-provided correction is available, re-run verification
        if clar_cfg.REQUIRE_USER_REWRITE and not corrected_claim:
            return ResolutionAction.REVERIFY_PIPELINE, True
        # If user provided a corrected claim, prefer adding to KG directly
        if clar_cfg.REQUIRE_USER_REWRITE and corrected_claim:
            return ResolutionAction.DIRECT_TO_KG, False
        # Development overrides could be added via ctx.metadata
        override = ctx.metadata.get("next_action_override") if ctx.metadata else None
        if override in {a.name for a in ResolutionAction}:
            act = ResolutionAction[override]
            return act, (act == ResolutionAction.REVERIFY_PIPELINE)

        # Default policy lookup
        policy_str = self.route_policy.get(ctx.issue_type.value, "REVERIFY_PIPELINE")
        policy = ResolutionAction[policy_str]

        # Confidence-based refinement
        if confidence >= clar_cfg.HIGH_CONFIDENCE_THRESHOLD and ctx.issue_type in {
            IssueType.AMBIGUITY, IssueType.KNOWLEDGE_CONTRADICTION
        }:
            # If we are highly confident the user clarified/corrected properly, send to KG directly
            return ResolutionAction.DIRECT_TO_KG, False
        if confidence < clar_cfg.MIN_RESOLUTION_CONFIDENCE:
            # Low confidence: reverify to be safe
            return ResolutionAction.REVERIFY_PIPELINE, True
        # Otherwise use policy
        return policy, (policy == ResolutionAction.REVERIFY_PIPELINE)
