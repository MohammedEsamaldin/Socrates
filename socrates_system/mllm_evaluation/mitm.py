from typing import Any, Dict, List, Tuple, Optional
from dataclasses import asdict
import os

from socrates_system.pipeline import SocratesPipeline
from socrates_system.modules.llm_manager import LLMManager


def _has_negation(text: Optional[str]) -> bool:
    """Heuristic detection of negation in a short claim/question.

    This is intentionally conservative: it only looks for common negation tokens
    and contractions. It is used to prevent unintended polarity flips when
    applying clarification corrections unless explicitly allowed.
    """
    if not text:
        return False
    s = f" {str(text).strip().lower()} "
    # Common negation cues
    neg_terms = [
        " not ", " no ", " never ", " none ", " without ", " nothing ", " nowhere ",
        "n't ", "n't,", "n't.", "n't?",  # contractions
    ]
    return any(t in s for t in neg_terms)


def _compute_corrected_text(original_text: str,
                            claims: List[Any],
                            clar_results: Dict[int, Dict[str, Any]],
                            stage: str = "pre",
                            min_conf: Optional[float] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build minimally-edited text by replacing only hallucinated tokens at claim spans.

    Args:
        original_text: the text used for extraction
        claims: list of ExtractedClaim objects returned by pipeline.run
        clar_results: pipeline._clarification_results mapping (1-indexed claim_id -> {"pre": obj, "post": obj})
        stage: "pre" to apply pre-routing corrections (for user input), "post" for post-factuality (for model output)
        min_conf: minimum resolution_confidence needed to apply a correction (overrides env SOC_MITM_MIN_CONF)

    Returns:
        (corrected_text, corrections)
        corrections: list of {claim_index, span, original, corrected}
    """
    # Resolve confidence threshold
    if min_conf is None:
        try:
            min_conf = float(os.getenv("SOC_MITM_MIN_CONF", "0.55"))
        except Exception:
            min_conf = 0.55

    # Gather replacements as (start, end, replacement, original, idx)
    repls: List[Tuple[int, int, str, str, int]] = []
    for i, cl in enumerate(claims, 1):
        cr = (clar_results or {}).get(i, {})
        ctx_obj = cr.get(stage)
        corrected = getattr(ctx_obj, "corrected_claim", None) if ctx_obj else None
        conf = float(getattr(ctx_obj, "resolution_confidence", 0.0) or 0.0) if ctx_obj else 0.0
        if (
            corrected and isinstance(corrected, str) and corrected.strip() and corrected != cl.text
            and conf >= (min_conf or 0.0)
        ):
            # Polarity flip guard (unless explicitly allowed)
            try:
                allow_flip_env = os.getenv("SOC_ALLOW_POLARITY_FLIP", "false").lower() == "true"
            except Exception:
                allow_flip_env = False
            if not allow_flip_env:
                try:
                    orig_neg = _has_negation(getattr(cl, "text", ""))
                    corr_neg = _has_negation(corrected)
                    if orig_neg != corr_neg:
                        # Skip this correction to avoid unintended polarity flip
                        continue
                except Exception:
                    # Best-effort guard only
                    pass
            start, end = int(getattr(cl, "start_char", 0)), int(getattr(cl, "end_char", 0))
            if 0 <= start < end <= len(original_text):
                repls.append((start, end, corrected, cl.text, i))
    if not repls:
        return original_text, []

    # Apply replacements left-to-right without overlap
    repls.sort(key=lambda x: x[0])
    out_parts: List[str] = []
    cursor = 0
    corrections: List[Dict[str, Any]] = []
    for start, end, new_text, old_text, idx in repls:
        if start < cursor:
            # overlapping claim spans; skip to avoid corrupting text
            continue
        out_parts.append(original_text[cursor:start])
        out_parts.append(new_text)
        corrections.append({
            "claim_index": idx,
            "span": [start, end],
            "original": old_text,
            "corrected": new_text,
            # Note: individual per-claim confidence can be obtained from pipeline _clarification_results
        })
        cursor = end
    out_parts.append(original_text[cursor:])
    return ("".join(out_parts), corrections)


def process_user_turn(pipeline: SocratesPipeline, text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
    """Run pipeline on user text, and minimally correct the text based on pre-routing clarifications.

    Args:
        pipeline: SocratesPipeline instance
        text: user text
        image_path: optional image path for multimodal verification
    """
    # Determine whether to apply MitM corrections on input
    use_mitm = os.getenv("SOC_USE_MITM", "true").lower() == "true"
    verify_input = os.getenv("SOC_MITM_VERIFY_INPUT", "true").lower() == "true"

    # Optionally disable pre-evidence extraction (MME preference):
    # - suppress external factuality and self-consistency/conflict resolution
    # - omit image to avoid cross-modal evidence before Q0
    disable_pre = os.getenv("SOC_DISABLE_PRE_EVIDENCE", "true").lower() == "true"
    _restore_after = False
    _orig_factuality_enabled = None
    _orig_self_checker = None
    _orig_conflict_resolver = None
    try:
        # Tag stage for downstream DEBUG logging
        try:
            setattr(pipeline, "_verification_stage", "PRE_Q0")
        except Exception:
            pass
        run_image = image_path
        if disable_pre:
            _restore_after = True
            try:
                _orig_factuality_enabled = getattr(pipeline, "factuality_enabled", True)
                _orig_self_checker = getattr(pipeline, "self_checker", None)
                _orig_conflict_resolver = getattr(pipeline, "conflict_resolver", None)
            except Exception:
                pass
            try:
                setattr(pipeline, "factuality_enabled", False)
            except Exception:
                pass
            try:
                setattr(pipeline, "self_checker", None)
            except Exception:
                pass
            try:
                setattr(pipeline, "conflict_resolver", None)
            except Exception:
                pass
            # Avoid cross-modal fetches by omitting the image
            run_image = None

        claims = pipeline.run(text, image_path=run_image)
    finally:
        if _restore_after:
            try:
                if _orig_factuality_enabled is not None:
                    setattr(pipeline, "factuality_enabled", _orig_factuality_enabled)
            except Exception:
                pass
            try:
                setattr(pipeline, "self_checker", _orig_self_checker)
            except Exception:
                pass
            try:
                setattr(pipeline, "conflict_resolver", _orig_conflict_resolver)
            except Exception:
                pass
    clar = getattr(pipeline, "_clarification_results", {})
    if use_mitm and verify_input:
        corrected_text, corrections = _compute_corrected_text(text, claims, clar, stage="pre")
    else:
        corrected_text, corrections = text, []

    # Evidence-informed Q1 rewrite (optional) and fallback neutral rewrite
    try:
        evidence_mode = os.getenv("SOC_Q1_EVIDENCE_INFORMED", "false").lower() == "true"
        always_rw = os.getenv("SOC_Q1_EVIDENCE_ALWAYS_REWRITE", "false").lower() == "true"
        fallback_rw_on = os.getenv("SOC_FALLBACK_Q_REWRITE", "true").lower() == "true"

        # Optional pre-Q1 verification subprocess to guide a neutral, non-leading rewrite
        ev_counts = {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0}
        ev_digest_text: str = ""
        negative_signal: bool = False
        if evidence_mode:
            _restore_ev = False
            _orig_fact = None
            _orig_self = None
            _orig_conf = None
            try:
                # Cross-modal-only precheck for neutral, non-leading cues
                _restore_ev = True
                try:
                    _orig_fact = getattr(pipeline, "factuality_enabled", True)
                    _orig_self = getattr(pipeline, "self_checker", None)
                    _orig_conf = getattr(pipeline, "conflict_resolver", None)
                except Exception:
                    pass
                try:
                    setattr(pipeline, "factuality_enabled", False)
                except Exception:
                    pass
                try:
                    setattr(pipeline, "self_checker", None)
                except Exception:
                    pass
                try:
                    setattr(pipeline, "conflict_resolver", None)
                except Exception:
                    pass
                try:
                    setattr(pipeline, "_verification_stage", "Q1_PRECHECK")
                except Exception:
                    pass
                # Only run if image is available; otherwise, just proceed with generic guidance
                if image_path:
                    _ = pipeline.run(text, image_path=image_path)
                    ev = getattr(pipeline, "_last_factuality_results", {}) or {}
                    digest_lines: List[str] = []
                    for _, v in (ev.items() if isinstance(ev, dict) else []):
                        st = str((v or {}).get("status", "")).upper() or "UNCERTAIN"
                        if st in ev_counts:
                            ev_counts[st] += 1
                        try:
                            conf = float((v or {}).get("confidence", 0.0) or 0.0)
                        except Exception:
                            conf = 0.0
                        vis = (v or {}).get("visual_description") or (v or {}).get("visual_summary") or ""
                        vis = str(vis or "")
                        if len(vis) > 120:
                            vis = vis[:117] + "..."
                        # Evidence strings
                        ev_list = (v or {}).get("evidence", []) or []
                        ev_texts: List[str] = []
                        for e in ev_list[:2]:
                            if isinstance(e, dict):
                                s = e.get("summary") or e.get("text") or str(e)
                            else:
                                s = str(e)
                            if s:
                                ev_texts.append(s)
                        ev_join = "; ".join(ev_texts)
                        if len(ev_join) > 160:
                            ev_join = ev_join[:157] + "..."
                        # Contradictions
                        cons = (v or {}).get("contradictions", []) or []
                        con_texts: List[str] = []
                        for c in cons[:2]:
                            if isinstance(c, dict):
                                cs = c.get("summary") or c.get("text") or str(c)
                            else:
                                cs = str(c)
                            if cs:
                                con_texts.append(cs)
                        con_join = "; ".join(con_texts)
                        if len(con_join) > 160:
                            con_join = con_join[:157] + "..."
                        digest_lines.append(f"- {st} (conf {conf:.2f}); visual: {vis or 'n/a'}; evidence: {ev_join or 'n/a'}; contradictions: {con_join or 'n/a'}")
                    # Mark negative signal for downstream guidance
                    negative_signal = (ev_counts.get('FAIL', 0) > 0) and (ev_counts.get('PASS', 0) == 0)
                    if digest_lines:
                        ev_digest_text = "\n".join(digest_lines)
                        if len(ev_digest_text) > 600:
                            ev_digest_text = ev_digest_text[:597] + '...'
                # else: keep ev_counts at zeros
            finally:
                if _restore_ev:
                    try:
                        if _orig_fact is not None:
                            setattr(pipeline, "factuality_enabled", _orig_fact)
                    except Exception:
                        pass
                    try:
                        setattr(pipeline, "self_checker", _orig_self)
                    except Exception:
                        pass
                    try:
                        setattr(pipeline, "conflict_resolver", _orig_conf)
                    except Exception:
                        pass

        # Decide whether to rewrite
        should_rewrite = False
        if evidence_mode:
            if always_rw:
                should_rewrite = True
            else:
                # Rewrite if fallback condition holds, or if precheck found any non-PASS signal
                has_issue = (ev_counts["FAIL"] > 0) or (ev_counts["UNCERTAIN"] > 0)
                if has_issue or ((not corrections) and corrected_text == text and fallback_rw_on):
                    should_rewrite = True
        else:
            # Original fallback behavior
            should_rewrite = (not corrections) and corrected_text == text and fallback_rw_on

        if should_rewrite:
            llm = getattr(pipeline, "llm_manager", None)
            if llm is not None:
                # Build guidance from the original text
                guidance_lines: List[str] = []
                t_low = (text or "").lower()
                pronouns = ["it", "they", "this", "that", "these", "those", "there"]
                pron_found = [p for p in pronouns if (f" {p} " in t_low) or t_low.startswith(p + " ") or t_low.endswith(" " + p)]
                if pron_found:
                    guidance_lines.append(
                        f"Resolve ambiguous pronouns ({', '.join(sorted(set(pron_found)))}) by restating the referent neutrally using only terms from the original question."
                    )
                if " and " in t_low or ";" in t_low or "," in t_low:
                    guidance_lines.append("If the question contains multiple parts, reduce it to a single clear Yes/No query without adding information.")
                if " any " in t_low:
                    guidance_lines.append("Avoid leading phrasing like 'any'; ask neutrally about presence or absence.")
                guidance_lines.append("Do not introduce details, attributes, or nouns not present in the original question.")
                guidance_lines.append("Preserve the original objects, attributes, and relationships; use the same nouns and modifiers as in the original question.")
                guidance_lines.append("Do not reveal or imply an answer; keep the wording neutral and verifier-friendly.")
                guidance_lines.append("Keep it a single sentence ending with a question mark.")

                # Evidence-informed neutral cues (do not disclose outcomes)
                if evidence_mode:
                    if ev_counts["FAIL"] > 0 or ev_counts["UNCERTAIN"] > 0:
                        guidance_lines.extend([
                            "Make the subject of the question visually explicit using only terms from the original question.",
                            "Avoid presuppositions; ask about what is visible rather than assuming facts.",
                            "Focus on a single, directly checkable property in the image (e.g., presence, color, count).",
                            "Avoid vague quantifiers or broad scopes; specify exactly what the question refers to using existing terms.",
                        ])
                    elif ev_counts["PASS"] > 0 and ev_counts["FAIL"] == 0:
                        guidance_lines.extend([
                            "Keep the meaning unchanged while removing residual ambiguity.",
                            "Maintain a crisp Yes/No formulation that remains easy to verify visually.",
                        ])
                    # Conditional polarity guidance based on evidence and env policy
                    try:
                        allow_flip_env_prompt = os.getenv("SOC_ALLOW_POLARITY_FLIP", "false").lower() == "true"
                    except Exception:
                        allow_flip_env_prompt = False
                    if negative_signal:
                        if allow_flip_env_prompt:
                            guidance_lines.append("If the evidence indicates absence of the claimed object/attribute, you may use explicit negation using only the original terms (e.g., 'Is it true that there is no ...?').")
                        else:
                            guidance_lines.append("Do not change polarity; ask neutrally about presence using only the original terms.")

                system = (
                    "You are a verifier-oriented question rewriter. Your goal is to make the question explicit, "
                    "unambiguous, and strictly non-leading for downstream verification. Preserve original objects, attributes, and relationships. "
                    "Use the verifier evidence summary ONLY to disambiguate and, if allowed, adjust presence/absence; do not quote it verbatim or add new entities. "
                    "Follow the rules strictly and do not add external knowledge or hints about the answer. Return ONLY the rewritten question."
                )
                evidence_section = ""
                if evidence_mode and ev_digest_text:
                    evidence_section = (
                        "Verifier evidence summary (for your reference; do not quote verbatim):\n" + ev_digest_text + "\n\n"
                    )
                prompt = (
                    evidence_section +
                    "Original question:\n" + text + "\n\n" +
                    "Rewrite the question to be explicit and non-leading while preserving the intended meaning.\n" +
                    "Apply these constraints:\n" +
                    "\n".join([f"- {gl}" for gl in guidance_lines]) +
                    "\n- Single Yes/No question only.\n" +
                    "Return only the rewritten question."
                )
                try:
                    max_tok = int(os.getenv("SOC_Q_REWRITE_MAX_TOKENS", "64"))
                except Exception:
                    max_tok = 64
                rw = llm.generate_text(prompt=prompt, system_prompt=system, max_tokens=max_tok, temperature=0.2)  # type: ignore
                if isinstance(rw, str) and rw.strip():
                    cleaned = rw.strip().splitlines()[0].strip()
                    if not cleaned.endswith("?"):
                        cleaned = cleaned.rstrip(".") + "?"
                    # Ensure the rewrite is non-trivial and still looks like a question
                    if len(cleaned.split()) >= 3:
                        corrected_text = cleaned
    except Exception:
        # Be conservative: ignore rewrite failures silently
        pass
    return {
        "claims": claims,
        "clarification": clar,
        "corrected_text": corrected_text,
        "corrections": corrections,
        "factuality": getattr(pipeline, "_last_factuality_results", {}),
        "image_path": image_path,
    }


def process_model_turn(pipeline: SocratesPipeline,
                       text: str,
                       image_path: Optional[str] = None,
                       hypothesis_mode: bool = False) -> Dict[str, Any]:
    """Run pipeline on model output, and minimally correct the text based on post-factuality clarifications.

    Args:
        pipeline: SocratesPipeline instance
        text: model output text
        image_path: optional image path for multimodal verification
        hypothesis_mode: when True (and SOC_HYPOTHESIS_CROSS_MODAL_ONLY is true), suppress external
            knowledge and self-consistency checks so that only cross-modal verification is performed.
    """
    # Determine whether to apply MitM corrections on output
    use_mitm = os.getenv("SOC_USE_MITM", "true").lower() == "true"
    verify_output = os.getenv("SOC_MITM_VERIFY_OUTPUT", "true").lower() == "true"

    # Optionally suppress non-cross-modal verification when evaluating derived hypotheses
    _restore_after = False
    _orig_factuality_enabled = None
    _orig_self_checker = None
    _orig_conflict_resolver = None
    try:
        if hypothesis_mode and os.getenv("SOC_HYPOTHESIS_CROSS_MODAL_ONLY", "true").lower() == "true":
            _restore_after = True
            try:
                _orig_factuality_enabled = getattr(pipeline, "factuality_enabled", True)
                _orig_self_checker = getattr(pipeline, "self_checker", None)
                _orig_conflict_resolver = getattr(pipeline, "conflict_resolver", None)
            except Exception:
                pass
            # Disable external factuality and self-consistency; cross-modal remains active
            try:
                setattr(pipeline, "factuality_enabled", False)
            except Exception:
                pass
            try:
                setattr(pipeline, "self_checker", None)
            except Exception:
                pass
            try:
                setattr(pipeline, "conflict_resolver", None)
            except Exception:
                pass

        # Tag stage for downstream DEBUG logging (may be overwritten by caller)
        try:
            if not getattr(pipeline, "_verification_stage", None):
                setattr(pipeline, "_verification_stage", "MODEL")
        except Exception:
            pass
        claims = pipeline.run(text, image_path=image_path)
        clar = getattr(pipeline, "_clarification_results", {})
        # Only apply post-output corrections if post-factuality clarifications are enabled in the pipeline
        if use_mitm and verify_output and getattr(pipeline, "post_factuality_clarification_enabled", True):
            corrected_text, corrections = _compute_corrected_text(text, claims, clar, stage="post")
        else:
            corrected_text, corrections = text, []
        return {
            "claims": claims,
            "clarification": clar,
            "corrected_text": corrected_text,
            "corrections": corrections,
            "factuality": getattr(pipeline, "_last_factuality_results", {}),
            "image_path": image_path,
        }
    finally:
        if _restore_after:
            # Restore original pipeline verification components
            try:
                if _orig_factuality_enabled is not None:
                    setattr(pipeline, "factuality_enabled", _orig_factuality_enabled)
            except Exception:
                pass
            try:
                setattr(pipeline, "self_checker", _orig_self_checker)
            except Exception:
                pass
            try:
                setattr(pipeline, "conflict_resolver", _orig_conflict_resolver)
            except Exception:
                pass


def build_pipeline(llm_manager: LLMManager,
                   factuality_enabled: bool = True,
                   clarification_enabled: bool = True,
                   clarification_dev_mode: bool = False,
                   question_gen_enabled: bool = False,
                   post_factuality_clarification_enabled: Optional[bool] = None) -> SocratesPipeline:
    return SocratesPipeline(
        llm_manager=llm_manager,
        factuality_enabled=factuality_enabled,
        clarification_enabled=clarification_enabled,
        clarification_dev_mode=clarification_dev_mode,
        question_gen_enabled=question_gen_enabled,
        post_factuality_clarification_enabled=post_factuality_clarification_enabled,
    )
