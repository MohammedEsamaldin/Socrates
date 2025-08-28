from typing import Any, Dict, List, Tuple, Optional
from dataclasses import asdict
import os

from socrates_system.pipeline import SocratesPipeline
from socrates_system.modules.llm_manager import LLMManager


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

    claims = pipeline.run(text, image_path=image_path)
    clar = getattr(pipeline, "_clarification_results", {})
    if use_mitm and verify_input:
        corrected_text, corrections = _compute_corrected_text(text, claims, clar, stage="pre")
    else:
        corrected_text, corrections = text, []

    # Fallback: if no corrections were applied (even if claims exist),
    # optionally attempt an LLM-based question rewrite to remove ambiguity.
    # This helps generate a meaningful Q1 prompt in dual-query flows.
    try:
        if (not corrections) and corrected_text == text and os.getenv("SOC_FALLBACK_Q_REWRITE", "true").lower() == "true":
            llm = getattr(pipeline, "llm_manager", None)
            if llm is not None:
                system = (
                    "You rewrite user questions to be precise, unambiguous Yes/No questions "
                    "without changing their meaning."
                )
                prompt = (
                    "Rewrite the following question to be explicit and unambiguous, preserving the intended meaning. "
                    "Keep it a single Yes/No question. Return only the rewritten question.\n\n"
                    f"Question: {text}"
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


def process_model_turn(pipeline: SocratesPipeline, text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
    """Run pipeline on model output, and minimally correct the text based on post-factuality clarifications.

    Args:
        pipeline: SocratesPipeline instance
        text: model output text
        image_path: optional image path for multimodal verification
    """
    # Determine whether to apply MitM corrections on output
    use_mitm = os.getenv("SOC_USE_MITM", "true").lower() == "true"
    verify_output = os.getenv("SOC_MITM_VERIFY_OUTPUT", "true").lower() == "true"

    claims = pipeline.run(text, image_path=image_path)
    clar = getattr(pipeline, "_clarification_results", {})
    if use_mitm and verify_output:
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


def build_pipeline(llm_manager: LLMManager,
                   factuality_enabled: bool = True,
                   clarification_enabled: bool = True,
                   clarification_dev_mode: bool = False,
                   question_gen_enabled: bool = False) -> SocratesPipeline:
    return SocratesPipeline(
        llm_manager=llm_manager,
        factuality_enabled=factuality_enabled,
        clarification_enabled=clarification_enabled,
        clarification_dev_mode=clarification_dev_mode,
        question_gen_enabled=question_gen_enabled,
    )
