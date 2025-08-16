from __future__ import annotations

from typing import Any, Dict, List, Optional

from .data_models import SocraticQuestion
from . import config as clar_cfg


def _print_header():
    print("\n=== Clarification Required ===")
    print("Please answer the following questions to clarify your claim.")
    print("Tip: If asked to rewrite the claim, provide a single precise, verifiable sentence.")


def _format_choices(choices: Optional[List[str]]) -> str:
    if not choices:
        return ""
    return "\n".join([f"  [{i+1}] {c}" for i, c in enumerate(choices)])


def _read_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def present_questions_interactive(questions: List[SocraticQuestion]) -> Dict[str, Any]:
    """Interactive CLI prompt to collect user responses to Socratic questions.

    Returns a mapping from question.id to the user's answer. For selection-type
    questions, the user may enter the index or the text of the choice.
    """
    _print_header()
    responses: Dict[str, Any] = {}

    for idx, q in enumerate(questions, start=1):
        print(f"\nQ{idx}: {q.text}")
        # If the generator provided a justification, show it to the user
        try:
            just = (q.metadata or {}).get("justification")
            if just:
                print(f"Why we're asking: {just}")
        except Exception:
            pass
        if q.choices:
            print(_format_choices(q.choices))
        if q.expects:
            hint = q.expects.replace("_", " ")
            print(f"Hint: {hint}")

        # Special hint for rewrites/corrections
        expects_lower = (q.expects or "").lower()
        if any(k in expects_lower for k in ["rewrite", "propose_correction", "revise", "correction"]):
            print("Note: Provide your corrected claim as one precise, verifiable sentence.")

        ans_raw = _read_input("> ").strip()
        if q.qtype == "selection" and q.choices:
            # Allow numeric index
            if ans_raw.isdigit():
                idx_choice = int(ans_raw) - 1
                if 0 <= idx_choice < len(q.choices):
                    responses[q.id] = q.choices[idx_choice]
                    continue
            # Or exact match on text
            if ans_raw:
                for c in q.choices:
                    if c.lower() == ans_raw.lower():
                        responses[q.id] = c
                        break
                else:
                    responses[q.id] = ans_raw  # free text fallback
            else:
                responses[q.id] = None
        else:
            responses[q.id] = ans_raw if ans_raw else None

    return responses
