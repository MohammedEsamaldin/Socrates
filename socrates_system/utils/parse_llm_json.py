"""Shared utility for extracting JSON from LLM responses.

LLM output frequently wraps JSON in markdown fences, adds prose preamble,
uses escape artefacts, or omits quotes around bare identifiers. This module
centralises all of that recovery logic in one tested place.

Usage:
    from socrates_system.utils.parse_llm_json import parse_llm_json

    data = parse_llm_json(llm_response)   # raises ValueError on total failure
"""
from __future__ import annotations

import json
import re
from typing import Any, List


def parse_llm_json(text: str) -> Any:
    """Extract and parse the first valid JSON value from an LLM response.

    Handles markdown fences, prose preamble, escape artefacts, and bare-word
    identifiers. Arrays are preferred over objects when both are present at
    the same nesting level.

    Raises:
        ValueError: if no valid JSON object or array can be found.
    """
    t = _strip_fences(text)
    t = _sanitize(t)

    # Attempt 1 — direct parse on the cleaned string
    try:
        return json.loads(t)
    except Exception:
        pass

    # Attempt 2 — balanced-bracket extraction (arrays preferred over objects)
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        for candidate in _extract_balanced(t, open_ch, close_ch):
            try:
                return json.loads(candidate)
            except Exception:
                continue

    # Attempt 3 — quote bare-word identifiers, then retry extraction
    t_quoted = _quote_bare_identifiers(t)
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        for candidate in _extract_balanced(t_quoted, open_ch, close_ch):
            try:
                return json.loads(candidate)
            except Exception:
                continue

    raise ValueError("No valid JSON object or array found in LLM output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences and stray backticks."""
    t = (text or "").strip()
    # ```json ... ``` or ``` ... ```
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"```\s*$", "", t).strip()
    # bare backtick fences
    t = t.strip("`")
    # leading "json\n" artefact some models emit
    if t.lower().startswith("json\n"):
        t = t[5:]
    return t.strip()


def _sanitize(text: str) -> str:
    """Fix common LLM JSON escaping artefacts."""
    # escaped underscores that break JSON parsers
    text = text.replace("\\_", "_")
    # literal \n / \t inside strings (not real escape sequences)
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    # collapse embedded real newlines and tabs
    text = text.replace("\n", " ").replace("\t", " ")
    # normalise whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> List[str]:
    """Return all top-level balanced substrings delimited by open_ch/close_ch."""
    candidates: List[str] = []
    stack = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == open_ch:
            if stack == 0:
                start = i
            stack += 1
        elif ch == close_ch and stack > 0:
            stack -= 1
            if stack == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None
    return candidates


def _quote_bare_identifiers(text: str) -> str:
    """Quote bare-word values that appear after a colon (JSON enum artefacts).

    Leaves JSON keywords (true, false, null) and numbers untouched.
    """
    # Quote common confidence-level tokens: "confidence": High → "confidence": "High"
    text = re.sub(
        r'("confidence"\s*:\s*)(Low|Medium|High)(\s*[,}])',
        r'\1"\2"\3',
        text,
    )

    def _maybe_quote(m: re.Match) -> str:  # type: ignore[type-arg]
        leading, ident, trailing = m.group(1), m.group(2), m.group(3)
        low = ident.lower()
        if low in {"true", "false", "null"}:
            return f"{leading}{ident}{trailing}"
        if re.fullmatch(r"-?\d+(?:\.\d+)?", ident):
            return f"{leading}{ident}{trailing}"
        return f'{leading}"{ident}"{trailing}'

    return re.sub(r"(:\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*[,}])", _maybe_quote, text)
