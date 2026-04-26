"""
Issue-specific Socratic question generators for the clarification module.

Each generator subclasses :class:`BaseQuestionGenerator` and produces
questions tailored to a specific :class:`IssueType`. A module-level registry
``GENERATOR_BY_ISSUE`` maps each issue type to its singleton generator
instance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import uuid
import json

from .data_models import ClarificationContext, SocraticQuestion, IssueType

# Try to access the project's LLM manager
try:
    from socrates_system.modules.llm_manager import get_llm_manager
except Exception:
    get_llm_manager = None  # type: ignore

_CACHED_LLM = None


def _get_llm():
    """Lazily initialise and cache the shared LLM manager singleton.

    Returns:
        A shared LLMManager instance, or ``None`` if initialisation fails.
    """
    global _CACHED_LLM
    if _CACHED_LLM is not None:
        return _CACHED_LLM
    if get_llm_manager is not None:
        try:
            _CACHED_LLM = get_llm_manager()
        except Exception:
            _CACHED_LLM = None
    return _CACHED_LLM


def _make_q(
    text: str,
    qtype: str = "open-ended",
    choices: Optional[list] = None,
    expects: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> SocraticQuestion:
    """Construct a :class:`SocraticQuestion` with an auto-generated UUID id.

    Args:
        text: The question text.
        qtype: Question type; one of ``"open-ended"``, ``"binary"``,
            ``"selection"``.
        choices: List of choice strings for selection-type questions.
        expects: Hint describing the desired answer (e.g.
            ``"rewrite_precise_claim"``).
        metadata: Optional dict of additional metadata to attach.

    Returns:
        A new :class:`SocraticQuestion` instance.
    """
    return SocraticQuestion(
        id=str(uuid.uuid4())[:8],
        text=text.strip(),
        qtype=qtype,
        choices=choices,
        expects=expects,
        metadata=metadata or {},
    )


def _summarize_evidence(ctx: ClarificationContext, limit: int = 3) -> List[str]:
    """Extract up to ``limit`` short evidence summary strings from the context.

    Args:
        ctx: Clarification context whose ``fact_check.evidence`` is scanned.
        limit: Maximum number of items to return.

    Returns:
        List of non-empty summary strings extracted from evidence dicts or
        plain strings.
    """
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


def _llm_tailored_question(ctx: ClarificationContext) -> Optional[SocraticQuestion]:
    """Ask the LLM to craft ONE tailored Socratic question with a justification.

    The question should guide the user to provide a single precise, verifiable
    corrected claim (in the same answer) and explain why we ask.
    """
    llm = _get_llm()
    if llm is None:
        return None

    ev_summaries = _summarize_evidence(ctx)
    payload = {
        "claim": ctx.claim_text,
        "issue_type": getattr(ctx.issue_type, "value", str(ctx.issue_type)),
        "failed_check_type": ctx.failed_check_type,
        "fact_check": {
            "verdict": getattr(ctx.fact_check, "verdict", None),
            "confidence": getattr(ctx.fact_check, "confidence", None),
            "reasoning": getattr(ctx.fact_check, "reasoning", None),
        },
        "evidence_summaries": ev_summaries,
        "constraints": {
            "single_question": True,
            "question_max_words": 30,
            "justification_max_words": 60,
            "expects": "rewrite_precise_claim",
        },
    }

    system_prompt = (
        "You are Socrates helping to resolve a claim verification issue. "
        "Given the claim, the fact-check result, and evidence summaries, craft EXACTLY ONE concise question that asks "
        "the user to provide a single precise, verifiable corrected claim if needed. Also provide a brief justification "
        "that explains the conflict and why this question is necessary."
    )
    user_prompt = (
        "Output ONLY valid JSON with fields: {\n"
        "  \"question\": string (<=30 words),\n"
        "  \"expects\": string (use \"rewrite_precise_claim\" unless a more specific hint is necessary),\n"
        "  \"justification\": string (<=60 words, explain the conflict and why we ask)\n"
        "}.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )

    text = llm.generate_text(prompt=user_prompt, system_prompt=system_prompt, max_tokens=300, temperature=0.2)  # type: ignore
    try:
        cleaned = text.strip().strip("`")
        if cleaned.startswith("json\n"):
            cleaned = cleaned[5:]
        obj = json.loads(cleaned)
        q_text = (obj.get("question") or "").strip()
        expects = (obj.get("expects") or "rewrite_precise_claim").strip()
        justification = (obj.get("justification") or "").strip()
        if not q_text:
            return None
        return _make_q(
            q_text,
            qtype="open-ended",
            expects=expects,
            metadata={
                "justification": justification,
                "evidence_used": ev_summaries,
                "verdict": getattr(ctx.fact_check, "verdict", None),
            },
        )
    except Exception:
        return None


class SocraticQuestionGenerator(ABC):
    """Abstract base class for issue-specific Socratic question generators."""

    @abstractmethod
    def generate_questions(self, ctx: ClarificationContext, max_questions: int = 3) -> List[SocraticQuestion]:
        """Generate Socratic questions for a given clarification context.

        Args:
            ctx: Clarification context containing the claim, issue type, and
                fact-check result.
            max_questions: Maximum number of questions to return.

        Returns:
            List of :class:`SocraticQuestion` objects.
        """
        raise NotImplementedError


class VisualConflictQuestionGenerator(SocraticQuestionGenerator):
    """Generates questions for claims that conflict with visual evidence."""

    def generate_questions(self, ctx: ClarificationContext, max_questions: int = 3) -> List[SocraticQuestion]:
        """Generate Socratic questions that guide rewriting a visually conflicting claim.

        Args:
            ctx: Clarification context including the conflicting claim and evidence.
            max_questions: Maximum number of questions to return.

        Returns:
            A list of :class:`SocraticQuestion` instances, preferring an LLM-tailored
            question and falling back to a static rewrite prompt.
        """
        # Prefer LLM-tailored single question with justification
        q = _llm_tailored_question(ctx)
        if q is not None:
            return [q][:max_questions]
        # Fallback static single question guiding to rewrite
        return [
            _make_q(
                "Based on what you actually see, rewrite your claim as one precise, verifiable sentence (specify exact objects and relationships).",
                qtype="open-ended",
                expects="rewrite_precise_claim",
            )
        ][:max_questions]


class KnowledgeContradictionQuestionGenerator(SocraticQuestionGenerator):
    """Generates questions for claims that contradict session knowledge graph facts."""

    def generate_questions(self, ctx: ClarificationContext, max_questions: int = 3) -> List[SocraticQuestion]:
        """Generate questions prompting rewriting of a knowledge-contradicting claim.

        Args:
            ctx: Clarification context with the contradicting claim and KG evidence.
            max_questions: Maximum number of questions to return.

        Returns:
            A list of :class:`SocraticQuestion` instances.
        """
        q = _llm_tailored_question(ctx)
        if q is not None:
            return [q][:max_questions]
        return [
            _make_q(
                "Considering external knowledge, rewrite your claim as one precise, verifiable sentence with exact entities, dates, and sources if known.",
                qtype="open-ended",
                expects="rewrite_precise_claim",
            )
        ][:max_questions]


class AmbiguityQuestionGenerator(SocraticQuestionGenerator):
    """Generates questions for ambiguous claims that require clarification."""

    def generate_questions(self, ctx: ClarificationContext, max_questions: int = 3) -> List[SocraticQuestion]:
        """Generate questions that disambiguate an ambiguous claim.

        Args:
            ctx: Clarification context with the ambiguous claim.
            max_questions: Maximum number of questions to return.

        Returns:
            A list of :class:`SocraticQuestion` instances.
        """
        q = _llm_tailored_question(ctx)
        if q is not None:
            return [q][:max_questions]
        return [
            _make_q(
                "Resolve the ambiguity by rewriting your claim as one precise, verifiable sentence (define entities, timeframe, and location).",
                qtype="open-ended",
                expects="rewrite_precise_claim",
            )
        ][:max_questions]


class ExternalFactualConflictQuestionGenerator(SocraticQuestionGenerator):
    """Generates questions for claims that conflict with external factuality sources."""

    def generate_questions(self, ctx: ClarificationContext, max_questions: int = 3) -> List[SocraticQuestion]:
        """Generate questions that surface external evidence and prompt claim rewriting.

        Args:
            ctx: Clarification context including external fact-check evidence.
            max_questions: Maximum number of questions to return.

        Returns:
            A list of :class:`SocraticQuestion` instances, with fallback text
            incorporating available evidence summaries.
        """
        q = _llm_tailored_question(ctx)
        if q is not None:
            return [q][:max_questions]
        ev_summary = "; ".join(
            e.get("summary", "") for e in ctx.fact_check.evidence[:2] if isinstance(e, dict)
        )
        if ev_summary:
            fallback_text = (
                f"Evidence suggests: {ev_summary}. Rewrite your claim as one precise, verifiable sentence consistent with the strongest evidence."
            )
        else:
            fallback_text = (
                "External check is conflicting/insufficient. Rewrite your claim as one precise, verifiable sentence and cite a credible source if possible."
            )
        return [
            _make_q(
                fallback_text,
                qtype="open-ended",
                expects="rewrite_precise_claim",
            )
        ][:max_questions]


GENERATOR_BY_ISSUE = {
    IssueType.VISUAL_CONFLICT: VisualConflictQuestionGenerator(),
    IssueType.KNOWLEDGE_CONTRADICTION: KnowledgeContradictionQuestionGenerator(),
    IssueType.AMBIGUITY: AmbiguityQuestionGenerator(),
    IssueType.EXTERNAL_FACTUAL_CONFLICT: ExternalFactualConflictQuestionGenerator(),
}
