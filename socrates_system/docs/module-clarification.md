# Module: clarification_resolution

**Directory**: `socrates_system/clarification_resolution/`

## Purpose

The clarification resolution module handles cases where a claim cannot be verified directly because it is ambiguous, contradicted by visual evidence, contradicted by external facts, or contradicted by session knowledge. It generates issue-specific Socratic questions, optionally refines them with an LLM, collects responses (from a user or dev-mode stub), and proposes a corrected claim text.

---

## Data Models (data_models.py)

### `IssueType` (Enum)

```python
class IssueType(Enum):
    VISUAL_CONFLICT = "VISUAL_CONFLICT"
    KNOWLEDGE_CONTRADICTION = "KNOWLEDGE_CONTRADICTION"
    AMBIGUITY = "AMBIGUITY"
    EXTERNAL_FACTUAL_CONFLICT = "EXTERNAL_FACTUAL_CONFLICT"
```

Maps to the four scenarios that trigger clarification.

### `ResolutionAction` (Enum)

```python
class ResolutionAction(Enum):
    REVERIFY_PIPELINE = "REVERIFY_PIPELINE"
    DIRECT_TO_KG = "DIRECT_TO_KG"
    REJECT_CLAIM = "REJECT_CLAIM"
    NO_ACTION = "NO_ACTION"
```

Determines what the pipeline should do after clarification.

### `FactCheckResult` (dataclass)

```python
@dataclass
class FactCheckResult:
    verdict: str                        # PASS/FAIL/UNCERTAIN or TRUE/FALSE/INSUFFICIENT_EVIDENCE
    confidence: float = 0.0
    reasoning: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
```

Normalized fact-check result consumed by the module. The caller maps from the pipeline's checker output format to this structure.

### `ClarificationContext` (dataclass)

```python
@dataclass
class ClarificationContext:
    claim_text: str
    category: ClaimCategoryType
    fact_check: FactCheckResult
    failed_check_type: str              # e.g., "CROSS_MODAL", "EXTERNAL_SOURCE"
    issue_type: IssueType = IssueType.AMBIGUITY
    claim_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Input payload for `ClarificationResolutionModule.resolve_claim`.

### `SocraticQuestion` (dataclass)

```python
@dataclass
class SocraticQuestion:
    id: str
    text: str
    qtype: str = "open-ended"          # binary, selection, open-ended
    choices: Optional[List[str]] = None
    expects: Optional[str] = None      # hint for desired information
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `ClarificationResult` (dataclass)

```python
@dataclass
class ClarificationResult:
    original_claim: str
    corrected_claim: Optional[str]
    questions: List[SocraticQuestion]
    responses: Dict[str, Any]
    resolution_confidence: float
    next_action: ResolutionAction
    reasoning: str = ""
    issue_type: IssueType = IssueType.AMBIGUITY
    rerun_verification: bool = False
```

Output from `resolve_claim`. `corrected_claim` is `None` if no correction could be produced. `rerun_verification` signals the pipeline to re-run the full verification on the corrected claim.

---

## Core Module (core.py)

### `ClarificationResolutionModule`

```python
class ClarificationResolutionModule:
    def __init__(
        self,
        llm_manager=None,
        dev_mode: bool = False,
        route_policy: Optional[Dict[str, str]] = None,
    ) -> None
```

**Args**:
- `llm_manager`: `LLMManager` instance. Uses singleton `get_llm_manager()` if not provided.
- `dev_mode`: If True, skips interactive user prompts (uses placeholder responses). Controlled by `CLARIFICATION_DEV_MODE` env.
- `route_policy`: Dict mapping `IssueType.value` strings to `ResolutionAction.value` strings. Defaults to `clar_cfg.DEFAULT_NEXT_ACTION`.

---

#### `resolve_claim(ctx, responses=None, response_provider=None, max_questions=MAX_QUESTIONS_PER_SESSION)`

```python
def resolve_claim(
    self,
    ctx: ClarificationContext,
    responses: Optional[Dict[str, Any]] = None,
    response_provider: Optional[Callable[[SocraticQuestion], Any]] = None,
    max_questions: int = ...,
) -> ClarificationResult
```

Main entry point for clarification.

**Args**:
- `ctx`: Context including claim text, category, fact-check result, issue type.
- `responses`: Pre-populated responses (for testing or automated use).
- `response_provider`: Callable `(SocraticQuestion) -> Any` for interactive use.
- `max_questions`: Maximum number of questions to generate per session.

**Workflow**:
1. `_generate_questions(ctx, max_questions)` — call issue-specific generator from `GENERATOR_BY_ISSUE`.
2. `_collect_responses(questions, response_provider)` — gather user or programmatic responses.
3. `_process_responses_and_correct_claim(ctx, questions, responses)` — call LLM to produce corrected claim.
4. `_calculate_resolution_confidence(ctx, responses, corrected_claim)` — compute confidence.
5. `_determine_next_action(ctx, corrected_claim, resolution_conf)` — look up `route_policy`.
6. Return `ClarificationResult`.

---

## Question Generators (question_generators.py)

The module exports `GENERATOR_BY_ISSUE: Dict[IssueType, Callable]` mapping each issue type to a generator function.

Each generator accepts `(ctx: ClarificationContext, max_q: int) -> List[SocraticQuestion]`.

| `IssueType` | Generator focus |
|------------|-----------------|
| `VISUAL_CONFLICT` | Questions about what is visible in the image vs. claim |
| `KNOWLEDGE_CONTRADICTION` | Questions about the conflicting KG facts |
| `AMBIGUITY` | Questions to disambiguate vague terms or references |
| `EXTERNAL_FACTUAL_CONFLICT` | Questions about external source evidence |

---

## Configuration (config.py)

Key constants:

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_QUESTIONS_PER_SESSION` | 5 | Max questions per clarification call |
| `DEV_MODE_DEFAULT` | `False` | Use placeholder responses |
| `DEFAULT_NEXT_ACTION` | dict | Default routing per issue type |
| `REFINE_QUESTIONS_WITH_LLM` | `True` | LLM-refine generated questions |
| `CORRECT_CLAIM_WITH_LLM` | `False` | Use LLM for claim correction |
| `REQUIRE_USER_REWRITE` | `True` | Require human-provided correction |
| `SELECTIVE_TOKEN_REPLACEMENT` | `True` | Apply minimal-diff correction |
| `SELECTIVE_MAX_CHAR_DIFF_RATIO` | `0.4` | Max allowed character change ratio |
| `SELECTIVE_MAX_TOKEN_CHANGE_RATIO` | `0.5` | Max allowed token change ratio |

All constants can be overridden via environment variables of the same name.

---

## Usage Example

```python
from socrates_system.clarification_resolution import ClarificationResolutionModule
from socrates_system.clarification_resolution.data_models import (
    ClarificationContext, FactCheckResult, IssueType,
)
from socrates_system.modules.shared_structures import ClaimCategoryType

module = ClarificationResolutionModule(dev_mode=True)

ctx = ClarificationContext(
    claim_text="The tower in the image is blue.",
    category=ClaimCategoryType.VISUAL_GROUNDING_REQUIRED,
    fact_check=FactCheckResult(
        verdict="FAIL",
        confidence=0.85,
        reasoning="Image shows a red tower.",
    ),
    failed_check_type="CROSS_MODAL",
    issue_type=IssueType.VISUAL_CONFLICT,
)

result = module.resolve_claim(ctx)
print(result.corrected_claim)       # e.g., "The tower in the image is red."
print(result.resolution_confidence) # e.g., 0.72
print(result.next_action)           # e.g., ResolutionAction.DIRECT_TO_KG
```
