# Module: core

**Files**: `socrates_system/core/socrates_agent.py`, `socrates_system/core/advanced_socrates_agent.py`

## Purpose

`core/` contains `SocratesAgent`, the Flask-oriented orchestrator. It differs from `SocratesPipeline` in that it owns session state directly as instance attributes (rather than passing a session ID through method calls) and is designed for interactive, per-request use rather than batch CLI processing.

---

## Data Models

### `CheckStatus` (Enum)

```python
class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    SKIP = "SKIP"
```

### `SocraticInquiry` (dataclass)

```python
@dataclass
class SocraticInquiry:
    question: str
    reasoning: str
    expected_answer_type: str
    confidence: float
    context: Dict[str, Any]
```

Represents a single Socratic question generated for a claim during verification.

### `ClaimVerificationResult` (dataclass)

```python
@dataclass
class ClaimVerificationResult:
    claim: str
    status: CheckStatus
    confidence: float
    evidence: List[str]
    contradictions: List[str]
    socratic_questions: List[SocraticInquiry]
    clarification_needed: Optional[str]
    timestamp: datetime
```

Result object produced by `_verify_claim_socratically` for each extracted claim.

---

## Public Classes

### `SocratesAgent`

```python
class SocratesAgent:
    def __init__(self) -> None
```

Initializes all verification modules and the remote AGLA client (if `AGLA_API_URL` is set). Holds shared session state:

| Attribute | Type | Description |
|-----------|------|-------------|
| `claim_extractor` | `ClaimExtractor` | Claim extraction |
| `llm_manager` | `LLMManager` | Shared LLM (singleton via `get_llm_manager`) |
| `claim_categorizer` | `ClaimCategorizer` | Category classification |
| `check_router` | `CheckRouter` | LLM-based routing |
| `socratic_generator` | `SocraticQuestionGenerator` | Question generation |
| `cross_alignment_checker` | `CrossAlignmentChecker` | Visual alignment |
| `external_factuality_checker` | `ExternalFactualityChecker` | External fact check |
| `self_contradiction_checker` | `SelfContradictionChecker` | Session KG consistency |
| `ambiguity_checker` | `AmbiguityChecker` | Ambiguity detection |
| `clarification_handler` | `ClarificationHandler` | Clarification prompts |
| `kg_manager` | `KnowledgeGraphManager` | Knowledge graph |
| `agla_client` | `AGLAClient \| None` | Remote AGLA |
| `session_id` | `str \| None` | Current session ID |
| `conversation_history` | `list` | Per-session inputs |
| `verified_claims` | `list` | Passed claims this session |

---

### `SocratesAgent.start_session(session_id=None)`

```python
def start_session(self, session_id: str = None) -> str
```

Resets session state and initializes a fresh KG session.

**Args**:
- `session_id`: Optional custom ID. Defaults to `session_YYYYMMDD_HHMMSS`.

**Returns**: The session ID string.

---

### `SocratesAgent.process_user_input(user_input, image_path=None)`

```python
def process_user_input(self, user_input: str, image_path: Optional[str] = None) -> Dict[str, Any]
```

Main processing pipeline for a single user turn.

**Args**:
- `user_input`: Text to verify.
- `image_path`: Optional image path for multimodal analysis.

**Returns**: Compiled verification response dict (see `_compile_socratic_response`).

**Pipeline stages**:
1. Store input in `conversation_history`.
2. `claim_extractor.extract_claims(user_input)`.
3. `claim_categorizer.categorize_claim` + `check_router.route_claim` for each claim.
4. `_verify_claim_socratically` for each categorized claim.
5. `_update_knowledge_base` — persist PASS claims to KG.
6. `_compile_socratic_response` — build structured response.

---

### `SocratesAgent.get_session_summary()`

```python
def get_session_summary(self) -> Dict[str, Any]
```

**Returns**:
```python
{
    "session_id": str,
    "total_inputs": int,
    "verified_claims": int,
    "knowledge_graph_size": int,
}
```

---

## Private Methods

### `_verify_claim_socratically(claim, original_input, image_path)`

```python
def _verify_claim_socratically(
    self,
    claim: ExtractedClaim,
    original_input: str,
    image_path: Optional[str],
) -> ClaimVerificationResult
```

Four-stage verification for a single claim:

1. If category is `SUBJECTIVE_OPINION` or `PROCEDURAL_DESCRIPTIVE`: return `SKIP`.
2. Generate Socratic questions via `socratic_generator.handle_multi_category_claims`.
3. Execute verification based on `claim.verification_route.method`:
   - `CROSS_MODAL`: try remote AGLA, fall back to `cross_alignment_checker`.
   - `EXTERNAL_SOURCE`: optionally run cross-alignment first, then `external_factuality_checker`.
   - `KNOWLEDGE_GRAPH`: `self_contradiction_checker.check_contradiction`.
   - `EXPERT_VERIFICATION`: `ambiguity_checker.check_ambiguity`.
   - No route: full fallback sequence (cross-alignment → external → self-contradiction).
4. Final ambiguity pass for non-ambiguous routes.

---

### `_map_socratic_questions(generated)`

```python
def _map_socratic_questions(
    self,
    generated: Dict[str, List[Any]],
) -> List[SocraticInquiry]
```

Converts the dict returned by `SocraticQuestionGenerator.handle_multi_category_claims` (keyed by category name) into a flat list of `SocraticInquiry` objects.

---

### `_fallback_clarification_inquiry(claim_text, context)`

```python
def _fallback_clarification_inquiry(
    self,
    claim_text: str,
    context: Dict[str, Any],
) -> SocraticInquiry
```

Creates a generic clarification `SocraticInquiry` when the AGLA or cross-alignment check fails.

---

### `_update_knowledge_base(verification_results)`

```python
def _update_knowledge_base(self, verification_results: List[ClaimVerificationResult]) -> None
```

For each `PASS` result, calls `kg_manager.add_claim` and appends to `self.verified_claims`.

---

### `_compile_socratic_response(verification_results, original_input)`

```python
def _compile_socratic_response(
    self,
    verification_results: List[ClaimVerificationResult],
    original_input: str,
) -> Dict[str, Any]
```

Assembles the final response dict:

```python
{
    "session_id": str,
    "timestamp": str,             # ISO format
    "original_input": str,
    "verification_summary": {
        "total_claims": int,
        "verified_claims": int,
        "failed_claims": int,
        "overall_status": "PASS" | "FAIL",
    },
    "socratic_dialogue": [...],   # list of {type, content, ...}
    "detailed_results": [...],    # per-claim detail
    "next_steps": [str],
}
```

---

### `_generate_socratic_dialogue(verification_results)`

```python
def _generate_socratic_dialogue(
    self,
    verification_results: List[ClaimVerificationResult],
) -> List[Dict[str, str]]
```

Iterates results, emitting `socratic_question`, `verification_result`, or `contradiction_found` items into the dialogue list.

---

### `_generate_next_steps(verification_results)`

```python
def _generate_next_steps(self, verification_results: List[ClaimVerificationResult]) -> List[str]
```

Returns a list of human-readable next-step suggestions based on which claims failed.
