# Module: middleware

**File**: `socrates_system/middleware/mitm_guard.py`

## Purpose

The `HallucinationMitM` class wraps any generative model with the Socrates verification pipeline as a transparent Man-in-the-Middle layer. It is used primarily by the `mllm_evaluation` harness to apply hallucination detection and correction to both the user prompt (input) and the model response (output) during benchmark evaluation.

---

## Public Classes

### `MainModelAdapter` (Protocol)

```python
class MainModelAdapter(Protocol):
    def generate(self, text: str, image_path: Optional[str] = None, **kwargs) -> str: ...
```

Interface that any main model adapter must satisfy. Implementations receive corrected text and an optional image path.

---

### `LLMMainModelAdapter`

```python
class LLMMainModelAdapter:
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        anthropic_base_url: Optional[str] = None,
    ) -> None
```

Convenience adapter backed by `LLMManager`. Calls `llm.generate_text(prompt, max_tokens=512)`. Image path is ignored (text-only).

#### `generate(text, image_path=None, **kwargs)`

```python
def generate(self, text: str, image_path: Optional[str] = None, **kwargs) -> str
```

---

### `Correction` (dataclass)

```python
@dataclass
class Correction:
    start: int
    end: int
    original: str
    replacement: str
    reason: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
```

Represents a span-level token replacement within the source text.

---

### `MitMRunResult` (dataclass)

```python
@dataclass
class MitMRunResult:
    corrected_input: str
    input_corrections: List[Correction]
    raw_output: str
    corrected_output: str
    output_corrections: List[Correction]
    session_id: str
```

Return value of `HallucinationMitM.run()`.

---

### `HallucinationMitM`

```python
class HallucinationMitM:
    def __init__(
        self,
        main_model: Optional[MainModelAdapter] = None,
        llm_manager: Optional[LLMManager] = None,
        session_id: Optional[str] = None,
        enable_external: bool = True,
        enable_cross_modal: bool = True,
        enable_self_contradiction: bool = True,
        clarification_only: bool = False,
    ) -> None
```

**Args**:
- `main_model`: The model to wrap. Defaults to `LLMMainModelAdapter()`.
- `llm_manager`: LLM for verification/correction prompts.
- `session_id`: Fixed session ID; uses `SOC_SESSION_ID` env or auto-generates UUID.
- `enable_external`: Toggle external factuality checking.
- `enable_cross_modal`: Toggle AGLA/cross-alignment checking.
- `enable_self_contradiction`: Toggle KG self-contradiction checking. Automatically disabled if `KnowledgeGraphManager` is unavailable.
- `clarification_only`: If True, only apply corrections when ambiguity is detected (not on FAIL/UNCERTAIN).

---

#### `run(text=None, image_path=None, **gen_kwargs)`

```python
def run(
    self,
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    **gen_kwargs,
) -> MitMRunResult
```

Full MitM pipeline:

1. `_process_text(text, image_path)` → `corrected_input`, `input_corrections`
2. `main_model.generate(corrected_input, image_path)` → `raw_output`
3. `_process_text(raw_output, image_path)` → `corrected_output`, `output_corrections`
4. Return `MitMRunResult`.

---

## Private Methods

### `_process_text(text, image_path)`

```python
def _process_text(self, text: str, image_path: Optional[str]) -> Tuple[str, List[Correction]]
```

Core logic:
1. Extract claims from text.
2. Categorize and route each claim.
3. Verify each claim via `_verify_claim_route`.
4. Update KG on PASS.
5. Generate corrections for FAIL/UNCERTAIN claims via `_propose_correction`.
6. Apply minimal token-level edits via `_apply_corrections_minimal`.

Diagnostics are stored on `self.last_claim_texts`, `self.last_routes`, `self.last_verdicts`.

---

### `_verify_claim_route(claim, route, image_path=None)`

```python
def _verify_claim_route(
    self,
    claim: ExtractedClaim,
    route: Any,
    image_path: Optional[str],
) -> Dict[str, Any]
```

Dispatches to the checker matching `route.method`:
- `CROSS_MODAL`: `_cross_modal_verify`.
- `EXTERNAL_SOURCE`: `external_checker.verify_claim`.
- `KNOWLEDGE_GRAPH`: `self_checker.check_contradiction`.

Returns a standardized `{status, confidence, evidence, sources, reasoning}` dict.

---

### `_cross_modal_verify(claim_text, image_path)`

```python
def _cross_modal_verify(self, claim_text: str, image_path: str) -> Dict[str, Any]
```

Tries remote AGLA first; falls back to local `CrossAlignmentChecker` on failure.

---

### `_propose_correction(claim, verdict, image_path=None)`

```python
def _propose_correction(
    self,
    claim: ExtractedClaim,
    verdict: Dict[str, Any],
    image_path: Optional[str],
) -> Optional[Correction]
```

Calls `LLMManager.generate_text` with a minimal-edit system prompt. Builds the replacement using `_minimal_token_rewrite`. Returns `None` if the LLM fails or produces no useful correction.

---

### `_minimal_token_rewrite(original, corrected)`

```python
def _minimal_token_rewrite(self, original: str, corrected: str) -> str
```

Uses `difflib.SequenceMatcher` on token lists to produce a minimal-diff replacement that preserves unchanged tokens.

---

### `_apply_corrections_minimal(text, claims, corrections)`

```python
def _apply_corrections_minimal(
    self,
    text: str,
    claims: List[ExtractedClaim],
    corrections: List[Correction],
) -> str
```

Applies corrections sorted by `start` position, adjusting for offset shifts as each replacement is applied.

---

## Usage Example

```python
from socrates_system.middleware.mitm_guard import HallucinationMitM, LLMMainModelAdapter

mitm = HallucinationMitM(
    main_model=LLMMainModelAdapter(provider="ollama", model_name="llama3.1:8b"),
    enable_external=True,
    enable_cross_modal=True,
    enable_self_contradiction=False,  # disable if spaCy not installed
)

result = mitm.run(
    text="The Eiffel Tower is located in Berlin.",
    image_path=None,
)

print(result.corrected_output)
# e.g. "The Eiffel Tower is located in Paris."

for corr in result.output_corrections:
    print(f"[{corr.start}:{corr.end}] {corr.original!r} -> {corr.replacement!r}")
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOC_USE_MITM` | — | Master toggle used by eval harness |
| `SOC_MITM_VERIFY_INPUT` | — | Apply pre-routing corrections |
| `SOC_MITM_VERIFY_OUTPUT` | — | Apply post-factuality corrections |
| `SOC_MITM_MIN_CONF` | `0.55` | Minimum resolution confidence to apply correction |
| `SOC_ALLOW_POLARITY_FLIP` | `false` | Allow corrections that invert negation |
