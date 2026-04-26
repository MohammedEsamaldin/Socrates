# Module: pipeline

**File**: `socrates_system/pipeline.py`

**Entry point**: `python -m socrates_system.pipeline`

## Purpose

`SocratesPipeline` is the CLI orchestrator for the full claim processing workflow. It wires together every major module — extraction, categorization, routing, Socratic question generation, external factuality checking, self-contradiction checking, conflict resolution, and clarification — into a single `run()` call.

All heavy imports are guarded with try/except and deferred to runtime, allowing the pipeline to partially initialize even when optional dependencies (spaCy, LLaVA, etc.) are not installed.

---

## Public Classes

### `SocratesPipeline`

```python
class SocratesPipeline:
    def __init__(
        self,
        llm_manager=None,
        factuality_enabled: bool = None,
        clarification_enabled: bool = None,
        clarification_dev_mode: bool = None,
        question_gen_enabled: bool = None,
        questions_per_category: int = None,
        qg_min_threshold: float = None,
        qg_max_complexity: float = None,
        qg_enable_fallback: bool = None,
        qg_prioritize_visual: bool = None,
        conflict_resolution_mode: str = None,
        factuality_context_mode: str = None,
        factuality_context_max_items: int = None,
        router_mode: str = None,
        post_factuality_clarification_enabled: bool = None,
    )
```

All constructor parameters default to `None`, falling back to environment variables (e.g., `FACTUALITY_ENABLED`, `SOC_ROUTER_MODE`). CLI flags override env vars, which override hardcoded defaults.

**Key attributes initialized**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `claim_extractor` | `ClaimExtractor` | Stage 1 |
| `claim_categorizer` | `ClaimCategorizer` | Stage 2 |
| `check_router` | `CheckRouter` | LLM router |
| `det_router` | `DeterministicRouter` | Heuristic router |
| `external_checker` | `ExternalFactualityChecker` | Wikipedia/Wikidata/GFC |
| `self_checker` | `SelfContradictionChecker` | KG consistency |
| `kg_manager` | `KnowledgeGraphManager` | SQLite + NetworkX |
| `conflict_resolver` | `ConflictResolver` | Verdict aggregation |
| `clarifier` | `ClarificationResolutionModule` | Dialogue clarification |
| `question_generator` | `SocraticQuestionGenerator` | Socratic questions |
| `agla_client` | `AGLAClient \| None` | Remote cross-modal |
| `router_mode` | `str` | `"llm"`, `"deterministic"`, or `"hybrid"` |
| `conflict_resolution_mode` | `str` | `"auto"` or `"manual"` |
| `session_id` | `str` | UUID (or `SOC_SESSION_ID`) |

---

### `SocratesPipeline.run(text, image_path=None)`

```python
def run(self, text: str, image_path: Optional[str] = None) -> List[ExtractedClaim]
```

Main pipeline entry point.

**Args**:
- `text`: Input text to process.
- `image_path`: Optional image path for cross-modal verification.

**Returns**: List of `ExtractedClaim` objects with verification results attached (categories, route, factuality_status, socratic_questions, etc.).

**Pipeline stages**:

1. `claim_extractor.extract_claims(text)` — extract atomic claims.
2. `claim_categorizer.categorize_claim(claim)` — classify each claim.
3. Pre-routing clarification — if `AMBIGUOUS_RESOLUTION_REQUIRED`, optionally call `clarifier.resolve_claim` and re-categorize.
4. `question_generator` — generate Socratic questions per category.
5. `_route_claim(claim)` — select verification method.
6. Verification per route:
   - `CROSS_MODAL`: `agla_client.verify` → fallback to `cross_checker.check_alignment`.
   - `EXTERNAL_SOURCE`: `external_checker.verify_claim` with optional Socratic-questions context.
   - `KNOWLEDGE_GRAPH`: `self_checker.check_contradiction`.
   - `EXPERT_VERIFICATION`: skipped (ambiguous claims).
   - `UNVERIFIABLE`: skipped (subjective/procedural).
7. `conflict_resolver.resolve` or `_manual_resolve_conflict` — aggregate external + self results.
8. Post-factuality clarification — if enabled and verdict is FAIL/UNCERTAIN.
9. KG update — add PASS claims to `kg_manager`.
10. Console output with ANSI colors.

---

### `SocratesPipeline._route_claim(claim)`

```python
def _route_claim(self, claim: ExtractedClaim) -> VerificationRoute
```

Selects the routing strategy based on `self.router_mode`:

- `"llm"`: calls `check_router.route_claim`; falls back to `det_router` on failure.
- `"deterministic"`: calls `det_router.route_claim`; falls back to `check_router`.
- `"hybrid"`: runs both and selects by a scoring function that considers confidence, KG coverage, `route_hint`, `vision_flag`, and category alignment.

Returns a `VerificationRoute` with a default fallback (`EXTERNAL_SOURCE`, confidence 0.5) if both routers fail.

---

### `SocratesPipeline._make_module_llm(provider_env, model_env, fallback_llm)`

```python
def _make_module_llm(
    self,
    provider_env: Optional[str],
    model_env: Optional[str],
    fallback_llm: Optional[LLMManager],
) -> Optional[LLMManager]
```

Creates a dedicated `LLMManager` for a module if env overrides are provided (`FACTUALITY_LLM_PROVIDER` / `FACTUALITY_LLM_MODEL`, `SELF_LLM_PROVIDER` / `SELF_LLM_MODEL`). Returns the shared `fallback_llm` otherwise.

---

### `SocratesPipeline._manual_resolve_conflict(claim, external_result, self_result)`

```python
def _manual_resolve_conflict(
    self,
    claim: str,
    external_result: Optional[Dict[str, Any]],
    self_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]
```

Interactive TTY conflict resolution. Prints external and self-consistency results, prompts for `status`, `confidence`, `should_add_to_kg`, and an optional reasoning note. Falls back to auto if `stdin` is not a TTY or an exception occurs.

---

## Supporting Classes

### `ConsoleColors`

ANSI color helper with role-based defaults overridable via `SOC_COLOR_<ROLE>` env vars. Disabled automatically when `NO_COLOR` is set or stdout is not a TTY.

```python
ConsoleColors.c(role: str, text: str) -> str
```

Roles: `heading`, `claim`, `label`, `value`, `entity`, `category`, `question`, `route`, `clarification`, `factuality_pass`, `factuality_fail`, `factuality_uncertain`, `summary`.

---

## CLI Arguments

Run `python -m socrates_system.pipeline --help` for the full list. Key flags:

| Flag | Env override | Description |
|------|-------------|-------------|
| `--text TEXT` | — | Input text |
| `--image PATH` | — | Image path |
| `--llm-provider PROV` | `SOC_LLM_PROVIDER` | LLM provider |
| `--llm-model MODEL` | `SOC_LLM_MODEL` | Model name |
| `--enable/disable-factuality` | `FACTUALITY_ENABLED` | External factuality |
| `--enable/disable-clarification` | `CLARIFICATION_ENABLED` | Clarification module |
| `--enable/disable-question-gen` | `QUESTION_GEN_ENABLED` | Socratic QG |
| `--questions-per-category N` | `QG_QUESTIONS_PER_CATEGORY` | Questions per category |
| `--router-mode MODE` | `SOC_ROUTER_MODE` | `llm` / `deterministic` / `hybrid` |
| `--conflict-mode MODE` | `SOC_CONFLICT_MODE` | `auto` / `manual` |
| `--factuality-context MODE` | `FACTUALITY_CONTEXT_MODE` | `socratic` / `claims` / `none` |
| `--show-kg` | `SOC_SHOW_KG` | Print KG summary |
| `--kg-max-items N` | `SOC_KG_MAX_ITEMS` | Max KG display items |

---

## Usage Example

```python
from socrates_system.pipeline import SocratesPipeline

pipeline = SocratesPipeline(
    factuality_enabled=True,
    question_gen_enabled=True,
    router_mode="hybrid",
)

claims = pipeline.run(
    text="The Eiffel Tower is located in Rome.",
    image_path="/path/to/photo.jpg",
)

for claim in claims:
    print(claim.text, claim.factuality_status, claim.factuality_confidence)
```
