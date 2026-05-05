# Architecture Deepening Candidates

Each session: pick one open candidate, run `/improve-codebase-architecture`, jump straight to the grilling loop for that candidate.

---

## ✅ 1 — LLM singleton → real injectable seam (DONE)

**Files:** `modules/llm_manager.py` + 7 callers  
**What changed:** Deleted dead `AdvancedSocratesAgent` cluster (5 files, ~2700 lines). Verified `SocratesPipeline.__init__(llm_manager=...)` propagates to all active modules. Added `tests/test_pipeline_integration.py` — 5 tests proving the seam is structural: swapping the stub changes pipeline behaviour.

---

## ✅ 2 — JSON parsing has no home (DONE)

**Files:** `modules/claim_extractor.py`, `modules/claim_categorizer.py`, `clarification_resolution/core.py`, `clarification_resolution/question_generators.py`  
**What changed:** Created `utils/parse_llm_json.py` — single `parse_llm_json(text) -> Any` that raises `ValueError`. Handles fences, escape artefacts, balanced-bracket extraction, bare-word quoting. Removed `demjson3` dependency from `claim_extractor`. Deleted `_strip_code_fences` and `_parse_json_safely` from `clarification_resolution/core.py`. All 4 active callers now use one call. Added `tests/test_parse_llm_json.py` — 19 tests. `enhanced_socrates_app.py` sites left for candidate 5.

---

## ✅ 3 — Four agent variants, one decision tree duplicated (DONE)

**Files:** `core/socrates_agent.py`, `app_simple.py`
**What changed:** Created `core/verification_pipeline.py` — single `VerificationPipeline` with `run(claim_text, route, session_id, image_path, original_input) → VerificationResult` and `persist(...)`. Moved `CheckStatus` enum here. Both agents now instantiate the pipeline in `__init__` and delegate `_verify_claim_socratically` to it; agent bodies shed ~80 lines each. Fixed latent bug in `SocratesAgent` where checker results (string `"FAIL"`) were compared against `CheckStatus.FAIL` (enum), silently never matching. `EnhancedSocratesAgent` left separate — it talks to Ollama directly and shares no checker infrastructure. Added `tests/test_verification_pipeline.py` — 13 tests proving the decision tree is structural: route dispatch, short-circuit on FAIL, fallback sequence, persist delegation.

---

## ✅ 4 — Config has no interface — it's scattered `os.getenv()` calls (DONE)

**Files:** `config.py`, `pipeline.py`, `clarification_resolution/config.py`, `middleware/mitm_guard.py`, `modules/llm_manager.py`, `modules/external_factuality_checker.py`  
**What changed:** Added `AppConfig` frozen dataclass + `get_app_config()` (lru_cache singleton) to `config.py` — the only file that now calls `os.getenv()`. Removed all `os.getenv()` from the 5 target modules; pipeline/clarification/factuality read from AppConfig, `LLMManager` gains `llava_orig_use_cli / llava_conv_template / llava_timeout_sec` constructor args (eliminating the `os.environ[…]` write anti-pattern in main). `clarification_resolution/config.py` delegates module-level constants to AppConfig. Backward-compat module-level aliases preserved in `config.py`. Added `tests/test_app_config.py` — 27 tests covering defaults, env overrides, caching, and frozen enforcement. One intentional survivor: `SOC_COLOR_<ROLE>` dynamic-key pattern in `ConsoleColors.c()` (per-role runtime color, not a fixed field).

---

## 5 — Three Flask apps, one API contract (open)

**Files:** `app.py`, `app_simple.py`, `enhanced_socrates_app.py`

**Problem:** All three expose `/start_session`, `/verify_claim`, `/process`. Each reimplements file upload handling, session management, error response formatting, and CORS. A change to the API contract must be made three times — and the three apps have already diverged silently.

**Solution:** One `app.py` with a factory that accepts a configuration flag (simple/full/enhanced). HTTP layer lives once. Agent variant is injected. `app_simple.py` and `enhanced_socrates_app.py` deleted or moved to `examples/`.

**Benefits:** HTTP seam becomes real — request handling testable without instantiating verification logic. One fix to session ID generation or error formatting applies everywhere. Codebase loses ~1500 lines paying no architectural rent.

---

## 6 — Two routers, different answers to the same question (open)

**Files:** `modules/check_router.py`, `modules/deterministic_router.py`

**Problem:** Both implement `route_claim(claim) → VerificationRoute` but disagree: `CheckRouter` sends `EXTERNAL_KNOWLEDGE_REQUIRED` to CROSS_MODAL; `DeterministicRouter` scores differently. Flask uses one; CLI can use either. Same claim, different routes, depending on entry point.

**Solution:** Define a `ClaimRouter` interface with `route(claim) → VerificationRoute`. Consolidate the two implementations into one router with configurable strategy. Document the invariant each strategy preserves.

**Benefits:** Consistent routing across all entry points. Router interface becomes the test surface. No more implicit coupling to which app file is running.
