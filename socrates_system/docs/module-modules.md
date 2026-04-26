# Module: modules

**Directory**: `socrates_system/modules/`

This directory contains all the individual verification and utility modules. This document covers every public class and function.

---

## shared_structures.py

Central data model file. All pipeline modules import from here to avoid circular imports.

### `ExtractedEntity` (dataclass)

```python
@dataclass
class ExtractedEntity:
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    canonical_id: Optional[str] = None
```

Named entity extracted from a claim. `canonical_id` is populated by `KnowledgeGraphManager.resolve_canonical_id` when the claim extractor has a KG manager attached.

### `ExtractedRelationship` (dataclass)

```python
@dataclass
class ExtractedRelationship:
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
```

### `VerificationMethod` (Enum)

```
EXTERNAL_SOURCE, KNOWLEDGE_GRAPH, CROSS_MODAL,
EXPERT_VERIFICATION, CALCULATION, DEFINITIONAL, UNVERIFIABLE
```

### `VerificationRoute` (dataclass)

```python
@dataclass
class VerificationRoute:
    method: VerificationMethod
    confidence: float
    justification: str
    estimated_cost: float
    estimated_latency: float
    secondary_actions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### `ClaimCategoryType` (Enum)

Six categories for MLLM hallucination detection:

| Name | Verification target |
|------|-------------------|
| `VISUAL_GROUNDING_REQUIRED` | Cross-modal / image |
| `EXTERNAL_KNOWLEDGE_REQUIRED` | External APIs |
| `SELF_CONSISTENCY_REQUIRED` | Session knowledge graph |
| `AMBIGUOUS_RESOLUTION_REQUIRED` | Clarification dialogue |
| `SUBJECTIVE_OPINION` | Skipped |
| `PROCEDURAL_DESCRIPTIVE` | Skipped |

Each enum value is a string description used in prompts.

### `ClaimCategory` (dataclass)

```python
@dataclass
class ClaimCategory:
    name: ClaimCategoryType
    confidence: float
    justification: str
```

### `ExtractedClaim` (dataclass)

```python
@dataclass
class ExtractedClaim:
    text: str
    start_char: int
    end_char: int
    confidence: float
    source_text: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    categories: List[ClaimCategory] = field(default_factory=list)
    verification_route: Optional[VerificationRoute] = None
    context_window: Optional[str] = None
    ambiguity_reason: Optional[str] = None
    route_hint: Optional[str] = None
    vision_flag: Optional[bool] = None
    socratic_questions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    factuality_status: Optional[str] = None   # PASS | FAIL | UNCERTAIN | ERROR
    factuality_confidence: float = 0.0
    factuality_verdict: Optional[bool] = None
    factuality_evidence: List[str] = field(default_factory=list)
    factuality_sources: List[str] = field(default_factory=list)
    factuality_reasoning: Optional[str] = None
```

The central data object flowing through the entire pipeline. After `run()` completes, all verification results are stored on this object.

---

## llm_manager.py

### `LLMProvider` (Enum)

```
OLLAMA, OPENAI, CLAUDE, LLAVA_HF, LLAVA_ORIGINAL
```

### `LLMTaskType` (Enum)

```
CLAIM_EXTRACTION, SOCRATIC_QUESTIONING, REASONING_GENERATION,
FACTUAL_VERIFICATION, RELATIONSHIP_EXTRACTION, KNOWLEDGE_INTEGRATION,
FAITHFULNESS_ASSESSMENT, CONTRADICTION_DETECTION, CONTRADICTION_DETECTION_SIMPLE
```

Each task type has a corresponding system prompt and template stored in `self.prompt_templates`.

### `LLMRequest` (dataclass)

```python
@dataclass
class LLMRequest:
    task_type: LLMTaskType
    prompt: str
    context: Dict[str, Any]
    images: Optional[List[str]] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
```

### `LLMResponse` (dataclass)

```python
@dataclass
class LLMResponse:
    content: str
    task_type: LLMTaskType
    confidence: float
    reasoning: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error: Optional[str] = None
```

### `LLMManager`

```python
class LLMManager:
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Union[LLMProvider, str, None] = None,
        base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        anthropic_base_url: Optional[str] = None,
        max_concurrent: int = 3,
    )
```

Unified LLM abstraction. Provider resolution order: constructor argument → `SOC_LLM_PROVIDER` env → `"ollama"` default. Model resolution: constructor argument → `SOC_LLM_MODEL` env → provider default.

**Key public methods**:

#### `generate_text(prompt, max_tokens=4096, temperature=0.2, system_prompt=None, images=None)`

```python
def generate_text(
    self,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    system_prompt: str = None,
    images: Optional[List[str]] = None,
) -> str
```

Synchronous wrapper around `_call_llm`. `images` accepts local file paths, HTTP URLs, or `data:image/...;base64,...` strings.

#### `process_request(request)` (async)

```python
async def process_request(self, request: LLMRequest) -> LLMResponse
```

Dispatches to the configured provider, parses the response JSON, and returns an `LLMResponse`.

#### `batch_process(requests)` (async)

```python
async def batch_process(self, requests: List[LLMRequest]) -> List[LLMResponse]
```

Concurrent processing with a semaphore limiting to `max_concurrent` simultaneous requests.

#### Task-specific convenience methods (all async)

- `extract_claims(text, context, images)`
- `generate_socratic_questions(claim, context, images)`
- `generate_reasoning(question, evidence, context, images)`
- `verify_claim(claim, evidence, context, images)`
- `extract_relationships(text, entities, context, images)`
- `integrate_knowledge(new_info, existing_knowledge, context, images)`
- `assess_faithfulness(original_claim, corrected_claim, evidence, images)`
- `detect_contradictions(claim, existing_claims, context, entities, entity_knowledge, images)`
- `detect_contradiction_simple(claim, session_facts, images)`

Synchronous wrappers: `detect_contradictions_sync(...)`, `detect_contradiction_simple_sync(...)`.

#### `shutdown()`

```python
def shutdown(self) -> None
```

Shuts down the `ThreadPoolExecutor`.

### Module-level helpers

```python
def get_llm_manager() -> LLMManager
def shutdown_llm_manager() -> None
```

Singleton pattern for sharing a single `LLMManager` instance across modules.

---

## claim_extractor.py

### `ClaimExtractor`

```python
class ClaimExtractor:
    def __init__(self, llm_manager: Optional[Any] = None)
```

**Key public methods**:

#### `extract_claims(text)`

```python
def extract_claims(self, text: str) -> List[ExtractedClaim]
```

Primary entry point. Tries LLM-based extraction first; falls back to regex patterns. Applies semantic similarity matching to map LLM claims back to source sentence offsets.

#### `set_kg_manager(kg_manager, session_id=None)`

```python
def set_kg_manager(self, kg_manager: Any, session_id: Optional[str] = None) -> None
```

Attaches a `KnowledgeGraphManager` for canonical entity ID resolution in `_post_process_claims`.

#### `extract_claims_batch(texts)`

```python
def extract_claims_batch(self, texts: List[str]) -> List[List[ExtractedClaim]]
```

Calls `extract_claims` for each text in sequence.

#### `get_claim_summary(claims)`

```python
def get_claim_summary(self, claims: List[ExtractedClaim]) -> Dict[str, Any]
```

Returns `{total_claims, total_entities, avg_entities_per_claim, avg_confidence}`.

**Internal design**:
- Prompt file: `modules/prompt_templates/claim_extraction.txt`
- JSON parsing: attempts `demjson3.decode` → `json.loads` → substring extraction + sanitization.
- LLM route hints (`route_hint`, `vision_flag`) are parsed from the JSON and stored on `ExtractedClaim`.
- When spaCy is unavailable, `_make_fallback_doc` produces a minimal sentence-split object.

---

## claim_categorizer.py

### `ClaimCategorizer`

```python
class ClaimCategorizer:
    def __init__(self, llm_manager: Optional[LLMManager])
```

**Key public methods**:

#### `categorize_claim(claim)`

```python
def categorize_claim(self, claim: ExtractedClaim) -> ExtractedClaim
```

Returns the same `ExtractedClaim` with `claim.categories` populated. Uses `claim_categorisation.txt` prompt. Falls back to `_categorize_with_rules` on LLM failure.

#### `get_category_descriptions()`

```python
def get_category_descriptions(self) -> Dict[str, str]
```

Returns `{category_name: description_string}` for all `ClaimCategoryType` values.

---

## check_router.py

### `CheckRouter`

```python
class CheckRouter:
    def __init__(self, available_methods: Optional[Set[VerificationMethod]] = None)
```

**Key mapping** (from `VERIFICATION_METHODS`):

| `VerificationMethod` | Categories handled | cost | latency |
|---------------------|-------------------|------|---------|
| `CROSS_MODAL` | `VISUAL_GROUNDING_REQUIRED` | 0.8 | 1.5 |
| `EXTERNAL_SOURCE` | `EXTERNAL_KNOWLEDGE_REQUIRED` | 0.7 | 2.0 |
| `KNOWLEDGE_GRAPH` | `SELF_CONSISTENCY_REQUIRED` | 0.2 | 0.5 |
| `EXPERT_VERIFICATION` | `AMBIGUOUS_RESOLUTION_REQUIRED` | 1.0 | 86400 |
| `UNVERIFIABLE` | `SUBJECTIVE_OPINION`, `PROCEDURAL_DESCRIPTIVE` | 0.0 | 0.0 |

Note: `EXTERNAL_KNOWLEDGE_REQUIRED` is currently redirected to `CROSS_MODAL` (for MME evaluation where all factual claims go to AGLA).

#### `route_claim(claim)`

```python
def route_claim(self, claim: ExtractedClaim) -> VerificationRoute
```

Direct category-to-method lookup with priority ordering.

#### `get_verification_methods()`

```python
def get_verification_methods(self) -> Dict[VerificationMethod, Dict[str, Any]]
```

Returns the subset of `VERIFICATION_METHODS` that are in `available_methods`.

---

## deterministic_router.py

### `DeterministicRouter`

```python
class DeterministicRouter:
    def __init__(
        self,
        available_methods: Optional[Set[VerificationMethod]] = None,
        *,
        kg_manager: Optional[Any] = None,
        session_id: Optional[str] = None,
    )
```

Heuristic router that checks:
1. `vision_flag` on the claim → `CROSS_MODAL`.
2. `route_hint` string prefix → matching method.
3. KG coverage ratio → `KNOWLEDGE_GRAPH` if entity is known.
4. Category-based fallback.

#### `route_claim(claim)`

```python
def route_claim(self, claim: ExtractedClaim) -> VerificationRoute
```

---

## external_factuality_checker.py

### `ExternalAPIClient` (base class)

```python
class ExternalAPIClient:
    def __init__(
        self,
        session: Optional[requests.Session] = None,
        max_retries: int = 2,
        backoff_sec: float = 0.5,
        timeout: float = 6.0,
    )
    def query(self, claim: str) -> List[Dict[str, Any]]
```

Base for all external source clients. Implements retry + exponential backoff. Subclasses implement `_build_requests(claim)` and `_interpret(claim, payload)`.

### `WikipediaClient(ExternalAPIClient)`

Uses Wikipedia Search API (`en.wikipedia.org/w/api.php?action=query&list=search`) to find relevant articles and extract snippets. Returns up to 2 results with `{source, status, confidence, content, evidence, sources}`.

### `WikidataClient(ExternalAPIClient)`

Uses `wikidata.org/w/api.php?action=wbsearchentities` to match entities. Returns up to 3 matches.

### `GoogleFactCheckClient(ExternalAPIClient)`

Uses `factchecktools.googleapis.com/v1alpha1/claims:search`. Requires `GOOGLE_FACTCHECK_API_KEY`.

### `ConceptNetClient(ExternalAPIClient)`

Uses `api.conceptnet.io/query` to find commonsense edges. Disabled by default in `ExternalFactualityChecker.__init__`.

### `FactCheckResult` (dataclass)

```python
@dataclass
class FactCheckResult:
    status: str          # PASS, FAIL, UNCERTAIN
    confidence: float
    external_facts: List[str]
    contradictions: List[str]
    evidence: List[str]
    sources: List[str]
    reasoning: str
```

### `ExternalFactualityChecker`

```python
class ExternalFactualityChecker:
    def __init__(
        self,
        enable_clients: Optional[bool] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        backoff_sec: Optional[float] = None,
        llm_manager: Optional[LLMManager] = None,
    )
```

**Key method**:

#### `verify_claim(claim, input_context=None)`

```python
def verify_claim(
    self,
    claim: str,
    input_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]
```

Queries all registered clients, then calls `_get_llm_factuality_verdict` to aggregate results. `input_context` is an optional `{type: "SOCRATIC_QUESTIONS"|"EXTRACTED_CLAIMS", items: [str]}` dict prepended to the evidence in the LLM prompt.

Returns: `{status, confidence, external_facts, contradictions, evidence, sources, reasoning}`.

---

## self_contradiction_checker.py

### `SelfContradictionChecker`

```python
class SelfContradictionChecker:
    def __init__(self, llm_manager: Optional[LLMManager] = None)
    def set_kg_manager(self, kg_manager: KnowledgeGraphManager) -> None
```

#### `check_contradiction(claim, session_id)`

```python
def check_contradiction(self, claim: str, session_id: str) -> Dict[str, Any]
```

Retrieves session claims from KG, linearizes them via `GraphRAGFactFormatter`, and calls `llm.detect_contradiction_simple_sync`. Returns `{status, confidence, contradictions, evidence, conflicting_claims}`.

---

## knowledge_graph_manager.py

### `Entity`, `Relation`, `Claim` (dataclasses)

Core graph objects stored in SQLite and used in the in-memory NetworkX graph.

### `StableId`

```python
class StableId:
    @staticmethod
    def entity(text: str, label: str) -> str
    @staticmethod
    def relation(subject_id: str, predicate: str, object_id: str) -> str
    @staticmethod
    def canonical(normalized_text: str, entity_type: str) -> str
```

Generates deterministic SHA-256-derived IDs for deduplication across sessions.

### `KnowledgeGraphManager`

```python
class KnowledgeGraphManager:
    def __init__(self, llm_manager: Any = None)
```

**Key public methods**:

#### `initialize_session(session_id)`

Creates a session-scoped record in the SQLite database and initializes an in-memory NetworkX graph.

#### `add_claim(claim, evidence, confidence, session_id)`

Extracts entities and relations from `claim` text (via spaCy), creates `Entity` and `Relation` objects with stable IDs, and writes them to both SQLite and the NetworkX graph.

#### `resolve_canonical_id(text, label, session_id=None)`

Looks up canonical entity ID for a given (text, label) pair.

#### `get_graph_size(session_id)`

Returns node count of the session graph.

#### `export_session_graph(session_id)`

Serializes the session graph to a dict for the `/knowledge_graph/<session_id>` API endpoint.

---

## conflict_resolver.py

### `ConflictResolverConfig` (dataclass)

```python
@dataclass
class ConflictResolverConfig:
    weight_external: float = 0.6
    weight_self: float = 0.4
    add_to_kg_threshold: float = 0.7
```

### `ConflictResolver`

```python
class ConflictResolver:
    def __init__(self, config: Optional[ConflictResolverConfig] = None)
```

#### `resolve(claim, external_result, self_result)`

```python
def resolve(
    self,
    claim: str,
    external_result: Optional[Dict[str, Any]],
    self_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]
```

Weighted aggregation of `external_result` and `self_result`. Maps status strings to scores (`PASS`→1.0, `FAIL`→-1.0, else 0.0) and computes a weighted average. Sets `should_add_to_kg=True` when final confidence ≥ `add_to_kg_threshold` and status is PASS.

Returns: `{status, confidence, reasoning, sources, contradictions, evidence, should_add_to_kg}`.

---

## agla_client.py

### `AGLAClient`

```python
class AGLAClient:
    def __init__(self, base_url: str, verify_path: str = "/verify", timeout: int = 120)
```

Thin HTTP client for the remote AGLA (Adaptive Grounded LLM Alignment) service.

#### `verify(image, claim, socratic_question=None, use_agla=None, alpha=None, beta=None, return_debug=False)`

```python
def verify(
    self,
    image: Union[str, bytes, bytearray, Image.Image],
    claim: str,
    socratic_question: Optional[str] = None,
    use_agla: Optional[bool] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    return_debug: bool = False,
) -> Dict[str, Any]
```

Sends `multipart/form-data` POST to `{base_url}{verify_path}`. Image can be a file path, raw bytes, a PIL Image, or an HTTP URL. Returns the JSON response from the AGLA service: `{verdict: "True"|"False"|"Uncertain", truth, latency_ms, debug?}`.

---

## ambiguity_checker.py

### `AmbiguityResult` (dataclass)

```python
@dataclass
class AmbiguityResult:
    needs_clarification: bool
    ambiguous_terms: List[str]
    clarification_questions: List[str]
    ambiguity_score: float
    reasoning: str
```

### `AmbiguityChecker`

```python
class AmbiguityChecker:
    def __init__(self)
    def check_ambiguity(self, claim: str, context: str = "") -> Dict[str, Any]
```

Rule-based detection of vague quantifiers, subjective terms, relative terms, temporal vagueness, pronouns, and modal uncertainty. Returns a dict with `needs_clarification`, `ambiguous_terms`, `clarification_questions`, `ambiguity_score`, `reasoning`.

---

## question_generator.py

### `SocraticQuestionGenerator`

Higher-level generator wrapping `LLMManager` and `VerificationCapabilities`. Generates questions per claim category. Used by both `SocratesAgent` and `SocratesPipeline`.

Key method: `handle_multi_category_claims(claim_text, categories, num_questions_per_category)` → `Dict[str, List[SocraticQuestion]]`.

### `SocraticConfig`

Configuration dataclass for the generator:
- `min_confidence_threshold`: minimum confidence for a question to be kept.
- `max_question_complexity_ratio`: controls question length relative to claim.
- `enable_fallback`: whether to use template-based fallback on LLM failure.
- `prioritize_visual_grounding`: if True, visual questions are generated first.

### `VerificationCapabilities`

```python
@dataclass
class VerificationCapabilities:
    visual_grounding: List[str]     # available visual methods
    external_knowledge: List[str]   # available external sources
    self_consistency: List[str]     # available KG methods
```

Passed to `SocraticQuestionGenerator` to constrain question generation to what the system can actually verify.

### `LLMInterfaceAdapter`

Wraps `LLMManager` to provide the `generate_socratic_questions_for_category(claim, category, capabilities, n)` interface expected by `SocraticQuestionGenerator`.

---

## fact_formatter.py

### `GraphRAGFactFormatter`

Linearizes session KG facts into a text string for use in LLM prompts. Used by `SelfContradictionChecker` to give the LLM a concise view of all established session knowledge.

---

## clarification_handler.py

### `ClarificationHandler`

Simple handler that generates a clarification request string for display when a cross-modal conflict is detected.

#### `generate_clarification(claim_text, visual_description=None)`

```python
def generate_clarification(
    self,
    claim_text: str,
    visual_description: Optional[str] = None,
) -> str
```

Returns a human-readable clarification prompt combining the claim text and the visual context.
