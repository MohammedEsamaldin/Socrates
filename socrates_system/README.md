# Socrates System

A multimodal hallucination detection and fact-verification system that applies Socratic methodology to identify, categorize, and verify factual claims in text and images.

## Project Overview

The Socrates System intercepts and interrogates claims made by or about Multimodal Large Language Models (MLLMs). It extracts atomic factual assertions from text, classifies them by verification type, routes them to the appropriate checker (cross-modal visual grounding, external knowledge retrieval, or session self-consistency), and produces a structured verdict with evidence and Socratic clarification questions.

The system supports two primary entry points:

- **Pipeline CLI** (`python -m socrates_system.pipeline`): batch processing with configurable LLM providers, claim routing modes, and evaluation harness integration.
- **Flask Web API** (`app.py`): HTTP endpoints for session-based interactive verification with file upload support.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Flask 2.3 |
| NLP / entity recognition | spaCy (`en_core_web_trf`) |
| Semantic similarity | `sentence-transformers` (`all-mpnet-base-v2`) |
| Knowledge graph storage | SQLite + NetworkX |
| Vision (local) | LLaVA-HF (HuggingFace), LLaVA original |
| LLM providers | Ollama, OpenAI, Anthropic Claude |
| Cross-modal service | Remote AGLA API (Modal deployment) |
| External fact sources | Wikipedia REST, Wikidata, Google Fact Check Tools |
| Image handling | Pillow |
| JSON parsing | `demjson3` (with stdlib `json` fallback) |

## Architecture Summary

```
User Input (text + optional image)
         │
         ▼
   ClaimExtractor          ← LLM + spaCy + sentence-transformers
         │
         ▼
   ClaimCategorizer        ← LLM classifies into 6 MLLM hallucination categories
         │
         ▼
   CheckRouter / DeterministicRouter
         │
    ┌────┴────────────────┐
    ▼                     ▼                     ▼
CrossAlignmentChecker  ExternalFactualityChecker  SelfContradictionChecker
  (AGLA remote API)    (Wikipedia/Wikidata/GFC)    (session KnowledgeGraph)
    └────┬────────────────┘
         │
         ▼
   ConflictResolver        ← weighted aggregation of checker verdicts
         │
         ▼
   ClarificationResolution ← LLM-driven Socratic dialogue for FAIL/UNCERTAIN
         │
         ▼
   KnowledgeGraphManager   ← persist verified claims for future sessions
         │
         ▼
   Structured response with socratic_dialogue, evidence, contradictions
```

The `HallucinationMitM` middleware wraps any main model to apply the same pipeline as a pre/post-processing step, enabling evaluation harness integration.

## Prerequisites

- Python 3.9–3.12
- One of: Ollama running locally (`http://localhost:11434`), OpenAI API key, or Anthropic API key
- spaCy model: `python -m spacy download en_core_web_trf` (or `en_core_web_sm` as fallback)
- For cross-modal verification: a deployed AGLA service URL, or a local LLaVA installation

## Installation

```bash
# Clone
git clone <repo-url>
cd Socrates/socrates_system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_trf
```

## Environment Variables and Configuration

All settings can be overridden via environment variables. See `COMMANDS_ENV_REFERENCE.txt` for the complete list.

| Variable | Default | Description |
|----------|---------|-------------|
| `SOC_LLM_PROVIDER` | `ollama` | LLM backend: `ollama`, `openai`, `claude`, `llava_hf`, `llava_original` |
| `SOC_LLM_MODEL` | provider-dependent | Model name (e.g., `llama3.1:8b`, `gpt-4o-mini`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | — | Required when `SOC_LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Required when `SOC_LLM_PROVIDER=claude` |
| `AGLA_API_URL` | `""` (disabled) | Remote AGLA service base URL |
| `AGLA_API_VERIFY_PATH` | `/verify` | Path on the AGLA service |
| `AGLA_API_TIMEOUT` | `120` | Request timeout in seconds |
| `GOOGLE_FACTCHECK_API_KEY` | — | Google Fact Check Tools API key |
| `TAVILY_API_KEY` | — | Tavily search fallback key |
| `FACTUALITY_ENABLED` | `true` | Toggle external factuality checking |
| `CLARIFICATION_ENABLED` | `true` | Toggle clarification resolution module |
| `QUESTION_GEN_ENABLED` | `true` | Toggle Socratic question generation |
| `SOC_ROUTER_MODE` | `llm` | Routing mode: `llm`, `deterministic`, `hybrid` |
| `SOC_CONFLICT_MODE` | `auto` | Conflict resolution: `auto` or `manual` |
| `SOC_SESSION_ID` | auto UUID | Fix session ID across runs |
| `FACTUALITY_CONTEXT_MODE` | `SOCRATIC_QUESTIONS` | Context fed to factuality LLM: `SOCRATIC_QUESTIONS`, `EXTRACTED_CLAIMS`, `NONE` |

Static configuration is in `config.py`. Key threshold constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum claim confidence |
| `CONTRADICTION_THRESHOLD` | 0.3 | Contradiction detection cutoff |
| `SIMILARITY_THRESHOLD` | 0.4 | Semantic matching cutoff |
| `CATEGORIZATION_CONFIDENCE_THRESHOLD` | 0.75 | Claim category acceptance |

## Quickstart

### CLI (pipeline mode)

```bash
# Verify text with Ollama (default)
python -m socrates_system.pipeline --text "The Eiffel Tower is located in Berlin."

# Verify text + image with OpenAI
python -m socrates_system.pipeline \
  --text "The image shows a red car parked in front of the building." \
  --image /path/to/image.jpg \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --show-kg

# Disable external factuality for faster category-only runs
python -m socrates_system.pipeline \
  --text "Water boils at 100 degrees Celsius." \
  --disable-factuality

# Full CLI help
python -m socrates_system.pipeline --help
```

### Flask Web API

```bash
python socrates_system/app.py
# Listens on http://0.0.0.0:5000
```

```bash
# Start session
curl -X POST http://localhost:5000/start_session

# Verify a text claim
curl -X POST http://localhost:5000/verify_claim \
  -F "user_input=The Eiffel Tower is in Berlin." \
  -F "session_id=session_20250425_120000"

# Health check
curl http://localhost:5000/api/health
```

## Project Directory Structure

```
socrates_system/
├── __init__.py
├── app.py                          # Flask web application
├── app_simple.py                   # Minimal Flask variant
├── config.py                       # Central configuration & env var loading
├── pipeline.py                     # SocratesPipeline (CLI entry point)
├── enhanced_socrates_app.py        # Extended Flask app (alternate)
├── COMMANDS_ENV_REFERENCE.txt      # Quick reference for CLI and env vars
├── requirements.txt
├── requirements_simple.txt
│
├── core/
│   ├── socrates_agent.py           # SocratesAgent (Flask orchestrator)
│   └── advanced_socrates_agent.py  # Extended agent variant
│
├── modules/
│   ├── shared_structures.py        # Data models (ExtractedClaim, VerificationRoute, ...)
│   ├── llm_manager.py              # Multi-provider LLM abstraction
│   ├── claim_extractor.py          # Stage 1: claim extraction
│   ├── claim_categorizer.py        # Stage 2: claim classification
│   ├── check_router.py             # LLM-based claim routing
│   ├── deterministic_router.py     # Heuristic-based claim routing
│   ├── question_generator.py       # Socratic question generation
│   ├── external_factuality_checker.py  # Wikipedia / Wikidata / GFC / LLM verdict
│   ├── cross_alignment_checker.py  # Vision-text alignment (local)
│   ├── cross_alignment_checker_simple.py
│   ├── self_contradiction_checker.py  # KG-based session consistency
│   ├── knowledge_graph_manager.py  # SQLite + NetworkX KG
│   ├── conflict_resolver.py        # Evidence-weighted verdict aggregation
│   ├── ambiguity_checker.py        # Ambiguity detection
│   ├── clarification_handler.py    # Clarification generation
│   ├── agla_client.py              # Remote AGLA API client
│   ├── fact_formatter.py           # GraphRAG-style fact linearization
│   └── prompt_templates/           # Prompt text files
│
├── clarification_resolution/
│   ├── core.py                     # ClarificationResolutionModule
│   ├── data_models.py              # ClarificationContext, SocraticQuestion, ...
│   ├── question_generators.py      # Per-issue question generators
│   ├── user_interface.py           # Interactive CLI helpers
│   └── config.py                   # Module-level configuration
│
├── middleware/
│   └── mitm_guard.py               # HallucinationMitM middleware
│
├── mllm_evaluation/
│   ├── base.py                     # BaseEvaluator (MitM harness)
│   ├── mitm.py                     # Pipeline bridge for evaluation
│   ├── eval_mme.py                 # MME benchmark evaluator
│   ├── eval_mmhal.py               # MMHal-Bench evaluator
│   ├── eval_amber.py               # AMBER benchmark evaluator
│   ├── eval_pope.py                # POPE benchmark evaluator
│   ├── eval_hallusion_bench.py     # HallusionBench evaluator
│   ├── eval_seed.py                # SEED benchmark evaluator
│   ├── providers/
│   │   └── llava_hf.py             # HuggingFace LLaVA provider
│   └── utils/                      # Checkpointing, logging, serialization
│
├── utils/
│   └── logger.py                   # Dual-handler logger (file + console)
│
├── data/
│   └── knowledge_graph.db          # SQLite KG (auto-created)
│
├── logs/                           # Per-module log files (auto-created)
└── tests/
    ├── test_claim_extractor.py
    ├── test_claim_categorizer.py
    ├── test_check_router.py
    ├── test_ambiguity_detector.py
    ├── test_question_generator.py
    ├── test_pipeline_qg.py
    └── test_shared_structures.py
```

## Testing

```bash
# Run all tests
python -m pytest socrates_system/tests/ -v

# Run a specific test file
python -m pytest socrates_system/tests/test_pipeline_qg.py -v

# Run with log capture disabled (show logger output)
python -m pytest socrates_system/tests/ -v -s
```

Tests use `unittest.mock` to stub out heavy LLM and model dependencies. No live API keys are required to run the test suite.

## Contributing Guidelines

1. All public functions and classes must have Google-style docstrings.
2. New modules must be registered in `socrates_system/__init__.py` if they are part of the public package API.
3. New verification sources should be implemented as a subclass of `ExternalAPIClient` (in `modules/external_factuality_checker.py`) and registered in `ExternalFactualityChecker.__init__`.
4. New LLM providers must be added to the `LLMProvider` enum and handled in `LLMManager._call_llm`.
5. All configuration that may vary across environments must be read from environment variables in `config.py`. Do not hard-code API keys or paths.
6. Do not commit `.env` files, API keys, or large model weights.
7. Keep `requirements.txt` minimal; place optional heavy dependencies (torch, transformers) behind lazy imports with graceful fallback.
