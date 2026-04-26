# Setup Guide

## Prerequisites

- **Python** 3.9–3.12 (3.12 recommended)
- **pip** 23+ (comes with Python 3.12)
- One of the following LLM backends:
  - **Ollama** running locally on `http://localhost:11434` (default)
  - An **OpenAI API key** (`SOC_LLM_PROVIDER=openai`)
  - An **Anthropic API key** (`SOC_LLM_PROVIDER=claude`)
- **spaCy** language model (see below)
- *(Optional)* A deployed AGLA service URL for cross-modal verification
- *(Optional)* A CUDA-capable GPU for local vision model inference (BLIP, LLaVA-HF)

### System dependencies (macOS/Linux)

No special system packages are required beyond a working Python installation. On Linux, ensure `libgomp` is installed if you intend to run PyTorch locally (`apt install libgomp1` on Debian/Ubuntu).

---

## Virtual environment setup

```bash
# From the repo root
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

---

## Dependency installation

```bash
pip install --upgrade pip
pip install -r socrates_system/requirements.txt
```

### Download the spaCy NER model

The system defaults to the transformer-based model (`en_core_web_trf`) for highest entity recognition accuracy. A lightweight fallback (`en_core_web_sm`) can be used in resource-constrained environments.

```bash
# Recommended (used by default in config.py)
python -m spacy download en_core_web_trf

# Lightweight fallback
python -m spacy download en_core_web_sm
```

---

## Environment variables and configuration

All tuneable settings are read from environment variables at startup. Static defaults live in `socrates_system/config.py`.

### Core LLM settings

| Variable | Default | Description |
|---|---|---|
| `SOC_LLM_PROVIDER` | `ollama` | Backend: `ollama`, `openai`, `claude`, `llava_hf`, `llava_original` |
| `SOC_LLM_MODEL` | provider-dependent | Model name (e.g. `llama3.1:8b`, `gpt-4o-mini`, `claude-3-haiku-20240307`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### API keys

| Variable | Required when | Description |
|---|---|---|
| `OPENAI_API_KEY` | `SOC_LLM_PROVIDER=openai` | OpenAI API key |
| `ANTHROPIC_API_KEY` | `SOC_LLM_PROVIDER=claude` | Anthropic Claude API key |
| `GOOGLE_FACT_CHECK_API_KEY` | Optional | Google Fact Check Tools API key |

### AGLA cross-modal service

| Variable | Default | Description |
|---|---|---|
| `AGLA_API_URL` | `""` (disabled) | Base URL of the remote AGLA service |
| `AGLA_API_VERIFY_PATH` | `/verify` | Endpoint path on the AGLA service |
| `AGLA_API_TIMEOUT` | `120` | Request timeout in seconds |

### Pipeline feature toggles

| Variable | Default | Description |
|---|---|---|
| `FACTUALITY_ENABLED` | `true` | Enable/disable external factuality checking |
| `CLARIFICATION_ENABLED` | `true` | Enable/disable the clarification resolution module |
| `QUESTION_GEN_ENABLED` | `true` | Enable/disable Socratic question generation |
| `SOC_ROUTER_MODE` | `llm` | Routing mode: `llm`, `deterministic`, `hybrid` |
| `SOC_CONFLICT_MODE` | `auto` | Conflict resolution: `auto` or `manual` |
| `SOC_SESSION_ID` | auto UUID | Pin a session ID across pipeline runs |
| `FACTUALITY_CONTEXT_MODE` | `SOCRATIC_QUESTIONS` | Context passed to factuality LLM: `SOCRATIC_QUESTIONS`, `EXTRACTED_CLAIMS`, `NONE` |
| `SOC_POST_FACTUALITY_CLAR` | `true` | Apply clarification corrections to model output |
| `SOC_MITM_MIN_CONF` | `0.55` | Minimum confidence score to apply a MitM correction |
| `SOC_LOG_LEVEL` | `INFO` | Log verbosity for pipeline and evaluation scripts |
| `NO_COLOR` | unset | Set to any value to disable ANSI colours in CLI output |

### Static thresholds (config.py)

These can be changed directly in `socrates_system/config.py` but should be treated as environment-invariant defaults:

| Constant | Value | Purpose |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum claim confidence to accept |
| `CONTRADICTION_THRESHOLD` | 0.3 | Threshold for flagging contradictions |
| `SIMILARITY_THRESHOLD` | 0.4 | Semantic similarity cutoff for deduplication |
| `CATEGORIZATION_CONFIDENCE_THRESHOLD` | 0.75 | Minimum confidence for LLM-assigned category |
| `MAX_CONTENT_LENGTH` | 16 MB | Maximum upload size for Flask file uploads |

### Setting variables for a single run

```bash
# Linux/macOS inline
SOC_LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python -m socrates_system.pipeline \
  --text "The Eiffel Tower is located in Berlin."
```

Or export them before running:

```bash
export SOC_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

---

## Running the project locally

### CLI — pipeline mode

```bash
# Basic text verification with Ollama (default)
python -m socrates_system.pipeline --text "The Eiffel Tower is located in Berlin."

# Text + image with OpenAI
python -m socrates_system.pipeline \
  --text "The image shows a red car parked in front of the building." \
  --image /path/to/image.jpg \
  --llm-provider openai \
  --llm-model gpt-4o-mini

# Disable external factuality for faster runs
python -m socrates_system.pipeline \
  --text "Water boils at 100 degrees Celsius." \
  --disable-factuality

# Show session knowledge graph after verification
python -m socrates_system.pipeline --text "Einstein developed relativity in 1905." --show-kg

# Full list of options
python -m socrates_system.pipeline --help
```

### Flask Web API

```bash
# From socrates_system/
python app.py
# Listens on http://0.0.0.0:5000
```

Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/start_session` | Create a new session, returns `session_id` |
| `POST` | `/verify_claim` | Verify text (+ optional image upload) |
| `GET` | `/session_summary/<session_id>` | Session statistics |
| `GET` | `/knowledge_graph/<session_id>` | Export session knowledge graph |
| `GET` | `/api/health` | Health check |

Quick test:

```bash
curl -X POST http://localhost:5000/start_session
curl -X POST http://localhost:5000/verify_claim \
  -F "user_input=The Eiffel Tower is in Berlin." \
  -F "session_id=session_20250425_120000"
curl http://localhost:5000/api/health
```

### Simplified Flask variant

A lighter-weight app with no LLM dependencies beyond the simple claim extractor:

```bash
python socrates_system/app_simple.py
```

---

## Running the test suite

Tests are located in `socrates_system/tests/` and use `unittest.mock` to stub heavy LLM and model dependencies. No live API keys or running services are required.

```bash
# Run all tests
python -m pytest socrates_system/tests/ -v

# Run a specific test file
python -m pytest socrates_system/tests/test_pipeline_qg.py -v

# Show logger output (disable capture)
python -m pytest socrates_system/tests/ -v -s

# Stop on first failure
python -m pytest socrates_system/tests/ -x
```

---

## Troubleshooting common issues

### `ModuleNotFoundError: No module named 'socrates_system'`

Run the pipeline as a module from the **repo root** (the directory containing `socrates_system/`):

```bash
cd /path/to/Socrates
python -m socrates_system.pipeline --help
```

### `OSError: [E050] Can't find model 'en_core_web_trf'`

Download the spaCy model:

```bash
python -m spacy download en_core_web_trf
# or for the lightweight fallback:
python -m spacy download en_core_web_sm
```

The code will fall back to `en_core_web_sm` automatically if the transformer model is missing; you will see a warning in the logs.

### Ollama connection refused

Ensure Ollama is running before starting the pipeline:

```bash
ollama serve
# then in another terminal:
python -m socrates_system.pipeline --text "..."
```

If Ollama is on a non-default port, set `OLLAMA_BASE_URL`:

```bash
OLLAMA_BASE_URL=http://localhost:11435 python -m socrates_system.pipeline --text "..."
```

### AGLA API timeout / startup failure

If `AGLA_API_URL` is set, the Flask app (`app.py`) will wait for the remote AGLA service to become ready on startup and will exit with an error if it is unreachable. Either:
- Remove / unset `AGLA_API_URL` to disable remote AGLA, or
- Ensure the remote service is healthy before starting `app.py`.

### `sentence_transformers` or `torch` import errors

These are heavy optional dependencies. The system falls back gracefully when they are unavailable, but you can install them explicitly:

```bash
pip install sentence-transformers torch
```

For GPU acceleration on CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 413 Request Entity Too Large

The Flask app caps uploads at 16 MB (`MAX_CONTENT_LENGTH` in `config.py`). Compress or resize your image before uploading, or raise the limit in `config.py`.

### SQLite database locked

If multiple processes attempt to write to `data/knowledge_graph.db` simultaneously, SQLite may report a lock error. Ensure only one pipeline or Flask instance writes to the same `DATA_DIR` at a time, or set different `DATA_DIR` paths per process via a custom `config.py`.
