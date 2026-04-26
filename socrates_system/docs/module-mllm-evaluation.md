# Module: mllm_evaluation

**Directory**: `socrates_system/mllm_evaluation/`

## Purpose

Evaluation harness for measuring hallucination detection performance of the Socrates pipeline on standard MLLM benchmarks. Each benchmark is implemented as a subclass of `BaseEvaluator`, which applies the MitM pipeline to every sample, collects results, and writes them to a JSONL checkpoint file for resumable runs.

---

## BaseEvaluator (base.py)

### `BaseEvaluator`

```python
class BaseEvaluator:
    BENCHMARK_NAME: str = "generic"

    def __init__(
        self,
        dataset_path: str,
        run_dir: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: Optional[int] = None,
        resume: bool = True,
        max_gen_tokens: int = 512,
        temperature: float = 0.2,
        prompt_key: Optional[str] = None,
        id_key: Optional[str] = None,
        fallback_keys: Optional[List[str]] = None,
        image_key: Optional[str] = None,
        image_root: Optional[str] = None,
        sut_provider: Optional[str] = None,
        sut_model_name: Optional[str] = None,
        pipeline_provider: Optional[str] = None,
        pipeline_model_name: Optional[str] = None,
        force_yes_no: bool = False,
    ) -> None
```

**Key attributes**:

| Attribute | Description |
|-----------|-------------|
| `pipeline_llm_manager` | `LLMManager` for claim extraction / factuality |
| `sut_llm_manager` | `LLMManager` for the system-under-test (may differ from pipeline) |
| `ckpt` | `CheckpointManager` for resumable JSONL output |
| `logger` | Run-specific file logger |

Subclasses must implement:
- `load_dataset(path) -> List[Dict[str, Any]]`
- `sample_to_prompt(sample) -> str`

Optionally override:
- `get_sample_id(sample) -> Any`
- `get_image_path(sample) -> Optional[str]`
- `evaluate_results(results) -> Dict[str, Any]`

---

### Key base methods

#### `run()`

Main evaluation loop. For each sample:
1. Generate the SUT answer via the pipeline's `process_user_turn`.
2. Apply MitM post-processing via `process_model_turn`.
3. Write the result to the checkpoint.

Handles `limit` (early stop), `resume` (skip already-checkpointed samples), and per-sample error recovery.

---

## Benchmark Evaluators

| Class | File | Benchmark |
|-------|------|-----------|
| `MMEEvaluator` | `eval_mme.py` | MME (Multimodal Evaluation) |
| `MMHalEvaluator` | `eval_mmhal.py` | MMHal-Bench |
| `AMBEREvaluator` | `eval_amber.py` | AMBER |
| `POPEEvaluator` | `eval_pope.py` | POPE |
| `HallusionBenchEvaluator` | `eval_hallusion_bench.py` | HallusionBench |
| `SEEDEvaluator` | `eval_seed.py` | SEED |

Each subclass overrides `load_dataset` and `sample_to_prompt` to handle its specific data format.

### `MMEEvaluator`

```python
class MMEEvaluator(BaseEvaluator):
    BENCHMARK_NAME = "mme"
```

Loads from the `MME_Benchmark` directory structure (category subdirs with paired image + `.txt` QA files). Supports `force_yes_no=True` to constrain model output to Yes/No answers.

---

## MitM Bridge (mitm.py)

### `build_pipeline(llm_manager) -> SocratesPipeline`

```python
def build_pipeline(llm_manager: LLMManager) -> SocratesPipeline
```

Creates a `SocratesPipeline` instance from the provided `LLMManager`. Respects `SOC_USE_MITM` and feature toggle env vars.

### `process_user_turn(pipeline, text, image_path=None) -> Tuple[str, List[ExtractedClaim]]`

```python
def process_user_turn(
    pipeline: SocratesPipeline,
    text: str,
    image_path: Optional[str] = None,
) -> Tuple[str, List[ExtractedClaim]]
```

Runs claim extraction and pre-routing clarification on the user prompt. Returns the (possibly corrected) text and the list of extracted claims.

### `process_model_turn(pipeline, model_output, claims, image_path=None) -> str`

```python
def process_model_turn(
    pipeline: SocratesPipeline,
    model_output: str,
    claims: List[Any],
    image_path: Optional[str] = None,
) -> str
```

Applies post-factuality clarification corrections to the model's output, returning corrected text.

### `_compute_corrected_text(original_text, claims, clar_results, stage, min_conf=None)`

```python
def _compute_corrected_text(
    original_text: str,
    claims: List[Any],
    clar_results: Dict[int, Dict[str, Any]],
    stage: str = "pre",
    min_conf: Optional[float] = None,
) -> Tuple[str, List[Dict[str, Any]]]
```

Applies minimal span-level replacements from `pipeline._clarification_results`. Applies a polarity-flip guard (controlled by `SOC_ALLOW_POLARITY_FLIP` env) and a minimum confidence threshold (`SOC_MITM_MIN_CONF`, default 0.55).

---

## Providers (providers/)

### `LlavaHFGenerator` (providers/llava_hf.py)

HuggingFace LLaVA model wrapper. Uses a singleton `get()` method for cached model loading.

```python
class LlavaHFGenerator:
    @classmethod
    def get(
        cls,
        model_name: str,
        no_4bit: bool = False,
        use_slow_tokenizer: bool = False,
    ) -> "LlavaHFGenerator"

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str
```

Environment toggles:
- `SOC_LLAVA_NO_4BIT=1`: disable 4-bit quantization.
- `SOC_LLAVA_SLOW_TOKENIZER=1`: use slow tokenizer.

---

## Utilities (utils/)

| File | Class / Function | Description |
|------|-----------------|-------------|
| `checkpointing.py` | `CheckpointManager` | Resumable JSONL output with state.json for processed IDs |
| `logging_utils.py` | `setup_run_logger` | Per-run file logger under `run_dir` |
| `model_io.py` | `build_llm_manager` | Helper to build `LLMManager` from provider/model strings |
| `dataset.py` | `load_dataset_generic`, `get_prompt_text` | Generic dataset loading (JSONL, JSON, CSV, dir) |
| `serialization.py` | `to_jsonable` | Recursively converts objects to JSON-serializable dicts |
| `fill_model_answers.py` | — | Utility to fill model answer fields in benchmark output files |

---

## Usage Example

```bash
# Run MME evaluation
python -m socrates_system.mllm_evaluation.eval_mme \
  --dataset-path socrates_system/mllm_evaluation/datasets/MME_Benchmark \
  --run-dir ./runs/mme_run1 \
  --provider openai \
  --model-name gpt-4o-mini \
  --limit 100
```

Results are written to `runs/mme_run1/mme/results.jsonl`. Resuming a run automatically skips already-processed sample IDs.
