import os
import json
import re
from typing import Any, Dict, List, Optional, Iterable

from .utils.checkpointing import CheckpointManager
from .utils.logging_utils import setup_run_logger
from .utils.model_io import build_llm_manager
from .mitm import build_pipeline, process_user_turn, process_model_turn
from .utils.serialization import to_jsonable
from .utils.dataset import load_dataset_generic, get_prompt_text
from socrates_system.modules.llm_manager import LLMProvider, LLMManager


class BaseEvaluator:
    """
    Base evaluator that implements man-in-the-middle execution and checkpointing.

    Subclasses must implement `load_dataset(path)` and `sample_to_prompt(sample)`
    to provide the user-facing prompt text (and optional metadata).
    """

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
        # New: allow separate providers/models for SUT vs pipeline
        sut_provider: Optional[str] = None,
        sut_model_name: Optional[str] = None,
        pipeline_provider: Optional[str] = None,
        pipeline_model_name: Optional[str] = None,
    ) -> None:
        self.dataset_path = dataset_path
        # If the provided run_dir already contains checkpoints for this run, honor it as-is.
        # Otherwise, place checkpoints under <run_dir>/<BENCHMARK_NAME> (default behavior).
        candidate_dir = run_dir
        use_as_is = False
        try:
            if os.path.isdir(candidate_dir):
                for fname in ("results.jsonl", "state.json", "mmhal_results.jsonl"):
                    if os.path.exists(os.path.join(candidate_dir, fname)):
                        use_as_is = True
                        break
        except Exception:
            use_as_is = False

        self.run_dir = candidate_dir if use_as_is else os.path.join(run_dir, self.BENCHMARK_NAME)
        os.makedirs(self.run_dir, exist_ok=True)

        self.logger = setup_run_logger(self.run_dir, name=self.BENCHMARK_NAME)
        self.ckpt = CheckpointManager(self.run_dir)

        # Build model (system-under-test) and pipeline
        # Backward compatible defaults: if not specified, use `provider`/`model_name` for both.
        _sut_provider = sut_provider or provider
        _sut_model = sut_model_name or model_name
        _pipe_provider = pipeline_provider or provider
        _pipe_model = pipeline_model_name or model_name

        # LLM used for pipeline (claims/clarification/factuality)
        self.pipeline_llm_manager: LLMManager = build_llm_manager(provider=_pipe_provider, model_name=_pipe_model)
        # Alias for backward compat (other modules may reference self.llm_manager)
        self.llm_manager: LLMManager = self.pipeline_llm_manager
        # LLM used for generating the system-under-test answer (can be different, e.g., LLaVA)
        self.sut_llm_manager: LLMManager = build_llm_manager(provider=_sut_provider, model_name=_sut_model)

        self.pipeline = build_pipeline(
            llm_manager=self.pipeline_llm_manager,
            factuality_enabled=True,
            clarification_enabled=True,
            clarification_dev_mode=False,
            question_gen_enabled=False,
        )

        self.limit = limit
        self.resume = resume
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.prompt_key = prompt_key
        self.id_key = id_key
        self.fallback_keys = fallback_keys
        self.image_key = image_key
        self.image_root = image_root

        # Write meta
        self.ckpt.write_meta({
            "benchmark": self.BENCHMARK_NAME,
            "dataset_path": os.path.abspath(self.dataset_path),
            "sut_provider": getattr(self.sut_llm_manager, "provider").value,
            "sut_model": getattr(self.sut_llm_manager, "model_name"),
            "pipeline_provider": getattr(self.pipeline_llm_manager, "provider").value,
            "pipeline_model": getattr(self.pipeline_llm_manager, "model_name"),
            "max_gen_tokens": self.max_gen_tokens,
            "temperature": self.temperature,
            "image_key": self.image_key,
            "image_root": self.image_root,
        })

    # ------ to be implemented by subclasses ------
    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        return load_dataset_generic(path)

    def sample_to_prompt(self, sample: Dict[str, Any]) -> str:
        return get_prompt_text(sample, key_override=self.prompt_key, fallback_keys=self.fallback_keys)
    # ------------------------------------------------

    def sample_to_image_path(self, sample: Dict[str, Any]) -> Optional[str]:
        """Hook to extract an image path/URL from a sample. May be overridden by subclasses.

        Default behavior checks a few common keys and resolves relative paths against
        image_root or the dataset directory.
        """
        # Preferred override
        keys: List[Optional[str]] = [self.image_key, "image", "image_path", "image_file", "img", "imageUrl", "imageURL", "image_uri"]
        val: Optional[str] = None
        for k in keys:
            if k and isinstance(sample.get(k), str) and sample[k].strip():
                val = sample[k].strip()
                break
        if not val:
            return None
        # URLs are returned as-is (caller may handle downloading if needed)
        if val.startswith("http://") or val.startswith("https://"):
            return val
        # Resolve relative paths
        if os.path.isabs(val):
            return val
        root = self.image_root or os.path.dirname(os.path.abspath(self.dataset_path))
        candidate = os.path.join(root, val)
        return candidate

    def _claims_to_mmhal(self,
                         claims: List[Any],
                         clar_results: Dict[int, Dict[str, Any]],
                         factuality: Dict[int, Any],
                         stage: str) -> List[Dict[str, Any]]:
        """Convert pipeline claims + clarifications + factuality into MMHal-style items.

        Each item contains: index, text, span, categories, corrected_text (if any),
        factuality (status/confidence/evidence/etc), clarification (stage-specific).
        """
        items: List[Dict[str, Any]] = []
        if not claims:
            return items
        for i, cl in enumerate(claims, 1):
            # categories -> names
            cat_names: List[str] = []
            try:
                for c in getattr(cl, "categories", []) or []:
                    nm = getattr(c, "name", None)
                    if nm is None:
                        continue
                    try:
                        cat_names.append(nm.name)
                    except Exception:
                        cat_names.append(str(nm))
            except Exception:
                pass

            # clarification at stage
            clar = (clar_results or {}).get(i, {})
            clar_stage_obj = clar.get(stage)
            corrected_text = None
            if clar_stage_obj is not None:
                try:
                    ct = getattr(clar_stage_obj, "corrected_claim", None)
                    if isinstance(ct, str) and ct.strip():
                        corrected_text = ct
                except Exception:
                    pass

            item = {
                "index": i,
                "text": getattr(cl, "text", None),
                "span": [int(getattr(cl, "start_char", -1)), int(getattr(cl, "end_char", -1))],
                "categories": cat_names,
                "corrected_text": corrected_text,
                "clarification": to_jsonable(clar_stage_obj) if clar_stage_obj is not None else None,
                "factuality": to_jsonable((factuality or {}).get(i)),
            }
            items.append(item)
        return items

    def _detect_yes_no(self, text: str) -> Optional[bool]:
        """Detect if a response is effectively a yes/no answer.

        Returns True for yes, False for no, None if indeterminate.
        """
        if not text:
            return None
        s = text.strip().lower()
        # Normalize punctuation and commas
        s = re.sub(r"^[^a-zA-Z]*(yes|no)\b.*$", r"\1", s)
        # Single-token quick path
        if s in {"yes", "y", "yeah", "yep", "true", "correct", "affirmative"}:
            return True
        if s in {"no", "n", "nope", "false", "incorrect", "negative"}:
            return False
        # Look at the beginning of the string for a yes/no cue
        m = re.match(r"^(yes|no)[\s,!.]?\b", text.strip().lower())
        if m:
            return True if m.group(1) == "yes" else False
        return None

    def _question_to_declarative(self, question: str, answer_yes: bool) -> str:
        """Convert a yes/no question into a simple declarative hypothesis.

        This is a conservative heuristic. For unhandled forms, we fall back to
        "It is (not) the case that <question_without_?>".
        """
        if not question:
            return ""
        q = question.strip().strip("?").strip()
        low = q.lower()
        def yes(expr: str) -> str:
            return expr if answer_yes else f"not {expr}".strip()

        # Handle "Is there ..." / "Are there ..."
        if low.startswith("is there "):
            core = q[len("is there "):]
            return f"There is {core}" if answer_yes else f"There is not {core}"
        if low.startswith("are there "):
            core = q[len("are there "):]
            return f"There are {core}" if answer_yes else f"There are not {core}"

        # Default: wrap as a generic hypothesis
        if answer_yes:
            return q
        return f"It is not the case that {q}"

    def get_sample_id(self, sample: Dict[str, Any]):
        """Hook for subclasses to extract a stable sample identifier."""
        if self.id_key and sample.get(self.id_key) is not None:
            return sample.get(self.id_key)
        return sample.get("id") or sample.get("sample_id") or sample.get("idx")

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = self.get_sample_id(sample)
        prompt = self.sample_to_prompt(sample)

        # 1) User turn: correct minimally (with optional image)
        image_path = self.sample_to_image_path(sample)
        user_res = process_user_turn(self.pipeline, prompt, image_path=image_path)
        edited_prompt = user_res["corrected_text"] or prompt

        # 2) Model generation on edited prompt (SUT LLM)
        try:
            model_out = self.sut_llm_manager.generate_text(
                edited_prompt,
                max_tokens=self.max_gen_tokens,
                temperature=self.temperature,
                images=[image_path] if image_path else None,
            )
        except Exception as e:
            self.logger.error(f"Generation error for sample {sample_id}: {e}")
            model_out = ""

        # 3) Model turn: correct minimally (with optional image)
        model_res = process_model_turn(self.pipeline, model_out, image_path=image_path)
        # Determine corrected output string and log both original and corrected
        corrected_out = model_res.get("corrected_text") or model_out
        try:
            self.logger.info(f"Sample {sample_id} - Original model output: {model_out}")
            self.logger.info(f"Sample {sample_id} - Corrected model output: {corrected_out}")
        except Exception:
            pass

        # If output is terse yes/no and produced no claims, derive a hypothesis from the question
        analysis_source = "original"
        claims_for_analysis = model_res.get("claims", [])
        clar_for_analysis = model_res.get("clarification", {})
        fact_for_analysis = model_res.get("factuality", {})
        try:
            yn = self._detect_yes_no(model_out)
            if (not claims_for_analysis) and (yn is not None):
                hypo_text = self._question_to_declarative(prompt, answer_yes=bool(yn))
                if hypo_text:
                    self.logger.info(
                        "Model output appears to be yes/no without claims; deriving hypothesis for analysis"
                    )
                    derived = process_model_turn(self.pipeline, hypo_text, image_path=image_path)
                    # Use derived analysis artifacts for MMHal reporting, but keep original model_res intact
                    claims_for_analysis = derived.get("claims", [])
                    clar_for_analysis = derived.get("clarification", {})
                    fact_for_analysis = derived.get("factuality", {})
                    analysis_source = "derived_yesno"
        except Exception as _e:
            # Non-fatal; continue with original artifacts
            pass

        # Build MMHal-style output focusing on model turn claims (using analysis artifacts)
        mmhal_claims = self._claims_to_mmhal(
            claims_for_analysis,
            clar_for_analysis,
            fact_for_analysis,
            stage="post",
        )
        mmhal_prompt_claims = self._claims_to_mmhal(
            user_res.get("claims", []),
            user_res.get("clarification", {}),
            user_res.get("factuality", {}),
            stage="pre",
        )

        # Summary statistics
        num_pass = sum(1 for it in mmhal_claims if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "PASS")
        num_fail = sum(1 for it in mmhal_claims if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "FAIL")
        num_uncertain = sum(1 for it in mmhal_claims if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "UNCERTAIN")

        mmhal = {
            "version": "0.1",
            "id": sample_id,
            "image": image_path,
            "question": prompt,
            "response_original": model_out,
            "response_corrected": model_res.get("corrected_text", model_out),
            "claims": mmhal_claims,
            "prompt_claims": mmhal_prompt_claims or None,
            "summary": {
                "num_claims": len(mmhal_claims),
                "num_pass": num_pass,
                "num_fail": num_fail,
                "num_uncertain": num_uncertain,
                "has_hallucination": num_fail > 0,
            },      
        }

        record = {
            "sample_id": sample_id,
            "image_path": image_path,
            "input_original": prompt,
            "input_corrected": user_res.get("corrected_text", prompt),
            "input_corrections": user_res.get("corrections", []),
            "input_factuality": user_res.get("factuality", {}),
            "input_claims": to_jsonable(user_res.get("claims", [])),
            "input_clarification": to_jsonable(user_res.get("clarification", {})),
            "model_output_original": model_out,
            "model_output_corrected": model_res.get("corrected_text", model_out),
            "model_output_corrections": model_res.get("corrections", []),
            "model_output_factuality": model_res.get("factuality", {}),
            "model_output_claims": to_jsonable(model_res.get("claims", [])),
            "model_output_clarification": to_jsonable(model_res.get("clarification", {})),
            # Analysis artifacts (may be derived from yes/no + question hypothesis)
            "analysis_source": analysis_source,
            "analysis_claims": to_jsonable(claims_for_analysis),
            "analysis_clarification": to_jsonable(clar_for_analysis),
            "analysis_factuality": to_jsonable(fact_for_analysis),
            "mmhal": mmhal,
        }
        return record

    def run(self) -> None:
        data = self.load_dataset(self.dataset_path)
        self.logger.info(f"Loaded {len(data)} samples from {self.dataset_path}")

        processed_ids = self.ckpt.load_processed_ids() if self.resume else set()
        self.logger.info(f"Resuming with {len(processed_ids)} already processed samples")

        total = 0
        for idx, sample in enumerate(data):
            # Use the same identifier logic used elsewhere (respects --id-key and evaluator overrides)
            sid = self.get_sample_id(sample) or idx
            if self.resume and sid in processed_ids:
                continue

            try:
                rec = self.evaluate_sample(sample)
            except Exception as e:
                self.logger.error(f"Error evaluating sample {sid}: {e}")
                rec = {"sample_id": sid, "error": str(e)}

            # Save checkpoint record
            self.ckpt.append_result(rec, sample_id=sid, index=idx)

            total += 1
            if self.limit and total >= self.limit:
                self.logger.info(f"Reached sample limit {self.limit}; stopping.")
                break

        self.logger.info("Evaluation complete.")
