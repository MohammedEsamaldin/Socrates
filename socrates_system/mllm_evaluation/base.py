import os
import json
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
    ) -> None:
        self.dataset_path = dataset_path
        self.run_dir = os.path.join(run_dir, self.BENCHMARK_NAME)
        os.makedirs(self.run_dir, exist_ok=True)

        self.logger = setup_run_logger(self.run_dir, name=self.BENCHMARK_NAME)
        self.ckpt = CheckpointManager(self.run_dir)

        # Build model (system-under-test) and pipeline
        self.llm_manager: LLMManager = build_llm_manager(provider=provider, model_name=model_name)
        self.pipeline = build_pipeline(
            llm_manager=self.llm_manager,
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

        # Write meta
        self.ckpt.write_meta({
            "benchmark": self.BENCHMARK_NAME,
            "dataset_path": os.path.abspath(self.dataset_path),
            "provider": getattr(self.llm_manager, "provider").value,
            "model": getattr(self.llm_manager, "model_name"),
            "max_gen_tokens": self.max_gen_tokens,
            "temperature": self.temperature,
        })

    # ------ to be implemented by subclasses ------
    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        return load_dataset_generic(path)

    def sample_to_prompt(self, sample: Dict[str, Any]) -> str:
        return get_prompt_text(sample, key_override=self.prompt_key, fallback_keys=self.fallback_keys)
    # ------------------------------------------------

    def get_sample_id(self, sample: Dict[str, Any]):
        """Hook for subclasses to extract a stable sample identifier."""
        if self.id_key and sample.get(self.id_key) is not None:
            return sample.get(self.id_key)
        return sample.get("id") or sample.get("sample_id") or sample.get("idx")

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = self.get_sample_id(sample)
        prompt = self.sample_to_prompt(sample)

        # 1) User turn: correct minimally
        user_res = process_user_turn(self.pipeline, prompt)
        edited_prompt = user_res["corrected_text"] or prompt

        # 2) Model generation on edited prompt
        try:
            model_out = self.llm_manager.generate_text(
                edited_prompt,
                max_tokens=self.max_gen_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            self.logger.error(f"Generation error for sample {sample_id}: {e}")
            model_out = ""

        # 3) Model turn: correct minimally
        model_res = process_model_turn(self.pipeline, model_out)

        record = {
            "sample_id": sample_id,
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
        }
        return record

    def run(self) -> None:
        data = self.load_dataset(self.dataset_path)
        self.logger.info(f"Loaded {len(data)} samples from {self.dataset_path}")

        processed_ids = self.ckpt.load_processed_ids() if self.resume else set()
        self.logger.info(f"Resuming with {len(processed_ids)} already processed samples")

        total = 0
        for idx, sample in enumerate(data):
            sid = sample.get("id") or sample.get("sample_id") or idx
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
