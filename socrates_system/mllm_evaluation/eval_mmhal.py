import argparse
import os
from typing import Any, Dict, List, Optional

from .base import BaseEvaluator


class MMHalEvaluator(BaseEvaluator):
    BENCHMARK_NAME = "mmhal_bench"

    def get_sample_id(self, sample: Dict[str, Any]):
        # Common fields seen in MMHal-like datasets
        for k in [self.id_key, "question_id", "uid", "id", "idx"]:
            if k and sample.get(k) is not None:
                return sample.get(k)
        return super().get_sample_id(sample)

    def sample_to_prompt(self, sample: Dict[str, Any]) -> str:
        # Typical key for question-based datasets
        for k in [self.prompt_key, "question", "instruction", "prompt", "query", "text"]:
            if k and isinstance(sample.get(k), str) and sample[k].strip():
                return sample[k]
        return super().sample_to_prompt(sample)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MMHal-Bench with Socrates MITM pipeline")
    p.add_argument("--dataset", required=True, help="Path to the dataset file (json/jsonl/csv)")
    p.add_argument("--run-dir", default=os.path.join("mllm_evaluation", "runs"), help="Directory to store run outputs")
    p.add_argument("--provider", default=None, help="LLM provider (ollama|openai|claude)")
    p.add_argument("--model", default=None, help="Model name for the provider")
    p.add_argument("--limit", type=int, default=None, help="Max samples to process")
    p.add_argument("--no-resume", action="store_true", help="Disable resume from checkpoints")
    p.add_argument("--max-gen-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--prompt-key", default=None, help="Override prompt field name in dataset")
    p.add_argument("--id-key", default=None, help="Override id field name in dataset")
    return p


def main():
    args = build_arg_parser().parse_args()

    evaluator = MMHalEvaluator(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        provider=args.provider,
        model_name=args.model,
        limit=args.limit,
        resume=not args.no_resume,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        prompt_key=args.prompt_key,
        id_key=args.id_key,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
