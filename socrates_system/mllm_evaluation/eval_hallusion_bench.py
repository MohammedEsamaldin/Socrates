import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseEvaluator


class HallusionBenchEvaluator(BaseEvaluator):
    BENCHMARK_NAME = "hallusion_bench"

    def __init__(
        self,
        *,
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
        hb_output_path: Optional[str] = None,
        hb_use_corrected: bool = False,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            run_dir=run_dir,
            provider=provider,
            model_name=model_name,
            limit=limit,
            resume=resume,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
            prompt_key=prompt_key,
            id_key=id_key,
            fallback_keys=fallback_keys,
            image_key=image_key,
            image_root=image_root,
        )
        # Default outputs to dataset directory per official evaluation.py expectations
        dataset_dir = os.path.dirname(os.path.abspath(self.dataset_path))
        self.hb_output_path = hb_output_path or os.path.join(dataset_dir, "HallusionBench_result.json")
        # Whether to place corrected output instead of the raw model output
        self.hb_use_corrected = hb_use_corrected
        # If image_root not provided, infer HallusionBench images root: <dataset_dir>/hallusion_bench
        if self.image_root is None:
            inferred_root = os.path.join(dataset_dir, "hallusion_bench")
            if os.path.isdir(inferred_root):
                self.image_root = inferred_root

    def get_sample_id(self, sample: Dict[str, Any]):
        # Common IDs seen across VQA-like datasets
        for k in [self.id_key, "question_id", "qid", "uid", "id", "idx", "index"]:
            if k and sample.get(k) is not None:
                return sample.get(k)
        return super().get_sample_id(sample)

    def sample_to_prompt(self, sample: Dict[str, Any]) -> str:
        # HallusionBench is question-centric
        for k in [self.prompt_key, "question", "instruction", "prompt", "query", "text"]:
            if k and isinstance(sample.get(k), str) and sample[k].strip():
                return sample[k]
        return super().sample_to_prompt(sample)

    # Derive image path from HallusionBench schema if not explicitly provided in the sample.
    def sample_to_image_path(self, sample: Dict[str, Any]) -> Optional[str]:
        # If explicit path/url fields exist, defer to base behavior
        explicit = super().sample_to_image_path(sample)
        if explicit:
            return explicit
        # Respect visual_input flag (0 -> no image)
        vis_raw = sample.get("visual_input")
        try:
            vis_flag = int(vis_raw) if vis_raw is not None else None
        except Exception:
            vis_flag = None
        if vis_flag == 0:
            return None
        # Build path: <image_root>/<category>/<subcategory>/<set_id>_<figure_id>.png
        cat = sample.get("category")
        sub = sample.get("subcategory")
        set_id = sample.get("set_id")
        fig_id = sample.get("figure_id")
        if not all(x is not None for x in [cat, sub, set_id, fig_id]):
            return None
        filename = f"{set_id}_{fig_id}.png"
        root = self.image_root or os.path.dirname(os.path.abspath(self.dataset_path))
        candidate = os.path.join(root, str(cat), str(sub), filename)
        return candidate

    def run(self) -> None:
        # Load dataset (we also need the original entries to write model_prediction back)
        data = self.load_dataset(self.dataset_path)
        self.logger.info(f"Loaded {len(data)} samples from {self.dataset_path}")

        processed_ids = self.ckpt.load_processed_ids() if self.resume else set()
        self.logger.info(f"Resuming with {len(processed_ids)} already processed samples")

        # Prepare container for HallusionBench_result.json (same ordering as dataset)
        hb_augmented: List[Dict[str, Any]] = []
        # When resuming, load prior predictions so we can still output a complete HB file
        prev_preds: Dict[Any, str] = {}
        if self.resume and processed_ids:
            try:
                with open(self.ckpt.results_path, "r", encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        sid = obj.get("sample_id")
                        if sid in processed_ids:
                            pred = obj.get("model_output_corrected") if self.hb_use_corrected else obj.get("model_output_original")
                            if isinstance(pred, str):
                                prev_preds[sid] = pred
            except Exception:
                pass

        total = 0
        for idx, sample in enumerate(data):
            sid = sample.get("id") or sample.get("sample_id") or sample.get("question_id") or sample.get("qid") or idx
            if self.resume and sid in processed_ids:
                # When resuming, still pass-through the original sample and fill model_prediction
                out_entry = dict(sample)
                out_entry["model_prediction"] = prev_preds.get(sid, "")
                hb_augmented.append(out_entry)
                continue

            try:
                rec = self.evaluate_sample(sample)
            except Exception as e:
                self.logger.error(f"Error evaluating sample {sid}: {e}")
                rec = {"sample_id": sid, "error": str(e)}

            # Save checkpoint record
            self.ckpt.append_result(rec, sample_id=sid, index=idx)

            # Decide what to write as model_prediction for HallusionBench
            prediction: str = rec.get("model_output_corrected") if self.hb_use_corrected else rec.get("model_output_original")
            if prediction is None:
                prediction = ""  # guard

            out_entry = dict(sample)
            out_entry["model_prediction"] = prediction
            hb_augmented.append(out_entry)

            total += 1
            if self.limit and total >= self.limit:
                self.logger.info(f"Reached sample limit {self.limit}; stopping.")
                break

        # Write HallusionBench_result.json
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.hb_output_path)), exist_ok=True)
            with open(self.hb_output_path, "w", encoding="utf-8") as f:
                json.dump(hb_augmented, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Wrote HallusionBench results to {self.hb_output_path}")
        except Exception as e:
            self.logger.error(f"Failed to write HallusionBench_result.json: {e}")

        self.logger.info("Evaluation complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate HallusionBench with Socrates MITM pipeline")
    p.add_argument("--dataset", required=True, help="Path to HallusionBench.json")
    p.add_argument("--run-dir", default=os.path.join("mllm_evaluation", "runs"), help="Directory to store run outputs")
    p.add_argument("--provider", default=None, help="LLM provider (ollama|openai|claude|llava_hf)")
    p.add_argument("--model", default=None, help="Model name for the provider")
    p.add_argument("--model-id", default=None, help="Alias for --model (e.g., llava-hf/llava-1.5-7b-hf)")
    p.add_argument("--limit", type=int, default=None, help="Max samples to process")
    p.add_argument("--no-resume", action="store_true", help="Disable resume from checkpoints")
    p.add_argument("--max-gen-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--prompt-key", default=None, help="Override prompt field name in dataset")
    p.add_argument("--id-key", default=None, help="Override id field name in dataset")
    p.add_argument("--image-key", default=None, help="Override image field name in dataset (e.g., image/image_path)")
    p.add_argument("--image-root", default=None, help="Directory to resolve relative image paths")
    # MitM toggles (mirror eval_mmhal)
    p.add_argument("--no-mitm", action="store_true", help="Disable MitM corrections entirely (sets SOC_USE_MITM=false)")
    p.add_argument("--no-mitm-input", action="store_true", help="Disable input (pre) MitM corrections")
    p.add_argument("--no-mitm-output", action="store_true", help="Disable output (post) MitM corrections")
    p.add_argument("--mitm-min-conf", type=float, default=None, help="Minimum resolution confidence to apply a correction [0-1]")
    # HallusionBench-specific output options
    p.add_argument("--hb-output", default=None, help="Path to write HallusionBench_result.json (defaults to the dataset directory alongside HallusionBench.json)")
    p.add_argument("--hb-use-corrected", action="store_true", help="Use corrected output instead of raw model output for model_prediction")
    return p


def main():
    args = build_arg_parser().parse_args()

    # Apply CLI overrides to env for MitM toggles (mirror eval_mmhal)
    if args.no_mitm:
        os.environ["SOC_USE_MITM"] = "false"
    if args.no_mitm_input:
        os.environ["SOC_MITM_VERIFY_INPUT"] = "false"
    if args.no_mitm_output:
        os.environ["SOC_MITM_VERIFY_OUTPUT"] = "false"
    if args.mitm_min_conf is not None:
        os.environ["SOC_MITM_MIN_CONF"] = str(args.mitm_min_conf)

    # Prefer --model-id over --model if provided
    model_name = args.model_id or args.model

    evaluator = HallusionBenchEvaluator(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        provider=args.provider,
        model_name=model_name,
        limit=args.limit,
        resume=not args.no_resume,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        prompt_key=args.prompt_key,
        id_key=args.id_key,
        image_key=args.image_key,
        image_root=args.image_root,
        hb_output_path=args.hb_output,
        hb_use_corrected=args.hb_use_corrected,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
