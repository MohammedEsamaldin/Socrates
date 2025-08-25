#!/usr/bin/env python3
"""
Wrap the full Socrates MMHal pipeline (eval_mmhal.py) using LLaVA-HF as the SUT,
then export a judge-ready JSON in the original MMHal-Bench schema.

This mirrors your GPT-4o judging methodology: run the MITM pipeline end-to-end
(clarification, cross-modal/external factuality via your pipeline), then evaluate the
final answer. By default we use the corrected model output (response_corrected) as
model_answer; you can switch to the raw output via --use-original.

Outputs a JSON list with fields:
- image_id
- question_type
- question_topic
- image_content (list[str])
- question (str)
- gt_answer (str)
- model_answer (str)

Files read/written by the evaluator:
- run_dir/mmhal_bench/results.jsonl         (per-sample records)
- run_dir/mmhal_bench/mmhal_results.jsonl   (one mmhal block per sample)

Usage example:
python -m socrates_system.mllm_evaluation.analysis.MMHal_Bench.wrap_eval_llava_hf \
  --dataset /path/to/mmhal.json \
  --output responses/llava15_pipeline.json \
  --model-id llava-hf/llava-1.5-7b-hf \
  --image-root /path/to/images \
  --max-gen-tokens 256 --temperature 0.2

Then judge:
python -m socrates_system.mllm_evaluation.analysis.MMHal_Bench.eval_gpt4 \
  --response responses/llava15_pipeline.json \
  --evaluation responses/llava15_pipeline_eval.json \
  --api-key $OPENAI_API_KEY \
  --gpt-model gpt-4o-2024-08-06
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from socrates_system.mllm_evaluation.eval_mmhal import MMHalEvaluator
from socrates_system.mllm_evaluation.utils.dataset import load_dataset_generic


def ensure_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        parts = [t.strip() for t in x.split(",")]
        return [p for p in parts if p] or [x]
    return []


def load_mmhal_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def build_judge_records(
    dataset: List[Dict[str, Any]],
    mmhals: List[Dict[str, Any]],
    prefer_corrected: bool = True,
) -> List[Dict[str, Any]]:
    # Attempt id-based mapping; fallback to positional
    id_to_mmhal: Dict[Any, Dict[str, Any]] = {}
    for m in mmhals:
        mid = m.get("id")
        if mid is not None:
            id_to_mmhal[mid] = m

    records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(dataset):
        mm: Optional[Dict[str, Any]] = None
        sid = rec.get("id") or rec.get("sample_id") or rec.get("uid") or rec.get("question_id") or idx
        if sid in id_to_mmhal:
            mm = id_to_mmhal[sid]
        elif idx < len(mmhals):
            mm = mmhals[idx]

        if mm is None:
            # Not processed yet; fill empty model_answer
            model_answer = ""
        else:
            model_answer = (
                mm.get("response_corrected")
                if prefer_corrected
                else mm.get("response_original")
            ) or mm.get("response_original") or ""

        out = {
            "image_id": rec.get("image_id") or rec.get("id") or rec.get("uid"),
            "question_type": rec.get("question_type"),
            "question_topic": rec.get("question_topic"),
            "image_content": ensure_list_str(rec.get("image_content")),
            "question": rec.get("question") or rec.get("instruction") or rec.get("prompt") or rec.get("query") or "",
            "gt_answer": rec.get("gt_answer") or rec.get("standard_answer") or rec.get("answer") or "",
            "model_answer": model_answer,
        }
        records.append(out)
    return records


def main():
    ap = argparse.ArgumentParser(description="Run MMHal evaluator with llava_hf and export judge-ready JSON")
    ap.add_argument("--dataset", required=True, help="Path to dataset (json/jsonl/csv)")
    ap.add_argument("--output", required=True, help="Path to write judge-ready JSON")
    ap.add_argument("--run-dir", default=os.path.join("mllm_evaluation", "runs"))
    ap.add_argument("--provider", default="llava_hf")
    ap.add_argument("--model-id", default=os.environ.get("SOC_LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-gen-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--prompt-key", default=None)
    ap.add_argument("--id-key", default=None)
    ap.add_argument("--image-key", default=None)
    ap.add_argument("--image-root", default=None)
    # MitM toggles
    ap.add_argument("--no-mitm", action="store_true", help="Disable MitM corrections entirely")
    ap.add_argument("--no-mitm-input", action="store_true", help="Disable input (pre) MitM corrections")
    ap.add_argument("--no-mitm-output", action="store_true", help="Disable output (post) MitM corrections")
    ap.add_argument("--mitm-min-conf", type=float, default=None)
    # Export controls
    ap.add_argument("--use-original", action="store_true", help="Use raw model output instead of corrected")
    ap.add_argument("--skip-run", action="store_true", help="Skip running evaluator; just transform existing results")

    args = ap.parse_args()

    # Apply MitM toggles via env (same as eval_mmhal.py)
    if args.no_mitm:
        os.environ["SOC_USE_MITM"] = "true"
    if args.no_mitm_input:
        os.environ["SOC_MITM_VERIFY_INPUT"] = "true"
    if args.no_mitm_output:
        os.environ["SOC_MITM_VERIFY_OUTPUT"] = "true"
    if args.mitm_min_conf is not None:
        os.environ["SOC_MITM_MIN_CONF"] = str(args.mitm_min_conf)

    run_dir = os.path.join(args.run_dir, "mmhal_bench")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        evaluator = MMHalEvaluator(
            dataset_path=args.dataset,
            run_dir=args.run_dir,
            provider=args.provider,
            model_name=args.model_id,
            limit=args.limit,
            resume=not args.no_resume,
            max_gen_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            prompt_key=args.prompt_key,
            id_key=args.id_key,
            image_key=args.image_key,
            image_root=args.image_root,
        )
        evaluator.run()

    mmhal_path = os.path.join(run_dir, "mmhal_results.jsonl")
    mmhals = load_mmhal_jsonl(mmhal_path)

    dataset = load_dataset_generic(args.dataset)

    judge_records = build_judge_records(dataset, mmhals, prefer_corrected=(not args.use_original))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(judge_records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(judge_records)} judge-ready records -> {out_path}")


if __name__ == "__main__":
    main()
