#!/usr/bin/env python3
"""
Generate MMHal-Bench responses using Hugging Face LLaVA (llava-hf) models.

This script loads an MMHal-style dataset and produces a JSON list with the exact
schema expected by the evaluator at mllm_evaluation/analysis/MMHal_Bench/eval_gpt4.py:

Each output record contains at least the following keys:
- image_id
- question_type
- question_topic
- image_content (list[str])
- question (str)
- gt_answer (str)
- model_answer (str)

You can then run the judge script, for example:

python -m socrates_system.mllm_evaluation.analysis.MMHal_Bench.eval_gpt4 \
  --response responses/llava15_7b.json \
  --evaluation responses/llava15_7b_eval.json \
  --api-key $OPENAI_API_KEY \
  --gpt-model gpt-4o-2024-08-06

Notes:
- Enforces fp16/bf16 precision (no 4-bit) for LLaVA-HF per project requirements.
- Resolves image paths robustly, supporting common keys and an optional --image-root.
- If some optional fields are missing in the dataset, reasonable defaults are applied so
  the judge script can still run (e.g., empty image_content list).
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from pathlib import Path

from socrates_system.mllm_evaluation.providers.llava_hf import LlavaHFGenerator


def load_dataset(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if p.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        return records
    # default: JSON
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        raise ValueError("Unsupported dataset JSON format; expected a list of records or an object with 'data' list")


def resolve_image_path(sample: Dict[str, Any], image_root: Optional[str], dataset_dir: str) -> Optional[str]:
    # Common keys; follow BaseEvaluator.sample_to_image_path patterns
    keys: List[Optional[str]] = [
        "image",
        "image_path",
        "image_file",
        "img",
        "imageUrl",
        "imageURL",
        "image_uri",
    ]
    val: Optional[str] = None
    for k in keys:
        if k and isinstance(sample.get(k), str) and sample[k].strip():
            val = sample[k].strip()
            break
    if not val:
        return None
    if val.startswith("http://") or val.startswith("https://"):
        return val
    # resolve local path
    if image_root:
        candidate = os.path.join(image_root, val)
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(dataset_dir, val)
    if os.path.exists(candidate):
        return candidate
    # fallback to raw
    return val


def ensure_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        # If comma-separated, split, else single item list
        parts = [t.strip() for t in x.split(",")]
        return [p for p in parts if p] or [x]
    return []


def main():
    ap = argparse.ArgumentParser(description="Generate MMHal-Bench responses with LLaVA-HF")
    ap.add_argument("--dataset", required=True, help="Path to MMHal dataset (json/jsonl)")
    ap.add_argument("--output", required=True, help="Path to write responses JSON")
    ap.add_argument("--image-root", default=None, help="Optional directory to resolve relative image paths")
    ap.add_argument("--model-id", default=os.environ.get("SOC_LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf"),
                   help="HF model id, e.g., llava-hf/llava-1.5-7b-hf or llava-hf/llava-1.5-13b-hf")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--start", type=int, default=0, help="Start index (for partial runs)")
    ap.add_argument("--limit", type=int, default=None, help="Max samples to process")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    dataset_dir = str(Path(args.dataset).resolve().parent)

    gen = LlavaHFGenerator.get(model_id=args.model_id, no_4bit=True)

    outputs: List[Dict[str, Any]] = []

    start = max(0, int(args.start or 0))
    end = len(dataset)
    if args.limit is not None:
        end = min(end, start + int(args.limit))

    print(f"Loaded {len(dataset)} samples. Processing range: [{start}, {end}) with model {args.model_id}")

    for idx in range(start, end):
        sample = dataset[idx]
        # Extract fields with fallbacks
        image_id = sample.get("image_id") or sample.get("id") or sample.get("uid")
        question = (
            sample.get("question")
            or sample.get("instruction")
            or sample.get("prompt")
            or sample.get("query")
            or ""
        )
        gt_answer = sample.get("gt_answer") or sample.get("standard_answer") or sample.get("answer") or ""
        question_type = sample.get("question_type")
        question_topic = sample.get("question_topic")
        image_content = ensure_list_str(sample.get("image_content"))

        image_path = resolve_image_path(sample, args.image_root, dataset_dir)

        # Generate with LLaVA-HF
        try:
            model_answer = gen.generate(
                prompt=question,
                image_path=image_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"[WARN] Generation failed at index {idx} (image_id={image_id}): {e}")
            model_answer = ""

        out_rec = {
            "image_id": image_id,
            "question_type": question_type,
            "question_topic": question_topic,
            "image_content": image_content,
            "question": question,
            "gt_answer": gt_answer,
            "model_answer": model_answer,
        }

        # Preserve any extra fields from the original sample if needed
        # out_rec.update({k: v for k, v in sample.items() if k not in out_rec})

        outputs.append(out_rec)
        print(f"[{idx}] image_id={image_id} | len(answer)={len(model_answer)}")

    # Write JSON list
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(outputs)} responses -> {out_path}")


if __name__ == "__main__":
    main()
