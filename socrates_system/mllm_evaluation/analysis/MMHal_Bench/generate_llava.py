#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)


def load_image(path: str) -> Image.Image:
    """Load an image from disk and convert it to RGB.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        An RGB :class:`PIL.Image.Image` object.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    return img.convert("RGB")


def resolve_image_path(
    record: Dict[str, Any],
    images_dir: str,
    image_key: str,
    image_ext: Optional[str],
) -> str:
    """Resolve the image file path for a dataset record.

    Tries common image field names (``image_path``, ``image``, etc.) and falls
    back to constructing a path from ``image_id`` and ``image_ext``.

    Args:
        record: A single dataset record dict.
        images_dir: Root directory for resolving relative image paths.
        image_key: Primary field name to look up first.
        image_ext: Fallback extension (e.g., ``"jpg"``) used when the path is
            constructed from ``image_id``.

    Returns:
        An absolute or relative path string to the image file.

    Raises:
        KeyError: If no image path can be resolved from the record.
    """
    # Try common keys first
    for k in [image_key, "image_path", "image", "image_name", "img", "filename"]:
        if k in record and record[k]:
            p = record[k]
            return p if os.path.isabs(p) else os.path.join(images_dir, p)

    # Try constructing from image_id + extension
    if image_ext and "image_id" in record and record["image_id"] is not None:
        return os.path.join(images_dir, f"{record['image_id']}.{image_ext.lstrip('.')}")

    raise KeyError(
        "Could not resolve image path. Provide --image-key that exists in records "
        "or ensure records contain one of: image_path, image, image_name, img, filename; "
        "or provide image_id with --image-ext."
    )


def build_llava_inputs(processor: AutoProcessor, image: Image.Image, question: str):
    """Build tokenised inputs for a LLaVA-HF model from an image and question.

    Args:
        processor: The HuggingFace processor associated with the LLaVA model.
        image: An RGB PIL image to pass as the visual context.
        question: The text question to ask about the image.

    Returns:
        A dict of tensors (``input_ids``, ``pixel_values``, etc.) ready for model forward pass.
    """
    # LLaVA HF uses a chat template with an implicit <image> token
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return inputs


def main():
    """Entry point: generate LLaVA answers for MMHal-Bench records and write output JSON."""
    parser = argparse.ArgumentParser(
        description="Generate LLaVA-1.5-7B answers for MMHal-Bench and save to JSON compatible with eval_gpt4.py",
    )
    parser.add_argument("--dataset", required=True, help="Path to input JSON records (must include 'question' and an image reference)")
    parser.add_argument("--images-dir", required=True, help="Directory containing MMHal-Bench images")
    parser.add_argument("--output", required=True, help="Where to write the filled JSON with model_answer")
    parser.add_argument("--image-key", default="image", help="Key in each record that points to the image file (default: image)")
    parser.add_argument("--image-ext", default=None, help="Fallback extension (e.g., jpg/png) if using image_id")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf", help="HF model id for LLaVA-1.5-7B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (uses more VRAM)")
    parser.add_argument("--batch-size", type=int, default=1, help="Not batched by default; kept for future batching")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading model: {args.model_id} (device={device}, 4bit={not args.no_4bit})")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )
    if device != "cuda":
        model.to(device)

    with open(args.dataset, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    total = len(records)
    print(f"Loaded {total} records from {args.dataset}")

    for i, rec in enumerate(records):
        try:
            img_path = resolve_image_path(rec, args.images_dir, args.image_key, args.image_ext)
            image = load_image(img_path)
            inputs = build_llava_inputs(processor, image, rec.get("question", ""))
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            # Decode; for HF LLaVA, this returns the chat completion text
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            # Heuristic: strip any preceding roles
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:", 1)[-1].strip()
            rec["model_answer"] = text
        except Exception as e:
            rec["model_answer"] = ""
            print(f"[WARN] idx={i}: generation failed: {e}")

        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{total}")

    # Write filled JSON compatible with eval_gpt4.py (expects model_answer present)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved filled responses to {args.output}")


if __name__ == "__main__":
    main()
