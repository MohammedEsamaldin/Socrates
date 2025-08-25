#!/usr/bin/env python3
import io
import os
from typing import Optional, Dict, Tuple

import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig


def _load_image(maybe_path_or_url: str) -> Image.Image:
    if isinstance(maybe_path_or_url, str) and maybe_path_or_url.startswith(("http://", "https://")):
        r = requests.get(
            maybe_path_or_url,
            timeout=200,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if not os.path.exists(maybe_path_or_url):
        raise FileNotFoundError(f"Image not found: {maybe_path_or_url}")
    return Image.open(maybe_path_or_url).convert("RGB")


class LlavaHFGenerator:
    """
    Lightweight HF-based LLaVA generator for SUT (system-under-test) decoding.
    This is intentionally minimal and only used for producing the raw model answer.
    """

    _instances: Dict[Tuple[str, bool, bool], "LlavaHFGenerator"] = {}

    def __init__(self, model_id: str, no_4bit: bool = False, use_slow_tokenizer: bool = False) -> None:
        self.model_id = model_id
        self.no_4bit = no_4bit
        self.use_slow_tokenizer = use_slow_tokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quant_config = None
        if not no_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        tokenizer_kwargs = {"use_fast": not use_slow_tokenizer}
        self.processor = AutoProcessor.from_pretrained(self.model_id, tokenizer_kwargs=tokenizer_kwargs)
        # Prefer bf16 on Ampere+ GPUs, otherwise fp16 on CUDA; CPU stays float32
        if self.device == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability(0)
                use_bf16 = major >= 8  # Ampere (8.x) or newer
            except Exception:
                use_bf16 = False
            dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            dtype = torch.float32

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
        if self.device != "cuda":
            self.model.to(self.device)

    @classmethod
    def get(cls, model_id: str, no_4bit: bool = False, use_slow_tokenizer: bool = False) -> "LlavaHFGenerator":
        key = (model_id, no_4bit, use_slow_tokenizer)
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls(model_id=model_id, no_4bit=no_4bit, use_slow_tokenizer=use_slow_tokenizer)
            cls._instances[key] = inst
        return inst

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.2,
    ) -> str:
        image = _load_image(image_path) if image_path else None

        # Build chat template with optional image
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content = [{"type": "image"}] + content
        messages = [{"role": "user", "content": content}]

        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
            )
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[-1].strip()
        return text
