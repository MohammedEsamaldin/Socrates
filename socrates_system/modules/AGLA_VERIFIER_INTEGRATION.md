# AGLA Verifier Integration Guide

This guide explains how to use the reusable `AGLAVerifier` in the Socrates system.

## Overview
- AGLA verifies image-text claims in two stages:
  - Stage A: True/False constrained generation
  - Stage B: Open-ended correction when False
- The module lazily loads models on first use and supports CUDA and CPU. Half precision is used only on CUDA.
- LAVIS (BLIP-ITM) augmentation is optional; if unavailable, AGLA still works without augmentation.

## Environment
- Required:
  - `LLAVA_REPO_PATH`: path to the repo ROOT containing the `llava/` folder
    - Example: `/Users/mohammed/Desktop/Thesis/AGLA-1`
  - Internet/Hugging Face access to download the model unless cached
- Optional:
  - `LLAVA_MODEL`: LLaVA model id (default: `liuhaotian/llava-v1.5-7b`)
  - CUDA settings (e.g., `CUDA_VISIBLE_DEVICES=0`) if using GPU
  - Hugging Face token if required for gated models

## Dependencies
Ensure these packages are in your environment (versions compatible with your system):
- torch, torchvision
- transformers, accelerate
- pillow (PIL), einops, sentencepiece
- Optional for augmentation: LAVIS (BLIP-ITM) + torchvision transforms
  - If unavailable, augmentation is disabled gracefully
- A compatible `llava` package importable via `LLAVA_REPO_PATH` or installed

Example (CUDA 12.x wheels may vary by system):
```
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pillow einops sentencepiece
# Optional augmentation support
pip install git+https://github.com/salesforce/LAVIS.git
```

## Quick Start
```
from socrates_system.modules.agla_verifier import AGLAVerifier

agla = AGLAVerifier()  # uses env: LLAVA_REPO_PATH, LLAVA_MODEL; auto device
res = agla.verify_claim(
    image="/path/to/image.jpg",
    claim="There is a yellow taxi in the image.",
    use_agla=True,          # enable augmentation when available
    alpha=2.5,              # contrastive decoding alpha
    beta=0.8,               # contrastive decoding beta
    return_debug=True       # include prompts, timings
)
print(res)
# {
#   'verdict': 'True' | 'False' | 'Uncertain',
#   'truth': '... correction text when False ...',
#   'latency_ms': 1234,
#   'debug': {...}
# }
```

Inputs supported for `image`: file path (str), raw bytes, or `PIL.Image.Image`.

## Integrating with Socrates Routing
If you use the existing `CheckRouter`, route claims with category `VISUAL_GROUNDING_REQUIRED` to AGLA.

Example wiring inside your agent (pseudo-code):
```
from socrates_system.modules.agla_verifier import AGLAVerifier
from socrates_system.modules.shared_structures import VerificationMethod

class SocratesAgent:
    def __init__(self, ...):
        self.agla = AGLAVerifier()
        # ... other initializations ...

    def verify_claim_cross_modal(self, image, claim_obj):
        # image can be path/bytes/PIL.Image; claim_obj.text is the textual claim
        out = self.agla.verify_claim(image, claim_obj.text, use_agla=True, return_debug=False)

        # Map results back to the claim structure
        verdict = out.get('verdict', 'Uncertain')
        claim_obj.factuality_verdict = (verdict == 'True')
        claim_obj.factuality_status = 'PASS' if verdict == 'True' else (
            'FAIL' if verdict == 'False' else 'UNCERTAIN'
        )
        claim_obj.factuality_confidence = 0.8  # or model-driven heuristic
        if verdict == 'False':
            claim_obj.factuality_reasoning = out.get('truth', '')
        return out

    def process_claim(self, image, claim_obj, route):
        if route.method == VerificationMethod.CROSS_MODAL:
            return self.verify_claim_cross_modal(image, claim_obj)
        # ... handle other methods ...
```

If using your `CheckRouter`, it will typically choose `VerificationMethod.CROSS_MODAL` for
`ClaimCategoryType.VISUAL_GROUNDING_REQUIRED` claims. You can instantiate `AGLAVerifier` once at
agent/service startup and reuse it across requests.

## Notes & Best Practices
- First call will load models (can take time). Keep an instance cached for reuse.
- If on CPU, the module will avoid half precision.
- If LAVIS is not installed, augmentation is skipped automatically.
- For the `llava` package, set `LLAVA_REPO_PATH` to the repo root (parent directory that contains `llava/`).
- Use smaller LLaVA models in constrained environments, or set `LLAVA_MODEL` accordingly.

## Troubleshooting
- `ModuleNotFoundError: llava`: Ensure `LLAVA_REPO_PATH` points to the repo root or that `llava` is installed.
- CUDA OOM: Reduce batch sizes (only single inference here), lower resolution, or use CPU.
- Slow inference on CPU: Prefer GPU where possible. Consider quantization if supported by your stack.
- Augmentation disabled warning: Safe to ignore if you do not require augmentation.

