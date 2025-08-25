from typing import Optional, Union, Any, List
import os
from ..utils.io import ensure_dir
from socrates_system.modules.llm_manager import LLMManager, LLMProvider  # type: ignore


class LLaVAHFManager:
    """Adapter that uses LLaVA-HF only for final answer generation while delegating all
    other LLMManager methods/attributes to an inner text LLMManager.

    This enforces fp16/bf16 (no 4-bit) for LLaVA and supports both 7B and 13B models
    via the provided model_name (e.g., "llava-hf/llava-1.5-7b-hf" or "llava-hf/llava-1.5-13b-hf").
    """

    def __init__(self, llava_model_name: str, inner_manager: LLMManager) -> None:
        self._llava_model_name = llava_model_name
        self._inner = inner_manager

    def __getattr__(self, name: str) -> Any:
        # Delegate all unknown attributes/methods to the inner manager
        return getattr(self._inner, name)

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        system_prompt: str = None,
        images: Optional[List[str]] = None,
    ) -> str:
        # If no images are provided, delegate to the inner text LLM for pipeline tasks
        if not images:
            return self._inner.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                images=images,
            )

        # Use LLaVA-HF for multimodal (image+text) final answer generation in fp16/bf16 (no 4-bit)
        # Lazy import to avoid heavy deps at import time for text-only paths
        from socrates_system.mllm_evaluation.providers.llava_hf import LlavaHFGenerator  # type: ignore

        image_path = images[0] if images else None
        generator = LlavaHFGenerator.get(self._llava_model_name, no_4bit=True, use_slow_tokenizer=False)
        return generator.generate(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )


def build_llm_manager(provider: Optional[Union[str, LLMProvider]] = None,
                      model_name: Optional[str] = None) -> Any:
    """
    Construct an LLMManager for generation. If provider/model are None, env/defaults apply.
    If provider is llava_hf, return an adapter that delegates pipeline tasks to a
    normal LLMManager but uses LLaVA-HF (fp16/bf16) for the final answer generation.
    """
    # Normalize provider string if needed
    prov_str = (provider.value if isinstance(provider, LLMProvider) else provider) if provider else None
    if prov_str and prov_str.lower() == "llava_hf":
        # Choose LLaVA model (support 7B/13B). Default to 13B if not specified.
        llava_model = (
            model_name
            or os.getenv("SOC_LLAVA_MODEL")
            or "llava-hf/llava-1.5-13b-hf"
        )
        # Build inner text LLM for pipeline tasks. Allow override via env.
        inner_provider = os.getenv("SOC_PIPELINE_PROVIDER") or LLMProvider.OLLAMA.value
        inner_model = os.getenv("SOC_PIPELINE_MODEL")  # let LLMManager pick defaults if None
        inner_manager = LLMManager(model_name=inner_model, provider=inner_provider)
        return LLaVAHFManager(llava_model, inner_manager)

    # Fallback: regular manager
    return LLMManager(model_name=model_name, provider=provider)
