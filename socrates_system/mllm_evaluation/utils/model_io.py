from typing import Optional, Union
from ..utils.io import ensure_dir
from socrates_system.modules.llm_manager import LLMManager, LLMProvider  # type: ignore


def build_llm_manager(provider: Optional[Union[str, LLMProvider]] = None,
                      model_name: Optional[str] = None) -> LLMManager:
    """
    Construct an LLMManager for generation. If provider/model are None, env/defaults apply.
    """
    return LLMManager(model_name=model_name, provider=provider)
