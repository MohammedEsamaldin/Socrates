"""
ZANOBIA MitM wrapper for VLMEvalKit

A model-agnostic man-in-the-middle wrapper that transparently mediates between
VLMEvalKit benchmark runners and an arbitrary base VLM. It performs:
- Pre-model input hygiene (claim extraction, routing, verification, minimal edits)
- Post-model output hygiene (same pipeline, minimal edits)
- Protocol compliance enforcement for MCQ-like datasets
- Per-sample session memory (initialized and discarded per generate call)
- Diagnostics and latency logging

Usage:
    from vlmeval.vlm import LLaVA
    from vlmeval.vlm.socrates_mitm import ZanobiaMitM

    base = LLaVA()  # or any other BaseModel subclass
    model = ZanobiaMitM(base_model=base)
    out = model.generate([
        dict(type='image', value='path/to/image.jpg'),
        dict(type='text', value='What is in the image?')
    ], dataset='POPE')
    print(out)

Notes:
- This wrapper is intentionally conservative. For MCQ tasks, it enforces output protocol.
- Ablations can be controlled via init args or environment variables.
- Cross-modal verification leverages existing AGLA/cross-alignment logic, if configured.
"""
from __future__ import annotations

import logging
import json
import os
import re
import time
from typing import List, Optional, Tuple, Dict, Any, Union, Type

from .base import BaseModel
from ..smp import listinstr

# Import the existing MitM middleware
from socrates_system.middleware.mitm_guard import HallucinationMitM

logger = logging.getLogger(__name__)


class VLMEvalBaseModelAdapter:
    """Adapter to allow MitM to call a VLMEvalKit BaseModel via text/image interface."""

    def __init__(self, base_model: BaseModel):
        self.base_model = base_model

    def generate(self, text: str, image_path: Optional[str] = None, **kwargs) -> str:
        message: List[dict] = []
        if image_path:
            message.append(dict(type='image', value=image_path))
        message.append(dict(type='text', value=text))
        dataset = kwargs.get('dataset')
        return self.base_model.generate(message, dataset=dataset)


class ZanobiaMitM(BaseModel):
    """VLMEvalKit-compatible wrapper that applies ZANOBIA MitM around a base model.

    Args:
        base_model: The underlying VLMEvalKit BaseModel to wrap.
        enable_pre: Enable pre-model (input) processing. Default: True.
        enable_post: Enable post-model (output) processing. Default: True.
        enforce_protocol: Enforce dataset-specific output protocol (e.g., MCQ). Default: True.
    """

    INTERLEAVE = False  # use BaseModel utilities for prompt/image parsing

    def __init__(
        self,
        base_model: Optional[BaseModel] = None,
        enable_pre: Optional[bool] = None,
        enable_post: Optional[bool] = None,
        enforce_protocol: Optional[bool] = None,
        # Route-level ablations and modes
        enable_external: Optional[bool] = None,
        enable_cross_modal: Optional[bool] = None,
        enable_self_contradiction: Optional[bool] = None,
        clarification_only: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if base_model is None:
            logger.warning(
                "ZanobiaMitM instantiated without a base_model. It will act as a no-op pass-through.")
        self.base_model = base_model

        # Allow env overrides
        self.enable_pre = (
            (str(os.getenv('SOC_ENABLE_PRE', '')).lower() in ['1', 'true', 'yes'])
            if enable_pre is None else enable_pre
        ) if enable_pre is not False else False
        if enable_pre is True:
            self.enable_pre = True
        elif enable_pre is None and os.getenv('SOC_ENABLE_PRE') is None:
            self.enable_pre = True

        self.enable_post = (
            (str(os.getenv('SOC_ENABLE_POST', '')).lower() in ['1', 'true', 'yes'])
            if enable_post is None else enable_post
        ) if enable_post is not False else False
        if enable_post is True:
            self.enable_post = True
        elif enable_post is None and os.getenv('SOC_ENABLE_POST') is None:
            self.enable_post = True

        self.enforce_protocol = (
            (str(os.getenv('SOC_ENFORCE_PROTOCOL', '')).lower() in ['1', 'true', 'yes'])
            if enforce_protocol is None else enforce_protocol
        ) if enforce_protocol is not False else False
        if enforce_protocol is True:
            self.enforce_protocol = True
        elif enforce_protocol is None and os.getenv('SOC_ENFORCE_PROTOCOL') is None:
            self.enforce_protocol = True

        # Ablation flags: default True unless explicitly disabled
        def _env_bool(name: str, default: Optional[bool]) -> bool:
            if default is True:
                return True
            if default is False:
                return False
            val = os.getenv(name)
            if val is None:
                return True
            return str(val).lower() in ['1', 'true', 'yes']

        self.enable_external = _env_bool('SOC_ENABLE_EXTERNAL', enable_external)
        self.enable_cross_modal = _env_bool('SOC_ENABLE_CROSS_MODAL', enable_cross_modal)
        self.enable_self_contradiction = _env_bool('SOC_ENABLE_SELF_CONTRAD', enable_self_contradiction)
        # Clarification-only: default False unless enabled
        def _env_bool_default_false(name: str, default: Optional[bool]) -> bool:
            if default is True:
                return True
            if default is False:
                return False
            val = os.getenv(name)
            if val is None:
                return False
            return str(val).lower() in ['1', 'true', 'yes']

        self.clarification_only = _env_bool_default_false('SOC_CLARIFICATION_ONLY', clarification_only)

        self.last_diag = {}

    # -------------- Core overrides --------------
    def generate_inner(self, message, dataset=None):
        # Aggregate prompt and first image using BaseModel utility
        prompt, image = self.message_to_promptimg(message, dataset=dataset)

        # Initialize a fresh MitM per sample to keep session memory scoped
        mitm = HallucinationMitM(
            main_model=VLMEvalBaseModelAdapter(self.base_model),
            enable_external=self.enable_external,
            enable_cross_modal=self.enable_cross_modal,
            enable_self_contradiction=self.enable_self_contradiction,
            clarification_only=self.clarification_only,
        )

        t0 = time.time()
        pre_corrections = []
        pre_claims: List[str] = []
        pre_verdicts: List[Dict[str, Any]] = []
        if self.enable_pre and prompt:
            try:
                corrected_input, pre_corrections = mitm._process_text(text=prompt, image_path=image)
                # snapshot diagnostics before post overwrites
                pre_claims = list(getattr(mitm, 'last_claim_texts', []) or [])
                pre_verdicts = list(getattr(mitm, 'last_verdicts', []) or [])
            except Exception as e:
                logger.warning(f"ZanobiaMitM pre-process failed, falling back to original prompt: {e}")
                corrected_input = prompt
        else:
            corrected_input = prompt
        t1 = time.time()

        # Build corrected message for the base model
        corrected_message: List[dict] = []
        if image:
            corrected_message.append(dict(type='image', value=image))
        if corrected_input:
            corrected_message.append(dict(type='text', value=corrected_input))

        # Call the base model
        raw_output = self.base_model.generate(corrected_message, dataset=dataset) if self.base_model else prompt
        t2 = time.time()

        # Post-process
        post_corrections = []
        post_claims: List[str] = []
        post_verdicts: List[Dict[str, Any]] = []
        if self.enable_post and raw_output:
            try:
                corrected_output, post_corrections = mitm._process_text(text=raw_output, image_path=image)
                post_claims = list(getattr(mitm, 'last_claim_texts', []) or [])
                post_verdicts = list(getattr(mitm, 'last_verdicts', []) or [])
            except Exception as e:
                logger.warning(f"ZanobiaMitM post-process failed, falling back to raw output: {e}")
                corrected_output = raw_output
        else:
            corrected_output = raw_output
        t3 = time.time()

        final_output = self._maybe_enforce_protocol(corrected_output, dataset)

        # Record diagnostics
        self.last_diag = {
            'dataset': dataset,
            'latency_ms': {
                'pre': int((t1 - t0) * 1000),
                'base': int((t2 - t1) * 1000),
                'post': int((t3 - t2) * 1000),
                'total': int((t3 - t0) * 1000),
            },
            'num_pre_corrections': len(pre_corrections),
            'num_post_corrections': len(post_corrections),
            'session_id': getattr(mitm, 'session_id', None),
            'modes': {
                'enable_pre': self.enable_pre,
                'enable_post': self.enable_post,
                'enforce_protocol': self.enforce_protocol,
                'enable_external': self.enable_external,
                'enable_cross_modal': self.enable_cross_modal,
                'enable_self_contradiction': self.enable_self_contradiction,
                'clarification_only': self.clarification_only,
            },
            'pre': {
                'claims': pre_claims,
                'verdicts': pre_verdicts,
            },
            'post': {
                'claims': post_claims,
                'verdicts': post_verdicts,
            },
        }
        logger.info(f"ZANOBIA diag: {self.last_diag}")
        # Optional: append diagnostics as JSONL per-sample
        diag_file = os.getenv('SOC_DIAG_FILE')
        if diag_file:
            try:
                with open(diag_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(self.last_diag, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed writing SOC_DIAG_FILE {diag_file}: {e}")

        return final_output

    # -------------- Helpers --------------
    def _maybe_enforce_protocol(self, text: str, dataset: Optional[str]) -> str:
        if not self.enforce_protocol or not dataset or not isinstance(text, str):
            return text
        try:
            # Lazy import to avoid heavy deps during import time
            try:
                from ..dataset import DATASET_TYPE as _DT
                ds_type = _DT(dataset)
            except Exception:
                ds_type = ""
            if listinstr(['MCQ', 'multi-choice'], ds_type):
                # Enforce returning a single option letter A-D
                return self._extract_mcq_option(text)
            if ds_type == 'Y/N':
                return self._normalize_yes_no(text)
        except Exception:
            pass
        return text

    @staticmethod
    def _extract_mcq_option(s: str) -> str:
        s = s.strip()
        # Try strong patterns first
        for pat in [
            r"best option\s*[:：]?\s*\(?([A-Da-d])\)?",
            r"option\s*[:：]?\s*\(?([A-Da-d])\)?",
            r"^\(?([A-Da-d])\)?\.?$",
        ]:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
        # Fallback: find the first standalone A-D token
        m = re.search(r"\b([A-Da-d])\b", s)
        if m:
            return m.group(1).upper()
        return s

    @staticmethod
    def _normalize_yes_no(s: str) -> str:
        s_clean = s.strip().lower()
        # Strong anchors
        patterns = [
            r"^answer\s*[:：]?\s*(yes|no)\b",
            r"^(yes|no)\b",
        ]
        for pat in patterns:
            m = re.search(pat, s_clean, flags=re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        # Common synonyms mapping
        synonyms_yes = {"yes", "y", "yeah", "yep", "true", "correct"}
        synonyms_no = {"no", "n", "nope", "nah", "false", "incorrect"}
        tokens = re.findall(r"[a-zA-Z]+", s_clean)
        for tok in tokens[:3]:  # inspect a few leading tokens
            if tok in synonyms_yes:
                return "Yes"
            if tok in synonyms_no:
                return "No"
        return s


# -------------- Convenience factory --------------
def from_name(
    base: Union[str, Type[BaseModel], BaseModel],
    base_kwargs: Optional[Dict[str, Any]] = None,
    **wrapper_kwargs: Any,
) -> ZanobiaMitM:
    """Instantiate a base model by name/class/instance and wrap with ZanobiaMitM.

    Args:
        base: The base model identifier. Can be:
              - str: class name exposed under vlmeval.vlm (case-insensitive)
              - Type[BaseModel]: a class to be instantiated
              - BaseModel: an already-instantiated model
        base_kwargs: kwargs to construct the base model if needed
        **wrapper_kwargs: passed to ZanobiaMitM constructor (e.g., toggles)

    Returns:
        ZanobiaMitM instance wrapping the resolved base model.
    """
    import importlib
    import inspect

    base_kwargs = base_kwargs or {}

    # Already an instance
    if isinstance(base, BaseModel):
        return ZanobiaMitM(base_model=base, **wrapper_kwargs)

    # A class type
    if isinstance(base, type) and issubclass(base, BaseModel):
        return ZanobiaMitM(base_model=base(**base_kwargs), **wrapper_kwargs)

    # A string: resolve from vlmeval.vlm namespace
    if isinstance(base, str):
        try:
            vlm_pkg = importlib.import_module('vlmeval.vlm')
        except Exception as e:
            raise RuntimeError(f"Failed to import vlmeval.vlm: {e}")

        # exact match first
        cand = getattr(vlm_pkg, base, None)
        if cand is None:
            # case-insensitive search among BaseModel subclasses
            target = base.lower()
            for name in dir(vlm_pkg):
                obj = getattr(vlm_pkg, name)
                if inspect.isclass(obj) and issubclass(obj, BaseModel):
                    if name.lower() == target:
                        cand = obj
                        break
        if cand is None:
            raise ValueError(f"Base model '{base}' not found under vlmeval.vlm")
        if not inspect.isclass(cand) or not issubclass(cand, BaseModel):
            raise TypeError(f"Resolved '{base}' is not a BaseModel subclass: {cand}")
        return ZanobiaMitM(base_model=cand(**base_kwargs), **wrapper_kwargs)

    raise TypeError(f"Unsupported base identifier: {type(base)}")
