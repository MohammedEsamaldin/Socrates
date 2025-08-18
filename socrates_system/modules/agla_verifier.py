"""
AGLA Verifier Module

A reusable module that verifies image-text claims using the AGLA method:
- Stage A: TF-constrained generation for True/False decision
- Stage B: Open correction generation when False

Integration notes:
- This module lazily loads heavy dependencies on first use.
- It needs a LLaVA implementation that supports contrastive decoding with images_cd
  (as provided in your AGLA repo).
- Set environment variable LLAVA_REPO_PATH to point to the REPO ROOT that contains
  the `llava` folder (e.g., /Users/mohammed/Desktop/Thesis/AGLA-1), or ensure a
  compatible `llava` package is installed in your environment.

Optional environment variables:
- LLAVA_MODEL: overrides the default model id (default: liuhaotian/llava-v1.5-7b)
- LLAVA_REPO_PATH: path to the folder containing a compatible `llava` package

Example usage:
    from modules.agla_verifier import AGLAVerifier
    checker = AGLAVerifier()
    result = checker.verify_claim(image_path, "There is a yellow car.")
    print(result)
"""
from __future__ import annotations

import io
import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image
import torch

try:
    # Prefer Socrates logger if available
    from ..utils.logger import setup_logger  # type: ignore
    logger = setup_logger(__name__)
except Exception:  # pragma: no cover - fallback to std logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AGLAResult:
    verdict: str  # "True", "False", or "Uncertain"
    truth: str    # correction / restated truth when verdict == "False" else ""
    latency_ms: int
    debug: Optional[Dict[str, Any]] = None


class AGLAVerifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        llava_repo_path: Optional[str] = None,
        default_use_agla: bool = True,
        default_alpha: float = 2.0,
        default_beta: float = 0.5,
    ) -> None:
        self.device = device or DEVICE_DEFAULT
        self.model_path = model_path or os.environ.get("LLAVA_MODEL", "liuhaotian/llava-v1.5-7b")
        self.llava_repo_path = llava_repo_path or os.environ.get("LLAVA_REPO_PATH")

        self.default_use_agla = default_use_agla
        self.default_alpha = default_alpha
        self.default_beta = default_beta

        # Lazy-loaded handles
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.model_itm = None
        self.vis_processors = None
        self.text_processors = None
        self._to_tensor = None

    # --------------------------
    # Public API
    # --------------------------
    def load_models(self) -> None:
        """Eagerly load heavy models to prepare for serving.
        Safe to call multiple times; only loads once.
        """
        if not self._loaded:
            self._load_all_models()

    def verify_claim(
        self,
        image: Union[str, bytes, Image.Image],
        claim: str,
        use_agla: Optional[bool] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """Verify a textual claim against an image using AGLA.

        Args:
            image: Path to image, raw bytes, or a PIL.Image.Image
            claim: Textual claim
            use_agla: Whether to use AGLA augmentation (default True)
            alpha: Contrastive decoding alpha (default 2.0)
            beta: Contrastive decoding beta  (default 0.5)
            return_debug: include prompts and intermediate paths

        Returns:
            Dict[str, Any] with keys {verdict, truth, latency_ms, debug?}
        """
        t0 = time.time()
        if not self._loaded:
            self._load_all_models()

        use_agla = self.default_use_agla if use_agla is None else use_agla
        alpha = self.default_alpha if alpha is None else alpha
        beta = self.default_beta if beta is None else beta

        img = self._coerce_image(image)
        raw_pixel_values = self.image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(self.device)
        if str(self.device).startswith("cuda"):
            raw_pixel_values = raw_pixel_values.half()

        aug_img = None
        aug_pixel_values = None
        aug_path = None
        if use_agla:
            try:
                aug_img, aug_pixel_values = self._build_augmented_image(img, claim)
                if aug_img is not None:
                    aug_path = os.path.join("/tmp", f"agla_aug_{int(time.time())}.jpg")
                    try:
                        aug_img.save(aug_path)
                    except Exception:
                        aug_path = None
            except Exception as e:
                logger.warning(f"AGLA augmentation failed; proceeding without augmentation. Error: {e}")

        # Stage A (True/False constrained)
        truth_q = self._make_truth_q_from_claim(claim)
        verdict, stage_a_prompt, tf_text = self._llava_generate_tf_constrained(
            raw_pixel_values, aug_pixel_values, truth_q, alpha, beta
        )

        # Stage B (open correction)
        correction = ""
        stage_b_prompt = ""
        if verdict == "False":
            stage_b_prompt = (
                f'The claim "{claim}" is false. In one short sentence, restate the correct fact as precisely as possible.'
            )
            correction = self._llava_generate_open(
                raw_pixel_values, aug_pixel_values, stage_b_prompt, alpha, beta,
                min_new_tokens=8, max_new_tokens=48, disable_eos=True
            )

        payload: Dict[str, Any] = {
            "verdict": verdict,
            "truth": correction,
            "latency_ms": int((time.time() - t0) * 1000),
        }
        if return_debug:
            payload["debug"] = {
                "stage_a_prompt": stage_a_prompt,
                "stage_a_raw": tf_text,
                "stage_b_prompt": stage_b_prompt,
                "augmented_image_path": aug_path,
                "device": self.device,
            }
        return payload

    # --------------------------
    # Loading and setup
    # --------------------------
    def _ensure_llava_on_path(self) -> None:
        """Ensure Python can import `llava`.
        Accepts LLAVA_REPO_PATH set to either the repo root (containing the `llava/` folder)
        or the `llava/` folder itself. We normalize to the repo root on sys.path.
        """
        if not self.llava_repo_path:
            return
        path = os.path.normpath(self.llava_repo_path)
        if not os.path.isdir(path):
            logger.warning(f"LLAVA_REPO_PATH does not exist or is not a directory: {path}")
            return
        base = os.path.basename(path)
        repo_root = os.path.dirname(path) if base == "llava" else path
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            logger.info(f"Added to sys.path for llava import: {repo_root}")

    def _load_all_models(self) -> None:
        self._ensure_llava_on_path()
        try:
            # llava imports (must be available via repo path or pip)
            from llava.constants import (
                IMAGE_TOKEN_INDEX,
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
            )
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import tokenizer_image_token
            from llava.utils import disable_torch_init
        except Exception as e:
            hint = (
                "Could not import a compatible 'llava' package. Set LLAVA_REPO_PATH to a repo folder that "
                "contains the required llava implementation (e.g., your AGLA repo's llava), or install a "
                "compatible version in your environment."
            )
            raise RuntimeError(f"Failed to import 'llava': {e}. {hint}")

        # Store references used later
        self._llava_IMPORTED = {
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
            "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
            "conv_templates": conv_templates,
            "SeparatorStyle": SeparatorStyle,
            "load_pretrained_model": load_pretrained_model,
            "tokenizer_image_token": tokenizer_image_token,
            "disable_torch_init": disable_torch_init,
        }

        # Optional: patch Transformers generation with contrastive decoding support
        self._evolve_agla_sampling()

        # Build LLaVA model
        self._llava_IMPORTED["disable_torch_init"]()
        tokenizer, model, image_processor, _ = self._llava_IMPORTED["load_pretrained_model"](
            self.model_path, None, self.model_path.split("/")[-1], load_8bit=False, load_4bit=False
        )

        # BLIP-ITM for IPM (from LAVIS) - optional
        model_itm = None
        vis_processors = None
        text_processors = None
        to_tensor = None
        try:
            from lavis.models import load_model_and_preprocess  # Lazy import
            from torchvision import transforms  # Lazy import

            model_itm, vis_processors, text_processors = load_model_and_preprocess(
                "blip_image_text_matching", "large", device=self.device, is_eval=True
            )
            to_tensor = transforms.Compose([transforms.ToTensor()])
            logger.info("Loaded LAVIS BLIP-ITM components for AGLA augmentation.")
        except Exception as e:
            logger.warning(
                "LAVIS BLIP-ITM not available; AGLA augmentation will be disabled. "
                f"Error: {e}"
            )

        # Commit
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.model_itm = model_itm
        self.vis_processors = vis_processors
        self.text_processors = text_processors
        self._to_tensor = to_tensor
        self._loaded = True
        logger.info("AGLA models loaded successfully.")

    # --------------------------
    # Helpers: text / prompt
    # --------------------------
    def _build_prompt(self, question: str, conv_mode: str = "llava_v1", one_word: bool = False) -> Tuple[str, str]:
        cfg = getattr(self.model, "config", None)
        if getattr(cfg, "mm_use_im_start_end", False):
            qs = (
                self._llava_IMPORTED["DEFAULT_IM_START_TOKEN"]
                + self._llava_IMPORTED["DEFAULT_IMAGE_TOKEN"]
                + self._llava_IMPORTED["DEFAULT_IM_END_TOKEN"]
                + "\n"
                + question
            )
        else:
            qs = self._llava_IMPORTED["DEFAULT_IMAGE_TOKEN"] + "\n" + question
        conv = self._llava_IMPORTED["conv_templates"][conv_mode].copy()
        if one_word:
            conv.append_message(conv.roles[0], qs + " Please answer this question with one word, either True or False.")
        else:
            conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        sep = conv.sep if conv.sep_style != self._llava_IMPORTED["SeparatorStyle"].TWO else conv.sep2
        return conv.get_prompt(), sep

    @staticmethod
    def _canonicalize_claim(c: str) -> str:
        c = c.strip().rstrip("?.!").strip().replace('"', "'")
        lc = c.lower()
        if lc.startswith("are there "):
            return "There are " + c[10:] + "."
        if lc.startswith("is there "):
            return "There is " + c[9:] + "."
        return c + "."

    def _make_truth_q_from_claim(self, claim: str) -> str:
        prop = self._canonicalize_claim(claim)
        return f'Is it true that "{prop}"? Answer only "True" or "False".'

    # --------------------------
    # Helpers: augmentation (AGLA local view)
    # --------------------------
    @staticmethod
    def _augmentation(image, question, tensor_image, model, tokenized_text, raw_image):
        """Ported from eval/augmentation.py (kept minimal)."""
        import numpy as np
        from lavis.common.gradcam import getAttMap
        from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
        from torchvision import transforms

        with torch.set_grad_enabled(True):
            gradcams, _ = compute_gradcam(
                model=model,
                visual_input=image,
                text_input=question,
                tokenized_text=tokenized_text,
                block_num=6,
            )
        gradcams = [gradcam_[1] for gradcam_ in gradcams]
        gradcams1 = torch.stack(gradcams).reshape(image.size(0), -1)
        itc_score = model({"image": image, "text_input": question}, match_head='itc')
        ratio = 1 - itc_score / 2
        ratio = min(ratio, 1 - 10 ** (-5))
        resized_img = raw_image.resize((384, 384))
        norm_img = np.float32(resized_img) / 255
        gradcam = gradcams1.reshape(24, 24)

        avg_gradcam = getAttMap(norm_img, gradcam.cpu().numpy(), blur=True, overlap=False)
        temp, _ = torch.sort(torch.tensor(avg_gradcam).reshape(-1), descending=True)
        cam1 = torch.tensor(avg_gradcam).unsqueeze(2)
        cam = torch.cat([cam1, cam1, cam1], dim=2)

        mask = torch.where(cam < temp[int(384 * 384 * ratio)], 0, 1)
        new_image = tensor_image.permute(1, 2, 0) * mask
        unloader = transforms.ToPILImage()
        imag = new_image.clone().permute(2, 0, 1)
        imag = unloader(imag)
        return imag

    def _build_augmented_image(self, img: Image.Image, claim: str) -> Tuple[Optional[Image.Image], Optional[torch.Tensor]]:
        if self.model_itm is None:
            return None, None
        q_text_for_ipm = self.text_processors["eval"](claim)
        tokenized_text = self.model_itm.tokenizer(q_text_for_ipm, padding="longest", truncation=True, return_tensors="pt").to(self.device)
        vis_image = self.vis_processors["eval"](img).unsqueeze(0).to(self.device)
        tensor_image_384 = self._to_tensor(img.resize((384, 384)))
        aug_img = self._augmentation(vis_image, q_text_for_ipm, tensor_image_384, self.model_itm, tokenized_text, img)
        aug_pixel_values = self.image_processor.preprocess(aug_img, return_tensors="pt")["pixel_values"].to(self.device)
        if str(self.device).startswith("cuda"):
            aug_pixel_values = aug_pixel_values.half()
        return aug_img, aug_pixel_values

    # --------------------------
    # Helpers: generation
    # --------------------------
    def _llava_generate_tf_constrained(
        self,
        raw_pixel_values: torch.Tensor,
        aug_pixel_values: Optional[torch.Tensor],
        truth_q: str,
        alpha: float,
        beta: float,
    ) -> Tuple[str, str, str]:
        prompt, stop_str = self._build_prompt(truth_q, conv_mode="llava_v1", one_word=True)
        prompt_ids = self._llava_IMPORTED["tokenizer_image_token"](
            prompt, self.tokenizer, self._llava_IMPORTED["IMAGE_TOKEN_INDEX"], return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        # FULL token sequences for " True"/" False"
        true_seq = self.tokenizer.encode(" True", add_special_tokens=False)
        false_seq = self.tokenizer.encode(" False", add_special_tokens=False)
        prompt_len = prompt_ids.shape[1]

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            gen = input_ids[0, prompt_len:] if input_ids.dim() == 2 else input_ids[prompt_len:]
            gen = gen.tolist()
            allowed = set()
            if gen == true_seq[: len(gen)]:
                allowed.add(true_seq[len(gen)] if len(gen) < len(true_seq) else self.tokenizer.eos_token_id)
            if gen == false_seq[: len(gen)]:
                allowed.add(false_seq[len(gen)] if len(gen) < len(false_seq) else self.tokenizer.eos_token_id)
            if not allowed:
                allowed.update([true_seq[0], false_seq[0]])
            return list(allowed)

        out = self.model.generate(
            input_ids=prompt_ids,
            images=raw_pixel_values,
            images_cd=aug_pixel_values,
            cd_alpha=alpha,
            cd_beta=beta,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=None,
            max_new_tokens=max(len(true_seq), len(false_seq)),
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        gen_ids = out[0, prompt_len:].tolist()
        if gen_ids[: len(true_seq)] == true_seq:
            tf_text = "True"
        elif gen_ids[: len(false_seq)] == false_seq:
            tf_text = "False"
        else:
            tf_text = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()
        verdict = (
            "True"
            if tf_text.lower().startswith("true")
            else "False"
            if tf_text.lower().startswith("false")
            else "Uncertain"
        )
        return verdict, prompt, tf_text

    def _llava_generate_open(
        self,
        raw_pixel_values: torch.Tensor,
        aug_pixel_values: Optional[torch.Tensor],
        prompt_text: str,
        alpha: float,
        beta: float,
        min_new_tokens: int = 8,
        max_new_tokens: int = 48,
        disable_eos: bool = True,
    ) -> str:
        prompt, stop_str = self._build_prompt(prompt_text, conv_mode="llava_v1", one_word=False)
        prompt_ids = self._llava_IMPORTED["tokenizer_image_token"](
            prompt, self.tokenizer, self._llava_IMPORTED["IMAGE_TOKEN_INDEX"], return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        kwargs = dict(
            input_ids=prompt_ids,
            images=raw_pixel_values,
            images_cd=aug_pixel_values,
            cd_alpha=alpha,
            cd_beta=beta,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            top_k=None,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if disable_eos:
            kwargs["eos_token_id"] = None
        out = self.model.generate(**kwargs)
        text = self.tokenizer.batch_decode(out[:, prompt_ids.shape[1] :], skip_special_tokens=True)[0].strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        return text

    # --------------------------
    # Transformers monkey patch for CD (as in eval/sample.py)
    # --------------------------
    def _evolve_agla_sampling(self) -> None:
        try:
            import transformers
            from transformers.generation.logits_process import LogitsProcessorList
            from transformers.generation.utils import SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
            import torch.distributed as dist  # type: ignore
            from torch import nn  # type: ignore
            from typing import List, Optional, Union
            from transformers.generation.stopping_criteria import (
                StoppingCriteria,
                StoppingCriteriaList,
                validate_stopping_criteria,
            )

            def sample(
                self,
                input_ids: torch.LongTensor,
                logits_processor: Optional[LogitsProcessorList] = None,
                stopping_criteria: Optional[StoppingCriteriaList] = None,
                logits_warper: Optional[LogitsProcessorList] = None,
                max_length: Optional[int] = None,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[Union[int, List[int]]] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                output_scores: Optional[bool] = None,
                return_dict_in_generate: Optional[bool] = None,
                synced_gpus: bool = False,
                streamer: Optional["BaseStreamer"] = None,
                **model_kwargs,
            ) -> Union[SampleOutput, torch.LongTensor]:
                # init values
                logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
                stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
                if max_length is not None:
                    warnings.warn(
                        "`max_length` is deprecated in this function, use"
                        " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                        UserWarning,
                    )
                    stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
                logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
                pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
                eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

                if isinstance(eos_token_id, int):
                    eos_token_id = [eos_token_id]
                eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
                output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
                output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
                output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states

                return_dict_in_generate = (
                    return_dict_in_generate
                    if return_dict_in_generate is not None
                    else self.generation_config.return_dict_in_generate
                )

                # init attention / hidden states / scores tuples
                scores = () if (return_dict_in_generate and output_scores) else None
                decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
                cross_attentions = () if (return_dict_in_generate and output_attentions) else None
                decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

                # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
                if return_dict_in_generate and self.config.is_encoder_decoder:
                    encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                    encoder_hidden_states = (
                        model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                    )

                # keep track of which sequences are already finished
                unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

                this_peer_finished = False  # used by synced_gpus only

                # auto-regressive generation
                while True:
                    if synced_gpus:
                        this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                        dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                        if this_peer_finished_flag.item() == 0.0:
                            break

                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    outputs = self(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )

                    if synced_gpus and this_peer_finished:
                        continue

                    next_token_logits = outputs.logits[:, -1, :]

                    use_cd = model_kwargs.get("images_cd") is not None
                    output_attentions_wo_img = (
                        output_attentions if output_attentions is not None else self.generation_config.output_attentions
                    )
                    output_hidden_states_wo_img = (
                        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
                    )
                    model_kwargs_cd = model_kwargs.copy()

                    if use_cd:
                        model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
                        outputs_cd = self(
                            **model_inputs_cd,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                        )
                        next_token_logits_cd = outputs_cd.logits[:, -1, :]
                        cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1
                        cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.5
                        cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                        diffs = (next_token_logits + cd_alpha * next_token_logits_cd)
                        cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
                        cd_logits = logits_processor(input_ids, cd_logits)
                        cd_logits = logits_warper(input_ids, cd_logits)
                        cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                        next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
                    else:
                        next_token_scores = logits_processor(input_ids, next_token_logits)
                        next_token_scores = logits_warper(input_ids, next_token_scores)
                        probs = nn.functional.softmax(next_token_scores, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                    if return_dict_in_generate:
                        if output_scores:
                            scores += (next_token_scores,)
                        if output_attentions:
                            decoder_attentions += (
                                (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                            )
                            if self.config.is_encoder_decoder:
                                cross_attentions += (outputs.cross_attentions,)
                        if output_hidden_states:
                            decoder_hidden_states += (
                                (outputs.decoder_hidden_states,)
                                if self.config.is_encoder_decoder
                                else (outputs.hidden_states,)
                            )

                    if eos_token_id is not None:
                        if pad_token_id is None:
                            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    if streamer is not None:
                        streamer.put(next_tokens.cpu())
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                    if use_cd:
                        model_kwargs_cd = self._update_model_kwargs_for_generation(
                            outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
                        )

                    if eos_token_id_tensor is not None:
                        unfinished_sequences = unfinished_sequences.mul(
                            next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                            .ne(eos_token_id_tensor.unsqueeze(1))
                            .prod(dim=0)
                        )
                        if unfinished_sequences.max() == 0:
                            this_peer_finished = True

                    if stopping_criteria(input_ids, scores):
                        this_peer_finished = True

                    if this_peer_finished and not synced_gpus:
                        break

                if streamer is not None:
                    streamer.end()

                if return_dict_in_generate:
                    if self.config.is_encoder_decoder:
                        return SampleEncoderDecoderOutput(
                            sequences=input_ids,
                            scores=scores,
                            encoder_attentions=encoder_attentions,
                            encoder_hidden_states=encoder_hidden_states,
                            decoder_attentions=decoder_attentions,
                            cross_attentions=cross_attentions,
                            decoder_hidden_states=decoder_hidden_states,
                        )
                    else:
                        return SampleDecoderOnlyOutput(
                            sequences=input_ids,
                            scores=scores,
                            attentions=decoder_attentions,
                            hidden_states=decoder_hidden_states,
                        )
                else:
                    return input_ids

            import warnings
            from transformers.generation.streamers import BaseStreamer  # type: ignore
            transformers.generation.utils.GenerationMixin.sample = sample  # type: ignore
            logger.info("Patched Transformers GenerationMixin.sample with AGLA contrastive decoding support.")
        except Exception as e:
            logger.warning(f"Could not patch Transformers sampling for AGLA CD: {e}")

    # --------------------------
    # Utilities
    # --------------------------
    @staticmethod
    def _coerce_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, str):
            with open(image, "rb") as f:
                data = f.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        raise TypeError("Unsupported image type. Provide a file path, raw bytes, or a PIL.Image.Image.")
