import os
import json
import re
from typing import Any, Dict, List, Optional, Iterable

from .utils.checkpointing import CheckpointManager
from .utils.logging_utils import setup_run_logger
from .utils.model_io import build_llm_manager
from .mitm import build_pipeline, process_user_turn, process_model_turn
from .utils.serialization import to_jsonable
from .utils.dataset import load_dataset_generic, get_prompt_text
from socrates_system.modules.llm_manager import LLMProvider, LLMManager


class BaseEvaluator:
    """
    Base evaluator that implements man-in-the-middle execution and checkpointing.

    Subclasses must implement `load_dataset(path)` and `sample_to_prompt(sample)`
    to provide the user-facing prompt text (and optional metadata).
    """

    BENCHMARK_NAME: str = "generic"

    def __init__(
        self,
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
        # New: allow separate providers/models for SUT vs pipeline
        sut_provider: Optional[str] = None,
        sut_model_name: Optional[str] = None,
        pipeline_provider: Optional[str] = None,
        pipeline_model_name: Optional[str] = None,
        # MME option: force Yes/No responses from SUT
        force_yes_no: bool = False,
    ) -> None:
        self.dataset_path = dataset_path
        # If the provided run_dir already contains checkpoints for this run, honor it as-is.
        # Otherwise, place checkpoints under <run_dir>/<BENCHMARK_NAME> (default behavior).
        candidate_dir = run_dir
        use_as_is = False
        try:
            if os.path.isdir(candidate_dir):
                for fname in ("results.jsonl", "state.json", "mmhal_results.jsonl"):
                    if os.path.exists(os.path.join(candidate_dir, fname)):
                        use_as_is = True
                        break
        except Exception:
            use_as_is = False

        self.run_dir = candidate_dir if use_as_is else os.path.join(run_dir, self.BENCHMARK_NAME)
        os.makedirs(self.run_dir, exist_ok=True)

        self.logger = setup_run_logger(self.run_dir, name=self.BENCHMARK_NAME)
        self.ckpt = CheckpointManager(self.run_dir)

        # Build model (system-under-test) and pipeline
        # Backward compatible defaults: if not specified, use `provider`/`model_name` for both.
        _sut_provider = sut_provider or provider
        _sut_model = sut_model_name or model_name
        _pipe_provider = pipeline_provider or provider
        _pipe_model = pipeline_model_name or model_name

        # LLM used for pipeline (claims/clarification/factuality)
        self.pipeline_llm_manager: LLMManager = build_llm_manager(provider=_pipe_provider, model_name=_pipe_model)
        # Alias for backward compat (other modules may reference self.llm_manager)
        self.llm_manager: LLMManager = self.pipeline_llm_manager
        # LLM used for generating the system-under-test answer (can be different, e.g., LLaVA)
        self.sut_llm_manager: LLMManager = build_llm_manager(provider=_sut_provider, model_name=_sut_model)

        # Determine post-factuality clarification policy:
        # If SOC_POST_FACTUALITY_CLAR is set, honor it. Otherwise, disable for MME specifically.
        _pf_env = os.getenv("SOC_POST_FACTUALITY_CLAR")
        if _pf_env is not None:
            _pf_enabled = str(_pf_env).strip().lower() == "true"
        else:
            _pf_enabled = (self.BENCHMARK_NAME.lower() != "mme")

        self.pipeline = build_pipeline(
            llm_manager=self.pipeline_llm_manager,
            factuality_enabled=True,
            clarification_enabled=True,
            clarification_dev_mode=False,
            question_gen_enabled=False,
            post_factuality_clarification_enabled=_pf_enabled,
        )

        self.limit = limit
        self.resume = resume
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.prompt_key = prompt_key
        self.id_key = id_key
        self.fallback_keys = fallback_keys
        self.image_key = image_key
        self.image_root = image_root
        self.force_yes_no = force_yes_no

        # Write meta
        self.ckpt.write_meta({
            "benchmark": self.BENCHMARK_NAME,
            "dataset_path": os.path.abspath(self.dataset_path),
            "sut_provider": getattr(self.sut_llm_manager, "provider").value,
            "sut_model": getattr(self.sut_llm_manager, "model_name"),
            "pipeline_provider": getattr(self.pipeline_llm_manager, "provider").value,
            "pipeline_model": getattr(self.pipeline_llm_manager, "model_name"),
            "max_gen_tokens": self.max_gen_tokens,
            "temperature": self.temperature,
            "image_key": self.image_key,
            "image_root": self.image_root,
            "force_yes_no": self.force_yes_no,
        })

    # ------ to be implemented by subclasses ------
    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        return load_dataset_generic(path)

    def sample_to_prompt(self, sample: Dict[str, Any]) -> str:
        return get_prompt_text(sample, key_override=self.prompt_key, fallback_keys=self.fallback_keys)
    # ------------------------------------------------

    def sample_to_image_path(self, sample: Dict[str, Any]) -> Optional[str]:
        """Hook to extract an image path/URL from a sample. May be overridden by subclasses.

        Default behavior checks a few common keys and resolves relative paths against
        image_root or the dataset directory.
        """
        # Preferred override
        keys: List[Optional[str]] = [self.image_key, "image", "image_path", "image_file", "img", "imageUrl", "imageURL", "image_uri"]
        val: Optional[str] = None
        for k in keys:
            if k and isinstance(sample.get(k), str) and sample[k].strip():
                val = sample[k].strip()
                break
        if not val:
            return None
        # URLs are returned as-is (caller may handle downloading if needed)
        if val.startswith("http://") or val.startswith("https://"):
            return val
        # Resolve relative paths
        if os.path.isabs(val):
            return val
        root = self.image_root or os.path.dirname(os.path.abspath(self.dataset_path))
        candidate = os.path.join(root, val)
        return candidate

    def _claims_to_mmhal(self,
                         claims: List[Any],
                         clar_results: Dict[int, Dict[str, Any]],
                         factuality: Dict[int, Any],
                         stage: str) -> List[Dict[str, Any]]:
        """Convert pipeline claims + clarifications + factuality into MMHal-style items.

        Each item contains: index, text, span, categories, corrected_text (if any),
        factuality (status/confidence/evidence/etc), clarification (stage-specific).
        """
        items: List[Dict[str, Any]] = []
        if not claims:
            return items
        for i, cl in enumerate(claims, 1):
            # categories -> names
            cat_names: List[str] = []
            try:
                for c in getattr(cl, "categories", []) or []:
                    nm = getattr(c, "name", None)
                    if nm is None:
                        continue
                    try:
                        cat_names.append(nm.name)
                    except Exception:
                        cat_names.append(str(nm))
            except Exception:
                pass

            # clarification at stage
            clar = (clar_results or {}).get(i, {})
            clar_stage_obj = clar.get(stage)
            corrected_text = None
            if clar_stage_obj is not None:
                try:
                    ct = getattr(clar_stage_obj, "corrected_claim", None)
                    if isinstance(ct, str) and ct.strip():
                        corrected_text = ct
                except Exception:
                    pass

            item = {
                "index": i,
                "text": getattr(cl, "text", None),
                "span": [int(getattr(cl, "start_char", -1)), int(getattr(cl, "end_char", -1))],
                "categories": cat_names,
                "corrected_text": corrected_text,
                "clarification": to_jsonable(clar_stage_obj) if clar_stage_obj is not None else None,
                "factuality": to_jsonable((factuality or {}).get(i)),
            }
            items.append(item)
        return items

    def _detect_yes_no(self, text: str) -> Optional[bool]:
        """Detect if a response is effectively a yes/no answer.

        Returns True for yes, False for no, None if indeterminate.
        """
        if not text:
            return None
        s = text.strip().lower()
        # Normalize punctuation and commas
        s = re.sub(r"^[^a-zA-Z]*(yes|no)\b.*$", r"\1", s)
        # Single-token quick path
        if s in {"yes", "y", "yeah", "yep", "true", "correct", "affirmative"}:
            return True
        if s in {"no", "n", "nope", "false", "incorrect", "negative"}:
            return False
        # Look at the beginning of the string for a yes/no cue
        m = re.match(r"^(yes|no)[\s,!.]?\b", text.strip().lower())
        if m:
            return True if m.group(1) == "yes" else False
        return None

    def _question_to_declarative(self, question: str, answer_yes: bool) -> str:
        """Convert a yes/no question into a declarative hypothesis using the pipeline LLM.

        Falls back to a conservative heuristic if the LLM call fails or returns
        an unusable result.
        """
        if not question:
            return ""
        # Strip common trailing yes/no instruction phrases for cleaner verification
        q = question.strip()
        try:
            patterns = [
                r"(?:\s*[.?!])?\s*(?:please\s+)?answer(?:\s+strictly)?(?:\s+with)?(?:\s+a\s+single\s+word)?\s*[:\-]?\s*(?:\"?yes\"?\s*(?:\/|or)\s*\"?no\"?|yes\/no|yes\s+or\s+no)\.?$",
                r"(?:\s*[.?!])?\s*respond(?:\s+with)?\s*(?:\"?yes\"?\s*(?:\/|or)\s*\"?no\"?|yes\/no|yes\s+or\s+no)\.?$",
                r"(?:\s*[.?!])?\s*provide\s+(?:a\s+)?(?:yes\/no|yes\s+or\s+no)\s*answer\.?$",
                r"(?:\s*[.?!])?\s*please\s+answer\s+yes\s+or\s+no\.?$",
            ]
            for _pat in patterns:
                q = re.sub(_pat, "", q, flags=re.IGNORECASE).strip()
        except Exception:
            # Best-effort cleanup only
            pass
        q = q.strip().strip("?").strip()

        # Primary path: LLM-based declarative hypothesis
        try:
            ans_str = "Yes" if answer_yes else "No"
            system_prompt = (
                "You convert yes/no questions into a single concise declarative hypothesis for verification. "
                "Given a question and the intended yes/no answer, output exactly one plain sentence that would be true "
                "if and only if that answer is correct. Avoid explanations, prefixes, or quotes. Do not include 'Yes' or 'No'."
            )
            user_prompt = (
                f"Question: {q}\n"
                f"Intended answer: {ans_str}\n"
                "Write the declarative hypothesis:"
            )
            out = self.pipeline_llm_manager.generate_text(
                user_prompt,
                max_tokens=64,
                temperature=0.0,
                system_prompt=system_prompt,
            ) or ""
            hyp = (out or "").strip()
            # Keep only the first line and strip common labels/quotes
            hyp = hyp.splitlines()[0].strip()
            hyp = re.sub(r"^(Hypothesis|Declarative|Statement|Output)\s*[:\-]\s*", "", hyp, flags=re.IGNORECASE).strip()
            hyp = hyp.strip().strip('"').strip("'")
            # Guard against degenerate outputs
            if not hyp or hyp.lower() in {"yes", "no"}:
                raise ValueError("LLM returned non-declarative answer")
            return hyp
        except Exception:
            # Heuristic fallback for robustness
            low = q.lower()
            if low.startswith("is there "):
                core = q[len("is there ") :]
                return f"There is {core}" if answer_yes else f"There is not {core}"
            if low.startswith("are there "):
                core = q[len("are there ") :]
                return f"There are {core}" if answer_yes else f"There are not {core}"
            # Default: wrap as a generic hypothesis
            if answer_yes:
                return q
            return f"It is not the case that {q}"

    def get_sample_id(self, sample: Dict[str, Any]):
        """Hook for subclasses to extract a stable sample identifier."""
        if self.id_key and sample.get(self.id_key) is not None:
            return sample.get(self.id_key)
        return sample.get("id") or sample.get("sample_id") or sample.get("idx")

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = self.get_sample_id(sample)
        prompt = self.sample_to_prompt(sample)

        # 1) User turn on ORIGINAL question (collect clarifications; do NOT apply to Q0)
        image_path = self.sample_to_image_path(sample)
        user_res = process_user_turn(self.pipeline, prompt, image_path=image_path)
        corrected_prompt_candidate = user_res.get("corrected_text") or prompt

        # Optional instruction to enforce strict Yes/No output
        yn_instr = None
        if getattr(self, "force_yes_no", False):
            yn_instr = (
                "Instruction: Answer strictly with a single word: 'Yes' or 'No'. "
                "Do not include any explanation or extra words."
            )

        def _append_yn_instr(text: str) -> str:
            if yn_instr:
                return f"{text}\n\n{yn_instr}"
            return text

        # Helper: analyze a model output with MITM verification and (if yes/no) derived hypothesis
        def _analyze_output(model_text: str, question_text: str) -> Dict[str, Any]:
            # Tag stage for downstream DEBUG logging of cross-modal verification
            try:
                setattr(self.pipeline, "_verification_stage", "Q0")
            except Exception:
                pass
            res = process_model_turn(self.pipeline, model_text, image_path=image_path)
            corrected = res.get("corrected_text") or model_text
            try:
                self.logger.info(f"Sample {sample_id} - Model output: {model_text}")
                if corrected != model_text:
                    self.logger.info(f"Sample {sample_id} - Output corrected via MITM: {corrected}")
            except Exception:
                pass

            yn_det = self._detect_yes_no(model_text)
            analysis_src = "original"
            claims_a = res.get("claims", [])
            clar_a = res.get("clarification", {})
            fact_a = res.get("factuality", {})

            derived = None
            desired: Optional[bool] = None
            if yn_det is not None:
                try:
                    hypo = self._question_to_declarative(question_text, answer_yes=bool(yn_det))
                except Exception:
                    hypo = None
                if hypo:
                    try:
                        self.logger.info(
                            "Deriving hypothesis from yes/no answer for verification"
                        )
                    except Exception:
                        pass
                    # Tag as Q1 for downstream logging; suppress external knowledge/self-consistency checks for hypothesis verification
                    # to prioritize cross-modal-only verification signals.
                    try:
                        setattr(self.pipeline, "_verification_stage", "Q1")
                    except Exception:
                        pass
                    derived = process_model_turn(self.pipeline, hypo, image_path=image_path, hypothesis_mode=True)
                    # If original claims are empty, use derived artifacts for analysis
                    if (not claims_a) and derived is not None:
                        claims_a = derived.get("claims", [])
                        clar_a = derived.get("clarification", {})
                        fact_a = derived.get("factuality", {})
                        analysis_src = "derived_yesno"

            # Build MMHal-style items from chosen analysis artifacts
            mmhal_items = self._claims_to_mmhal(claims_a, clar_a, fact_a, stage="post")
            n_pass = sum(1 for it in mmhal_items if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "PASS")
            n_fail = sum(1 for it in mmhal_items if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "FAIL")
            n_unc = sum(1 for it in mmhal_items if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "UNCERTAIN")
            # Content-relevant filtering (exclude meta/ambiguous categories)
            CONTENT_CATS = {"VISUAL_GROUNDING_REQUIRED", "EXTERNAL_KNOWLEDGE_REQUIRED", "SELF_CONSISTENCY_REQUIRED"}
            EXCLUDED_META_CATS = {"AMBIGUOUS_RESOLUTION_REQUIRED", "SUBJECTIVE_OPINION", "PROCEDURAL_DESCRIPTIVE"}
            def _is_content_item(item: Dict[str, Any]) -> bool:
                cats = item.get("categories") or []
                try:
                    cats = list(cats)
                except Exception:
                    cats = []
                has_content = any(c in CONTENT_CATS for c in cats)
                has_meta = any(c in EXCLUDED_META_CATS for c in cats)
                return bool(has_content and not has_meta)
            n_pass_c = sum(1 for it in mmhal_items if _is_content_item(it) and isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "PASS")
            n_fail_c = sum(1 for it in mmhal_items if _is_content_item(it) and isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "FAIL")
            n_unc_c = sum(1 for it in mmhal_items if _is_content_item(it) and isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "UNCERTAIN")
            # Track strongest contradicting confidence among content-only FAILs
            max_fail_conf_c = 0.0
            try:
                for it in mmhal_items:
                    if not _is_content_item(it):
                        continue
                    f = it.get("factuality") or {}
                    if isinstance(f, dict) and (f.get("status") == "FAIL"):
                        try:
                            cf = float(f.get("confidence", 0.0) or 0.0)
                        except Exception:
                            cf = 0.0
                        if cf > max_fail_conf_c:
                            max_fail_conf_c = cf
            except Exception:
                max_fail_conf_c = 0.0

            # If we derived a hypothesis, compute directional signal for yes/no
            yn_sig = None
            yn_sig_content = None
            yn_sig_content_dir = "undecided"
            has_content_positive_pass = False
            has_any_content_fail = False
            if yn_det is not None and derived is not None:
                yn_items = self._claims_to_mmhal(
                    derived.get("claims", []),
                    derived.get("clarification", {}),
                    derived.get("factuality", {}),
                    stage="post",
                )
                y_pass = sum(1 for it in yn_items if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "PASS")
                y_fail = sum(1 for it in yn_items if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "FAIL")
                y_unc = sum(1 for it in yn_items if isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "UNCERTAIN")
                yn_sig = {"pass": y_pass, "fail": y_fail, "uncertain": y_unc}
                # Content-only directional signal
                y_pass_c = sum(1 for it in yn_items if _is_content_item(it) and isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "PASS")
                y_fail_c = sum(1 for it in yn_items if _is_content_item(it) and isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "FAIL")
                y_unc_c = sum(1 for it in yn_items if _is_content_item(it) and isinstance(it.get("factuality"), dict) and (it["factuality"] or {}).get("status") == "UNCERTAIN")
                yn_sig_content = {"pass": y_pass_c, "fail": y_fail_c, "uncertain": y_unc_c}
                if y_fail_c > 0:
                    desired = (not yn_det)
                elif y_fail_c == 0 and y_unc_c == 0 and y_pass_c > 0:
                    desired = yn_det
                # Summarize directional signal for logging
                if y_fail_c > 0:
                    yn_sig_content_dir = "no"
                elif y_fail_c == 0 and y_unc_c == 0 and y_pass_c > 0:
                    yn_sig_content_dir = "yes"
                else:
                    yn_sig_content_dir = "undecided"
                # Booleans for adjudication rules/logging
                has_any_content_fail = (y_fail_c > 0)
                has_content_positive_pass = (y_fail_c == 0 and y_pass_c > 0)
                # else: leave desired as None (insufficient evidence)

            return {
                "model_res": res,
                "corrected_out": corrected,
                "analysis_source": analysis_src,
                "claims_for_analysis": claims_a,
                "clar_for_analysis": clar_a,
                "fact_for_analysis": fact_a,
                "mmhal_claims": mmhal_items,
                "num_pass": n_pass,
                "num_fail": n_fail,
                "num_uncertain": n_unc,
                "num_pass_content": n_pass_c,
                "num_fail_content": n_fail_c,
                "num_uncertain_content": n_unc_c,
                "max_fail_conf_content": max_fail_conf_c,
                "yn_detected": yn_det,
                "yn_signal": yn_sig,
                "yn_signal_content": yn_sig_content,
                "yn_signal_content_direction": yn_sig_content_dir,
                "has_content_positive_pass": has_content_positive_pass,
                "has_any_content_fail": has_any_content_fail,
                "desired": desired,
            }

        # Prepare prompts: Q0 is ORIGINAL; Q1 uses corrected prompt if Q0 fails
        q0_prompt = _append_yn_instr(prompt)
        q1_prompt = _append_yn_instr(corrected_prompt_candidate)

        # 2) Query SUT with Q0 (original)
        try:
            a0_text = self.sut_llm_manager.generate_text(
                q0_prompt,
                max_tokens=self.max_gen_tokens,
                temperature=self.temperature,
                images=[image_path] if image_path else None,
            )
        except Exception as e:
            self.logger.error(f"Generation error (Q0) for sample {sample_id}: {e}")
            a0_text = ""

        a0 = _analyze_output(a0_text, prompt)
        try:
            # Color helpers
            GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; RESET = "\033[0m"
            self.logger.info(
                f"Sample {sample_id} - Q0 evidence | all: P={a0.get('num_pass',0)} "
                f"F={a0.get('num_fail',0)} U={a0.get('num_uncertain',0)}; "
                f"content: P={a0.get('num_pass_content',0)} "
                f"F={RED}{a0.get('num_fail_content',0)}{RESET} U={a0.get('num_uncertain_content',0)}; "
                f"yn_content={CYAN}{a0.get('yn_signal_content_direction','undecided')}{RESET} "
                f"has_pos_pass={a0.get('has_content_positive_pass', False)} "
                f"has_fail={a0.get('has_any_content_fail', False)}"
            )
        except Exception:
            pass

        # Pre-compute prompt-side MMHal claims for context
        mmhal_prompt_claims = self._claims_to_mmhal(
            user_res.get("claims", []),
            user_res.get("clarification", {}),
            user_res.get("factuality", {}),
            stage="pre",
        )

        # Early overturn: if Q0 has strong content-only contradiction with high confidence, skip Q1 entirely.
        early_overturn_enabled = str(os.getenv("SOC_EARLY_OVERTURN_ENABLED", "true")).lower() == "true"
        try:
            thr = float(os.getenv("SOC_EARLY_OVERTURN_MIN_CONF", "0.9") or 0.9)
        except Exception:
            thr = 0.9
        early_overturn = bool(early_overturn_enabled and int(a0.get("num_fail_content", 0) or 0) > 0 and float(a0.get("max_fail_conf_content", 0.0) or 0.0) >= thr)
        if early_overturn:
            try:
                RED = "\033[91m"; RESET = "\033[0m"
                self.logger.info(f"{RED}Sample {sample_id} - Early overturn: strong contradictory evidence in Q0 (max_conf={a0.get('max_fail_conf_content',0.0):.2f} >= {thr:.2f}); skipping Q1{RESET}")
            except Exception:
                pass
            final_decision = "Q0_early_overturn"
            # Tag strong contradiction for downstream coercion/logging
            a0["strong_content_contradiction"] = True
            final = a0
            final_original = a0_text
        else:
            # Fast-path: only content-relevant evidence can skip Q1/adjudication
            q0_yesno_supported = (
                a0.get("yn_detected") is not None
                and a0.get("desired") is not None
                and bool(a0.get("desired")) == bool(a0.get("yn_detected"))
            )
            q0_claims_supported = (
                a0.get("num_fail_content", 0) == 0 and a0.get("num_pass_content", 0) > 0
            )
            q0_passed = bool(q0_yesno_supported or q0_claims_supported)

            if not early_overturn and q0_passed:
                try:
                    GREEN = "\033[92m"; RESET = "\033[0m"
                    self.logger.info(f"{GREEN}Sample {sample_id} - Fast path: Q0 verified; skipping Q1 and adjudication{RESET}")
                except Exception:
                    pass
                final_decision = "Q0"
                final = a0
                final_original = a0_text
            elif not early_overturn:
                # 3) Query SUT with Q1 (corrected question from clarifications)
                try:
                    a1_text = self.sut_llm_manager.generate_text(
                        q1_prompt,
                        max_tokens=self.max_gen_tokens,
                        temperature=self.temperature,
                        images=[image_path] if image_path else None,
                    )
                except Exception as e:
                    self.logger.error(f"Generation error (Q1) for sample {sample_id}: {e}")
                    a1_text = ""

                a1 = _analyze_output(a1_text, corrected_prompt_candidate)
                try:
                    CYAN = "\033[96m"; RED = "\033[91m"; RESET = "\033[0m"
                    self.logger.info(
                        f"Sample {sample_id} - Q1 evidence | all: P={a1.get('num_pass',0)} F={a1.get('num_fail',0)} U={a1.get('num_uncertain',0)}; "
                        f"content: P={a1.get('num_pass_content',0)} F={RED}{a1.get('num_fail_content',0)}{RESET} U={a1.get('num_uncertain_content',0)}; "
                        f"yn_content={CYAN}{a1.get('yn_signal_content_direction','undecided')}{RESET} has_pos_pass={a1.get('has_content_positive_pass', False)} has_fail={a1.get('has_any_content_fail', False)}"
                    )
                except Exception:
                    pass

                # 4) Adjudicate between A0 and A1 deterministically using verification evidence
                def _score(ana: Dict[str, Any]) -> float:
                    # Content-only evidence score: stronger penalty for fails, light penalty for uncertain
                    pass_c = float(ana.get("num_pass_content", 0) or 0)
                    fail_c = float(ana.get("num_fail_content", 0) or 0)
                    unc_c = float(ana.get("num_uncertain_content", 0) or 0)
                    s = pass_c - 2.0 * fail_c - 0.5 * unc_c
                    # Dominant directional signal from verification (content-only)
                    if ana.get("desired") is True:
                        s += 2.0
                    elif ana.get("desired") is False:
                        s -= 2.0
                    return s

                s0, s1 = _score(a0), _score(a1)
                try:
                    YELLOW = "\033[93m"; RESET = "\033[0m"
                    self.logger.info(
                        f"{YELLOW}Sample {sample_id} - Adjudication scores (content-only): Q0={s0:.2f}, Q1={s1:.2f}{RESET}"
                    )
                except Exception:
                    pass
                # Helper: detect existence-type question
                def _is_existence_question(q: str) -> bool:
                    ql = (q or "").strip().lower()
                    return (
                        ql.startswith("is there ")
                        or ql.startswith("are there ")
                        or "does the image contain" in ql
                        or ql.startswith("is a ") or ql.startswith("is an ")
                        or "any" in ql and (ql.startswith("is there") or ql.startswith("are there"))
                    )

                is_existence = _is_existence_question(prompt) or _is_existence_question(corrected_prompt_candidate)

                if s1 > s0:
                    final_decision = "Q1"
                    final = a1
                    final_original = a1_text
                elif s1 < s0:
                    final_decision = "Q0"
                    final = a0
                    final_original = a0_text
                else:
                    # Tie-breaker per Rule B
                    if is_existence:
                        # If one branch has positive content support, pick it; else prefer 'No'
                        if bool(a0.get("has_content_positive_pass")) ^ bool(a1.get("has_content_positive_pass")):
                            choose_q1 = bool(a1.get("has_content_positive_pass"))
                        else:
                            # Prefer the branch whose desired direction is 'No'
                            if a0.get("desired") is False and a1.get("desired") is not True:
                                choose_q1 = False
                            elif a1.get("desired") is False and a0.get("desired") is not True:
                                choose_q1 = True
                            else:
                                # Default conservative: keep Q0
                                choose_q1 = False
                    else:
                        # Non-existence: prefer corrected only if strictly stronger content not equal
                        choose_q1 = False
                    if choose_q1:
                        final_decision = "Q1_tie"
                        final = a1
                        final_original = a1_text
                    else:
                        final_decision = "Q0_tie"
                        final = a0
                        final_original = a0_text

        # Log final adjudication decision
        try:
            GREEN = "\033[92m"; RESET = "\033[0m"
            self.logger.info(f"{GREEN}Sample {sample_id} - Final adjudication: {final_decision}{RESET}")
        except Exception:
            pass

        # Coerce final corrected output to strict Yes/No if requested
        final_corrected_out = final.get("corrected_out") or final_original
        if getattr(self, "force_yes_no", False):
            # Prefer verification (derived content-only) over raw model answer
            yn = self._detect_yes_no(final_corrected_out or "")
            desired = final.get("desired")
            ysig = final.get("yn_signal_content") or {}
            y_fail_c = int((ysig or {}).get("fail", 0) or 0)
            y_pass_c = int((ysig or {}).get("pass", 0) or 0)
            # 1) Any derived content FAIL -> No
            if y_fail_c > 0 or bool(final.get("strong_content_contradiction", False)):
                final_corrected_out = "No"
            # 2) Strong verification direction -> follow it
            elif desired is True:
                final_corrected_out = "Yes"
            elif desired is False:
                final_corrected_out = "No"
            # 3) Derived content PASS with no fails -> Yes
            elif y_pass_c > 0:
                final_corrected_out = "Yes"
            # 4) Fall back to raw detection if present
            elif yn is True:
                final_corrected_out = "Yes"
            elif yn is False:
                final_corrected_out = "No"
            # 5) Final conservative default
            else:
                final_corrected_out = "No"

        # Build MMHal summary from the FINAL analysis
        mmhal = {
            "version": "0.1",
            "id": sample_id,
            "image": image_path,
            "question": prompt,
            "response_original": final_original,
            "response_corrected": final_corrected_out,
            "claims": final.get("mmhal_claims", []),
            "prompt_claims": mmhal_prompt_claims or None,
            "summary": {
                "num_claims": len(final.get("mmhal_claims", []) or []),
                "num_pass": int(final.get("num_pass", 0)),
                "num_fail": int(final.get("num_fail", 0)),
                "num_uncertain": int(final.get("num_uncertain", 0)),
                "has_hallucination": bool(final.get("num_fail", 0) > 0),
            },
        }

        # Classify hallucination source explicitly for logging/analysis
        halluc_source = "NONE"
        try:
            if int(final.get("num_fail_content", 0) or 0) > 0:
                halluc_source = "MODEL_CONTENT_HALLUCINATION"
            else:
                # Detect question ambiguity/meta in prompt-side claims
                EXCLUDED_META_CATS = {"AMBIGUOUS_RESOLUTION_REQUIRED", "SUBJECTIVE_OPINION", "PROCEDURAL_DESCRIPTIVE"}
                prompt_has_meta = False
                try:
                    for it in (mmhal_prompt_claims or []):
                        cats = it.get("categories") or []
                        if any(c in EXCLUDED_META_CATS for c in cats):
                            prompt_has_meta = True
                            break
                except Exception:
                    prompt_has_meta = False
                if prompt_has_meta:
                    halluc_source = "QUESTION_AMBIGUITY"
                elif int(final.get("num_fail", 0) or 0) > 0:
                    # Fails exist but none in content-only -> meta/non-content
                    halluc_source = "META_NONCONTENT_HALLUCINATION"
        except Exception:
            halluc_source = "UNKNOWN"
        try:
            # Colorize source classification
            GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; MAGENTA = "\033[95m"; RESET = "\033[0m"
            color_map = {
                "NONE": GREEN,
                "MODEL_CONTENT_HALLUCINATION": RED,
                "QUESTION_AMBIGUITY": YELLOW,
                "META_NONCONTENT_HALLUCINATION": MAGENTA,
                "UNKNOWN": YELLOW,
            }
            col = color_map.get(halluc_source, YELLOW)
            self.logger.info(f"Sample {sample_id} - Hallucination source: {col}{halluc_source}{RESET}")
        except Exception:
            pass

        record = {
            "sample_id": sample_id,
            "image_path": image_path,
            "input_original": prompt,
            "input_corrected": corrected_prompt_candidate,
            "input_corrections": user_res.get("corrections", []),
            "input_factuality": user_res.get("factuality", {}),
            "input_claims": to_jsonable(user_res.get("claims", [])),
            "input_clarification": to_jsonable(user_res.get("clarification", {})),
            # Outputs
            "model_output_original": final_original,
            "model_output_corrected": final_corrected_out,
            "model_output_corrections": final.get("model_res", {}).get("corrections", []),
            "model_output_factuality": final.get("model_res", {}).get("factuality", {}),
            "model_output_claims": to_jsonable(final.get("model_res", {}).get("claims", [])),
            "model_output_clarification": to_jsonable(final.get("model_res", {}).get("clarification", {})),
            # Analysis artifacts (may be derived from yes/no + question hypothesis)
            "analysis_source": final.get("analysis_source"),
            "analysis_claims": to_jsonable(final.get("claims_for_analysis", [])),
            "analysis_clarification": to_jsonable(final.get("clar_for_analysis", {})),
            "analysis_factuality": to_jsonable(final.get("fact_for_analysis", {})),
            # Adjudication context
            "adjudication": {
                "decision": final_decision,
                "fast_path": q0_passed,
                "q0_prompt": q0_prompt,
                "q1_prompt": q1_prompt,
            },
            "hallucination_source": halluc_source,
            "mmhal": mmhal,
        }
        return record

    def run(self) -> None:
        data = self.load_dataset(self.dataset_path)
        self.logger.info(f"Loaded {len(data)} samples from {self.dataset_path}")

        processed_ids = self.ckpt.load_processed_ids() if self.resume else set()
        self.logger.info(f"Resuming with {len(processed_ids)} already processed samples")

        total = 0
        for idx, sample in enumerate(data):
            # Use the same identifier logic used elsewhere (respects --id-key and evaluator overrides)
            sid = self.get_sample_id(sample) or idx
            if self.resume and sid in processed_ids:
                continue

            try:
                rec = self.evaluate_sample(sample)
            except Exception as e:
                self.logger.error(f"Error evaluating sample {sid}: {e}")
                rec = {"sample_id": sid, "error": str(e)}

            # Save checkpoint record
            self.ckpt.append_result(rec, sample_id=sid, index=idx)

            total += 1
            if self.limit and total >= self.limit:
                self.logger.info(f"Reached sample limit {self.limit}; stopping.")
                break

        self.logger.info("Evaluation complete.")
