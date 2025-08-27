import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEvaluator


class MMEEvaluator(BaseEvaluator):
    BENCHMARK_NAME = "mme"

    def get_sample_id(self, sample: Dict[str, Any]):
        for k in [self.id_key, "id", "qid", "uid", "sample_id", "idx"]:
            if k and sample.get(k) is not None:
                return sample.get(k)
        return super().get_sample_id(sample)

    def sample_to_prompt(self, sample: Dict[str, Any]) -> str:
        for k in [self.prompt_key, "question", "prompt", "instruction", "query", "text"]:
            if k and isinstance(sample.get(k), str) and sample[k].strip():
                return sample[k]
        return super().sample_to_prompt(sample)

    # ---------- dataset loading for MME_Benchmark directory ----------
    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Load MME_Benchmark when a directory is provided, otherwise fall back.

        Directory patterns supported:
        - Pattern A (category/images + category/questions_answers_YN/*.txt)
        - Pattern B (category contains paired *.jpg|*.png and *.txt with same stem)
        Each .txt may contain multiple tab-separated lines: question \t gt
        """
        if not os.path.isdir(path):
            return super().load_dataset(path)

        root = os.path.abspath(path)
        samples: List[Dict[str, Any]] = []

        def _split_q_gt(line: str) -> Optional[Tuple[str, str]]:
            line = line.strip()
            if not line:
                return None
            parts = line.split("\t")
            if len(parts) < 2:
                return None
            q = parts[0].strip()
            gt = parts[1].strip()
            if not q:
                return None
            return q, gt

        def _resolve_img_with_stem(img_dir: str, stem: str) -> Optional[str]:
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"):
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    return candidate
            return None

        # Case 0: dataset points directly to a single category directory
        root_qa_dir = os.path.join(root, "questions_answers_YN")
        root_img_dir = os.path.join(root, "images")
        if os.path.isdir(root_qa_dir) and os.path.isdir(root_img_dir):
            cat = os.path.basename(root.rstrip(os.sep))
            for fname in sorted(os.listdir(root_qa_dir)):
                if not fname.endswith(".txt"):
                    continue
                stem = os.path.splitext(fname)[0]
                abs_img = _resolve_img_with_stem(root_img_dir, stem)
                if not abs_img:
                    cands = [f for f in os.listdir(root_img_dir) if f.startswith(stem + ".")]
                    if cands:
                        abs_img = os.path.join(root_img_dir, cands[0])
                if not abs_img:
                    continue
                img_base = os.path.basename(abs_img)
                with open(os.path.join(root_qa_dir, fname), "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        parsed = _split_q_gt(line)
                        if not parsed:
                            continue
                        question, gt = parsed
                        sid = f"{cat}/{stem}#{i}"
                        samples.append({
                            "id": sid,
                            "category": cat,
                            "image": abs_img,
                            "image_basename": img_base,
                            "question": question,
                            "gt": gt,
                        })
            return samples

        # Case 1: dataset points to a category dir with paired files
        has_images = any(name.lower().endswith((".jpg", ".jpeg", ".png")) for name in os.listdir(root))
        has_txts = any(name.lower().endswith(".txt") for name in os.listdir(root))
        if has_images and has_txts:
            cat = os.path.basename(root.rstrip(os.sep))
            stems: Dict[str, Tuple[str, str]] = {}
            for fname in sorted(os.listdir(root)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    stem = os.path.splitext(fname)[0]
                    abs_img = os.path.join(root, fname)
                    stems[stem] = (abs_img, fname)
            for stem, (abs_img, img_base) in stems.items():
                txt_path = os.path.join(root, stem + ".txt")
                if not os.path.exists(txt_path):
                    continue
                with open(txt_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        parsed = _split_q_gt(line)
                        if not parsed:
                            continue
                        question, gt = parsed
                        sid = f"{cat}/{stem}#{i}"
                        samples.append({
                            "id": sid,
                            "category": cat,
                            "image": abs_img,
                            "image_basename": img_base,
                            "question": question,
                            "gt": gt,
                        })
            return samples

        # Iterate immediate subdirectories as categories
        for cat in sorted(os.listdir(root)):
            cat_dir = os.path.join(root, cat)
            if not os.path.isdir(cat_dir) or cat.startswith('.'):
                continue

            qa_dir = os.path.join(cat_dir, "questions_answers_YN")
            img_dir = os.path.join(cat_dir, "images")

            if os.path.isdir(qa_dir) and os.path.isdir(img_dir):
                # Pattern A
                for fname in sorted(os.listdir(qa_dir)):
                    if not fname.endswith(".txt"):
                        continue
                    stem = os.path.splitext(fname)[0]
                    abs_img = _resolve_img_with_stem(img_dir, stem)
                    if not abs_img:
                        # try any image starting with stem
                        cands = [f for f in os.listdir(img_dir) if f.startswith(stem + ".")]
                        if cands:
                            abs_img = os.path.join(img_dir, cands[0])
                    if not abs_img:
                        continue
                    img_base = os.path.basename(abs_img)

                    with open(os.path.join(qa_dir, fname), "r", encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            parsed = _split_q_gt(line)
                            if not parsed:
                                continue
                            question, gt = parsed
                            sid = f"{cat}/{stem}#{i}"
                            samples.append({
                                "id": sid,
                                "category": cat,
                                "image": abs_img,
                                "image_basename": img_base,
                                "question": question,
                                "gt": gt,
                            })
                continue

            # Pattern B: paired files in same directory
            # Map stems -> (abs_img, img_base)
            stems: Dict[str, Tuple[str, str]] = {}
            for fname in sorted(os.listdir(cat_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    stem = os.path.splitext(fname)[0]
                    abs_img = os.path.join(cat_dir, fname)
                    stems[stem] = (abs_img, fname)

            for stem, (abs_img, img_base) in stems.items():
                txt_path = os.path.join(cat_dir, stem + ".txt")
                if not os.path.exists(txt_path):
                    continue
                with open(txt_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        parsed = _split_q_gt(line)
                        if not parsed:
                            continue
                        question, gt = parsed
                        sid = f"{cat}/{stem}#{i}"
                        samples.append({
                            "id": sid,
                            "category": cat,
                            "image": abs_img,
                            "image_basename": img_base,
                            "question": question,
                            "gt": gt,
                        })

        return samples


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MME with Socrates MITM pipeline")
    p.add_argument("--dataset", required=True, help="Path to the MME_Benchmark directory or dataset file (json/jsonl/csv)")
    p.add_argument("--run-dir", default=os.path.join("mllm_evaluation", "runs"), help="Directory to store run outputs")
    # Back-compat defaults (used if separate SUT/pipeline not specified)
    p.add_argument("--provider", default=None, help="Default LLM provider for both SUT and pipeline (ollama|openai|claude|llava_hf)")
    p.add_argument("--model", default=None, help="Default model name for the provider")
    # Separate configuration (optional)
    p.add_argument("--sut-provider", dest="sut_provider", default=None, help="Provider for baseline generation (system-under-test), e.g., llava_hf|ollama|openai|claude")
    p.add_argument("--sut-model", dest="sut_model", default=None, help="Model for SUT generation, e.g., liuhaotian/llava-v1.5-7b or llava:13b")
    p.add_argument("--pipeline-provider", dest="pipeline_provider", default=None, help="Provider for MITM pipeline (claims/clarification/etc), e.g., openai")
    p.add_argument("--pipeline-model", dest="pipeline_model", default=None, help="Model for MITM pipeline, e.g., gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None, help="Max samples to process")
    p.add_argument("--no-resume", action="store_true", help="Disable resume from checkpoints")
    p.add_argument("--max-gen-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--prompt-key", default=None, help="Override prompt field name in dataset")
    p.add_argument("--id-key", default=None, help="Override id field name in dataset")
    p.add_argument("--include-list", default=None, help="Path to a text file with one selector per line to include: sample_id 'cat/stem#i', a stem 'cat/stem' (all questions), an image basename like '0001.jpg' or '0001', or a category name. Lines starting with # are ignored.")
    p.add_argument("--mme-results-dir", default=None, help="Directory to write official MME results (.txt per category)")
    p.add_argument("--mme-original", action="store_true", help="Use original (uncorrected) model output in MME results; default uses corrected output")
    p.add_argument("--force-yes-no", action="store_true", help="Force SUT to answer strictly 'Yes' or 'No' and coerce output accordingly")
    return p


def main():
    args = build_arg_parser().parse_args()

    evaluator = MMEEvaluator(
        dataset_path=args.dataset,
        run_dir=args.run_dir,
        # Back-compat defaults
        provider=args.provider,
        model_name=args.model,
        # Separate SUT vs pipeline
        sut_provider=(args.sut_provider or args.provider),
        sut_model_name=(args.sut_model or args.model),
        pipeline_provider=(args.pipeline_provider or args.provider),
        pipeline_model_name=(args.pipeline_model or args.model),
        limit=args.limit,
        resume=not args.no_resume,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        prompt_key=args.prompt_key,
        id_key=args.id_key,
        image_root=args.dataset,
        image_key="image",
        force_yes_no=args.force_yes_no,
    )

    # Run with MME writer
    data = evaluator.load_dataset(args.dataset)
    # Optional: filter to a specific subset via include list
    if args.include_list:
        try:
            with open(args.include_list, "r", encoding="utf-8") as f:
                raw_selectors = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        except Exception as e:
            evaluator.logger.error(f"Failed to read include list {args.include_list}: {e}")
            raw_selectors = []

        if raw_selectors:
            existing_categories = set(str(s.get("category", "")) for s in data)
            id_selectors = set([s for s in raw_selectors if "#" in s])
            nohash = [s for s in raw_selectors if "#" not in s]
            category_selectors = set([s for s in nohash if ("/" not in s) and (s in existing_categories)])
            catstem_selectors = set([s for s in nohash if ("/" in s)])
            basename_selectors = set([s for s in nohash if ("." in s and "/" not in s and s not in category_selectors)])
            stem_selectors = set([s for s in nohash if ("." not in s and "/" not in s and s not in category_selectors)])

            def sample_matches(sample: Dict[str, Any]) -> bool:
                sid = evaluator.get_sample_id(sample) or ""
                if sid in id_selectors:
                    return True
                img_base = sample.get("image_basename") or os.path.basename(sample.get("image", ""))
                stem = os.path.splitext(img_base)[0]
                cat = sample.get("category", "")
                catstem = f"{cat}/{stem}"
                if cat in category_selectors:
                    return True
                if catstem in catstem_selectors:
                    return True
                if img_base in basename_selectors or stem in basename_selectors:
                    return True
                if stem in stem_selectors:
                    return True
                # Support 'stem#i', 'img.jpg#i', or 'cat/stem#i' via id_selectors
                for sel in id_selectors:
                    left, _, qidx = sel.partition("#")
                    if not qidx.isdigit():
                        continue
                    if left in (catstem, stem, img_base) and sid.endswith(f"#{qidx}"):
                        return True
                return False

            before = len(data)
            data = [s for s in data if sample_matches(s)]
            evaluator.logger.info(f"Applied include list {args.include_list}: {before} -> {len(data)} samples")
        else:
            evaluator.logger.warning(f"Include list {args.include_list} was empty; proceeding with all samples")
    evaluator.logger.info(f"Loaded {len(data)} MME samples from {args.dataset}")

    processed_ids = evaluator.ckpt.load_processed_ids() if evaluator.resume else set()
    evaluator.logger.info(f"Resuming with {len(processed_ids)} already processed samples")

    mme_dir = args.mme_results_dir or os.path.join(evaluator.run_dir, "mme_results")
    os.makedirs(mme_dir, exist_ok=True)

    total = 0
    for idx, sample in enumerate(data):
        sid = evaluator.get_sample_id(sample) or idx
        if evaluator.resume and sid in processed_ids:
            continue

        try:
            rec = evaluator.evaluate_sample(sample)
        except Exception as e:
            evaluator.logger.error(f"Error evaluating sample {sid}: {e}")
            rec = {"sample_id": sid, "error": str(e)}

        # Choose which output to use for official MME line
        use_corrected = (not args.mme_original)
        response_text = rec.get("model_output_corrected") if use_corrected else rec.get("model_output_original")
        # When requested, coerce output to canonical 'Yes' or 'No'
        if args.force_yes_no:
            yn = evaluator._detect_yes_no(response_text or "")
            if yn is True:
                response_text = "Yes"
            elif yn is False:
                response_text = "No"
            else:
                s = (response_text or "").strip().lower()
                if ("yes" in s) and ("no" not in s):
                    response_text = "Yes"
                elif ("no" in s) and ("yes" not in s):
                    response_text = "No"
                else:
                    # Default fallback to 'No' to ensure strict Y/N output
                    response_text = "No"
        img_base = sample.get("image_basename") or os.path.basename(sample.get("image", ""))
        category = sample.get("category", "unknown")
        question = sample.get("question", "")
        gt = sample.get("gt", "")

        # Append MME line: img\tquestion\tgt\tresponse
        out_path = os.path.join(mme_dir, f"{category}.txt")
        try:
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(f"{img_base}\t{question}\t{gt}\t{(response_text or '').strip()}\n")
        except Exception as e:
            evaluator.logger.error(f"Failed to write MME line for sample {sid} to {out_path}: {e}")

        # Save checkpoint record
        evaluator.ckpt.append_result(rec, sample_id=sid, index=idx)

        total += 1
        if evaluator.limit and total >= evaluator.limit:
            evaluator.logger.info(f"Reached sample limit {evaluator.limit}; stopping.")
            break

    evaluator.logger.info(f"Evaluation complete. MME results written to: {mme_dir}")


if __name__ == "__main__":
    main()
