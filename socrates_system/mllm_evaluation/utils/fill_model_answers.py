import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSON dataset expected to be a list of objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset at {path} is not a JSON array.")
    return data


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def build_mmhal_map(mmhal_results_path: str) -> Dict[str, str]:
    """Build a mapping from sample id -> response_corrected from mmhal_results.jsonl."""
    rows = load_jsonl(mmhal_results_path)
    mapping: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("id")
        resp = row.get("response_corrected")
        if sid is None or not isinstance(resp, str):
            continue
        mapping[str(sid)] = resp
    return mapping


def resolve_id(item: Dict[str, Any], id_key: Optional[str]) -> Optional[str]:
    """Resolve an identifier from a dataset item.

    Prefer the provided id_key. Fallback to common keys used in MMHal-like datasets.
    """
    if id_key and item.get(id_key) is not None:
        return str(item.get(id_key))
    for k in ("question_id", "uid", "id", "idx", "image_id"):
        if item.get(k) is not None:
            return str(item.get(k))
    return None


def fill_model_answers(
    dataset_path: str,
    mmhal_results_path: str,
    output_path: str,
    id_key: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """Fill model_answer in a copy of the dataset using response_corrected from mmhal results.

    Returns (matched, total) counts.
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace.")

    dataset = load_json_dataset(dataset_path)
    mm_map = build_mmhal_map(mmhal_results_path)

    matched = 0
    total = len(dataset)

    for item in dataset:
        sid = resolve_id(item, id_key)
        corrected = mm_map.get(str(sid)) if sid is not None else None
        if isinstance(corrected, str) and corrected.strip():
            item["model_answer"] = corrected
            matched += 1
        else:
            # Ensure the key exists with empty string for compatibility
            if "model_answer" not in item:
                item["model_answer"] = ""

    # Write output with same structure (list of dicts)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    return matched, total


def main():
    p = argparse.ArgumentParser(
        description="Fill model_answer in a JSON dataset using corrected responses from mmhal_results.jsonl",
    )
    p.add_argument("--dataset", required=True, help="Path to input dataset JSON (array of objects)")
    p.add_argument("--mmhal-results", required=True, help="Path to mmhal_results.jsonl produced by a run")
    p.add_argument("--output", required=True, help="Path to write the filled JSON dataset")
    p.add_argument("--id-key", default=None, help="Field name to use as dataset item id (e.g., image_id)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file if it exists")
    args = p.parse_args()

    matched, total = fill_model_answers(
        dataset_path=args.dataset,
        mmhal_results_path=args.mmhal_results,
        output_path=args.output,
        id_key=args.id_key,
        overwrite=args.overwrite,
    )
    print(f"Filled model_answer for {matched}/{total} items -> {args.output}")


if __name__ == "__main__":
    main()
