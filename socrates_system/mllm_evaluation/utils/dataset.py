import csv
import json
import os
from typing import Any, Dict, List, Optional

from .io import iter_jsonl


def load_dataset_generic(path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from JSON/JSONL/CSV.
    Returns a list of dict samples.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
        return list(iter_jsonl(path))
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            if isinstance(obj, list):
                return obj
            raise ValueError("Unsupported JSON structure. Expected list or dict with 'data' list.")
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    raise ValueError("Unsupported dataset extension. Use .jsonl, .json, or .csv")


DEFAULT_PROMPT_KEYS = [
    "question",
    "prompt",
    "instruction",
    "query",
    "Q",
    "text",
]


def get_prompt_text(sample: Dict[str, Any], key_override: Optional[str] = None, fallback_keys: Optional[List[str]] = None) -> str:
    if key_override:
        val = sample.get(key_override)
        if isinstance(val, str) and val.strip():
            return val
    keys = fallback_keys or DEFAULT_PROMPT_KEYS
    for k in keys:
        val = sample.get(k)
        if isinstance(val, str) and val.strip():
            return val
    # If nothing found, return stringified sample as last resort (for debugging)
    return json.dumps(sample, ensure_ascii=False)[:2048]
