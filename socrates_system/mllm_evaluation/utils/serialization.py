from dataclasses import is_dataclass, asdict
from enum import Enum
from typing import Any, Dict, List


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses and Enums to JSON-serializable structures."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        # Prefer enum name
        try:
            return obj.name
        except Exception:
            return str(obj.value)
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    # Fallback to string
    return str(obj)
