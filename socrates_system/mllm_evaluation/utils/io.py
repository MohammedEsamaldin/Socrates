import json
import os
import gzip
from typing import Any, Dict, Iterable, Iterator, List, Optional
from datetime import datetime, timezone


def ensure_dir(path: str) -> None:
    """Create a directory and any missing parents, no-op if path is empty.

    Args:
        path: Directory path to create.
    """
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        A timezone-aware ISO-8601 datetime string (e.g. ``'2024-01-15T12:00:00+00:00'``).
    """
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str) -> Any:
    """Read and return the parsed contents of a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    """Atomically write an object to a JSON file using a temp-file swap.

    Args:
        path: Destination file path.
        obj: Any JSON-serialisable Python object.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Iterate over records in a JSONL (or gzipped JSONL) file, skipping malformed lines.

    Args:
        path: Path to the JSONL or ``.jsonl.gz`` file.

    Yields:
        Parsed record dicts.
    """
    open_fn = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with open_fn(path, mode, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_lines(path: str) -> List[str]:
    """Read all lines from a text file, stripping trailing newlines.

    Args:
        path: Path to the text file.

    Returns:
        A list of line strings without trailing ``'\\n'``.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]
