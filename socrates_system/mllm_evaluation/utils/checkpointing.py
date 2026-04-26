import json
import os
import time
from typing import Any, Dict, Iterable, Optional, Set

from .io import ensure_dir, utc_timestamp


class CheckpointManager:
    """
    JSONL-based checkpoint manager for long-running benchmark evaluations.

    - results.jsonl: one JSON record per processed sample
    - state.json:    progress metadata (last_index, last_sample_id, count)
    - meta.json:     run metadata (benchmark, provider/model, start_time, etc.)
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        ensure_dir(self.run_dir)
        self.results_path = os.path.join(self.run_dir, "results.jsonl")
        self.mmhal_results_path = os.path.join(self.run_dir, "mmhal_results.jsonl")
        self.state_path = os.path.join(self.run_dir, "state.json")
        self.meta_path = os.path.join(self.run_dir, "meta.json")

        # Create empty files if they do not exist
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w", encoding="utf-8") as f:
                f.write("")
        if not os.path.exists(self.state_path):
            self._write_json(self.state_path, {"last_index": -1, "last_sample_id": None, "count": 0})
        if not os.path.exists(self.mmhal_results_path):
            with open(self.mmhal_results_path, "w", encoding="utf-8") as f:
                f.write("")

    def write_meta(self, meta: Dict[str, Any]) -> None:
        """Write run metadata (merged with a UTC start timestamp) to the meta JSON file.

        Args:
            meta: Dict of key-value metadata to persist alongside the run.
        """
        self._write_json(self.meta_path, {**meta, "start_time": utc_timestamp()})

    def append_result(self, record: Dict[str, Any], sample_id: Any, index: int) -> None:
        """Append a single evaluation record to the results JSONL file and update the state.

        If the record contains an ``'mmhal'`` key, its value is also mirrored to a
        dedicated ``mmhal_results.jsonl`` file for use by downstream judge tools.

        Args:
            record: The evaluation result dict to persist.
            sample_id: Stable identifier for the sample (used for resume detection).
            index: Zero-based index of the sample in the dataset.
        """
        # Append JSON line atomically and update state
        line = json.dumps(record, ensure_ascii=False)
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        # If MMHal block present, mirror it into a dedicated JSONL for downstream tools
        mmhal = record.get("mmhal") if isinstance(record, dict) else None
        if mmhal is not None:
            try:
                mm_line = json.dumps(mmhal, ensure_ascii=False)
                with open(self.mmhal_results_path, "a", encoding="utf-8") as mf:
                    mf.write(mm_line + "\n")
                    mf.flush()
                    os.fsync(mf.fileno())
            except Exception:
                # Non-fatal; continue
                pass
        self._write_json(self.state_path, {"last_index": index, "last_sample_id": sample_id, "count": index + 1, "updated": utc_timestamp()})

    def load_processed_ids(self) -> Set[Any]:
        """Load the set of already-processed sample IDs from checkpoint files.

        Reads ``results.jsonl`` as the primary source, then falls back to
        ``mmhal_results.jsonl`` and ``mmhal_results.with_ids.jsonl`` for older runs
        that did not record ``sample_id`` in results.

        Returns:
            A set of sample ID values (strings, ints, or any hashable type).
        """
        ids: Set[Any] = set()
        # Primary source: results.jsonl -> sample_id
        if os.path.exists(self.results_path):
            with open(self.results_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        sid = obj.get("sample_id")
                        if sid is not None:
                            ids.add(sid)
                    except Exception:
                        continue

        # Fallback/merge: MMHal outputs -> id
        # This helps resume when older runs didn't record sample_id in results.jsonl
        # but mmhal_results.jsonl (or a patched mmhal_results.with_ids.jsonl) has ids.
        alt_paths = [
            self.mmhal_results_path,
            self.mmhal_results_path.replace(".jsonl", ".with_ids.jsonl"),
        ]
        for path in alt_paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            sid = obj.get("id")
                            if sid is not None:
                                ids.add(sid)
                        except Exception:
                            continue
            except Exception:
                continue

        return ids

    def resume_info(self) -> Dict[str, Any]:
        """Return the last-saved checkpoint state dict.

        Returns:
            A dict with ``'last_index'``, ``'last_sample_id'``, and ``'count'`` keys,
            defaulting to ``{last_index: -1, last_sample_id: None, count: 0}`` when no
            state file exists.
        """
        if not os.path.exists(self.state_path):
            return {"last_index": -1, "last_sample_id": None, "count": 0}
        with open(self.state_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {"last_index": -1, "last_sample_id": None, "count": 0}

    @staticmethod
    def _write_json(path: str, obj: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
