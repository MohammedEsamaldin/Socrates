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
        self.state_path = os.path.join(self.run_dir, "state.json")
        self.meta_path = os.path.join(self.run_dir, "meta.json")

        # Create empty files if they do not exist
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w", encoding="utf-8") as f:
                f.write("")
        if not os.path.exists(self.state_path):
            self._write_json(self.state_path, {"last_index": -1, "last_sample_id": None, "count": 0})

    def write_meta(self, meta: Dict[str, Any]) -> None:
        self._write_json(self.meta_path, {**meta, "start_time": utc_timestamp()})

    def append_result(self, record: Dict[str, Any], sample_id: Any, index: int) -> None:
        # Append JSON line atomically and update state
        line = json.dumps(record, ensure_ascii=False)
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._write_json(self.state_path, {"last_index": index, "last_sample_id": sample_id, "count": index + 1, "updated": utc_timestamp()})

    def load_processed_ids(self) -> Set[Any]:
        ids: Set[Any] = set()
        if not os.path.exists(self.results_path):
            return ids
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
        return ids

    def resume_info(self) -> Dict[str, Any]:
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
