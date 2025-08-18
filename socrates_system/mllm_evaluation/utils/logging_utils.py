import logging
import os
from .io import ensure_dir


def setup_run_logger(run_dir: str, name: str = "eval") -> logging.Logger:
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, f"{name}.log")

    logger = logging.getLogger(f"mllm_eval.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger
