import logging
import os
from .io import ensure_dir


def setup_run_logger(run_dir: str, name: str = "eval") -> logging.Logger:
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, f"{name}.log")

    logger = logging.getLogger(f"mllm_eval.{name}")
    # Honor SOC_LOG_LEVEL (fallback to LOG_LEVEL or INFO)
    lvl_name = str(os.getenv("SOC_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))).upper()
    level = getattr(logging, lvl_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    # Also set root logger to the same level for consistency
    try:
        root = logging.getLogger()
        root.setLevel(level)
    except Exception:
        pass

    # Suppress noisy third-party DEBUG logs unless explicitly disabled
    try:
        quiet_tp = str(os.getenv("SOC_QUIET_THIRDPARTY", "true")).strip().lower() == "true"
        if quiet_tp:
            for n in (
                "transformers",
                "urllib3",
                "requests",
                "huggingface_hub",
                "httpx",
                "httpcore",
                "PIL",
                "matplotlib",
                "openai",
                "anthropic",
            ):
                logging.getLogger(n).setLevel(logging.WARNING)
    except Exception:
        pass

    return logger
