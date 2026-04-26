"""
Logging utility for the Socrates Agent System
"""
import logging
import sys
from pathlib import Path
from socrates_system.config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR, CONSOLE_LOG_LEVEL

def setup_logger(name: str) -> logging.Logger:
    """Set up a dual-handler logger with consistent formatting.

    Creates a logger that writes at `LOG_LEVEL` to a per-module file under
    `LOGS_DIR` and at `CONSOLE_LOG_LEVEL` to stdout. Idempotent: if the
    named logger already has handlers, it is returned as-is to avoid
    duplicate output.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
            The last component (after the final ``.``) is used as the log
            file stem.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Console handler (reduced verbosity)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, CONSOLE_LOG_LEVEL))
        
        # File handler (full verbosity per LOG_LEVEL)
        log_file = LOGS_DIR / f"{name.split('.')[-1]}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        
        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger
