"""Timestamped logging used across the package."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT = "shiftguard"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the ``shiftguard`` logger hierarchy.

    Calling this more than once replaces previously installed handlers, so a CLI command
    can redirect logs into a run directory.

    Parameters
    ----------
    level
        Logging level (name or number).
    log_file
        Optional file that receives a copy of every log record.
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``shiftguard`` logger for module ``name``."""
    if name == _ROOT or name.startswith(f"{_ROOT}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT}.{name}")
