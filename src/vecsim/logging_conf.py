from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging.

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO").
        log_file: If provided, also write logs to this file.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(lvl)
    handlers.append(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(lvl)
        handlers.append(file_handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for h in handlers:
        h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(lvl)

    # Clear existing handlers to avoid duplicated logs (useful in notebooks)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
