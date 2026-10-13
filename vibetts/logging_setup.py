"""Rotating file logging."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .config import ROOT

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(console: bool = False, level: int = logging.INFO) -> Path:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "vibetts.log"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_vibetts", False):
            root.removeHandler(h)
            h.close()
    fmt = logging.Formatter(_FORMAT)
    fh = logging.handlers.RotatingFileHandler(
        path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh._vibetts = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    if console and sys.stderr is not None:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        ch._vibetts = True  # type: ignore[attr-defined]
        root.addHandler(ch)
    # phonemizer warns "words count mismatch" on most lines; it is harmless noise
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    return path
