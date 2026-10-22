"""Shared configuration, logging, and reproducibility helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_project_root"] = str(path.parent.parent)
    return config


def project_path(config: dict[str, Any], configured_path: str) -> Path:
    return Path(config["_project_root"]) / configured_path


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logging.getLogger(__name__).debug("PyTorch unavailable; seeded Python and NumPy only")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)

