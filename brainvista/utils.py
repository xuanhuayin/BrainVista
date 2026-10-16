# brainvista/utils.py
"""Seeding, device selection and configuration snapshots."""
import json
import os
import random
import subprocess

import numpy as np
import torch


def set_seed(s: int = 2025):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device(cpu: bool = False) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


def enable_tf32():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def dump_config(cfg: dict, path: str, **extra):
    """Write the resolved runtime settings next to the checkpoints (reproducibility snapshot)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = dict(cfg)
    out.update(extra)
    out["_git_commit"] = _git_commit()
    out["_torch"] = torch.__version__
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
