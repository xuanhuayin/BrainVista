# brainvista/data/normalization.py
"""Training-set normalization (paper Appendix A.4).

For each subject, network k and model-space dimension j, mean and standard
deviation are estimated on training runs only,

    mu_{k,j} = mean_{(r,t) in I_train} x_{r,j}^{k,t},   sigma_{k,j} = std_{(r,t) in I_train} x_{r,j}^{k,t},

and the same fixed statistics are applied to training, validation, test and
rollout data: x_bar = (x - mu) / (sigma + eps). Run-wise z-scoring is kept only
as the offline diagnostic of Table "Normalization strategy".
"""
from typing import Dict, Iterable, Optional

import numpy as np

from .io import find_fmri_file, load_npy

EPS = 1e-6
NORM_MODES = ("train", "run")


def compute_train_stats(fmri_root: str, network: str, subj: str,
                        rids: Iterable[str]) -> Dict[str, np.ndarray]:
    """Streaming mean / population std over all frames of the given training runs."""
    n, s1, s2 = 0, None, None
    for rid in rids:
        fp = find_fmri_file(fmri_root, network, subj, rid)
        if fp is None:
            continue
        X = load_npy(fp).astype(np.float64)
        if s1 is None:
            s1 = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
        n += X.shape[0]
        s1 += X.sum(axis=0)
        s2 += (X ** 2).sum(axis=0)
    if n == 0:
        raise FileNotFoundError(
            f"No training fMRI found for network={network}, subj={subj} under {fmri_root}")
    mu = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0))
    return {"mu": mu.astype(np.float32), "std": std.astype(np.float32), "n_frames": int(n)}


def normalize(X: np.ndarray, stats: Optional[Dict[str, np.ndarray]], mode: str = "train") -> np.ndarray:
    """Apply fixed training-set statistics (mode="train") or run-wise z-scoring (mode="run")."""
    if mode == "run" or stats is None:
        mu, std = X.mean(axis=0), X.std(axis=0)
    else:
        mu, std = stats["mu"], stats["std"]
    return ((X - mu) / (std + EPS)).astype(np.float32)


def denormalize(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (X * (stats["std"] + EPS) + stats["mu"]).astype(np.float32)
