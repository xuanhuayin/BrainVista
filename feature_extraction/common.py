# feature_extraction/common.py
"""TR-locked, end-aligned sampling grid shared by the frozen stimulus encoders (Appendix A.3, A.8).

fMRI bin t covers scan time [t*TR, (t+1)*TR). Each bin receives `ratio`
feature samples whose extraction windows END at

    tau_{t,j} = t*TR + (j+1) * TR/ratio,     j = 0..ratio-1,

so every window ends no later than the bin boundary t_u = (t+1)*TR and the
stride is TR/ratio (e.g. 1.49 s / 3 ~ 0.5 s for Algonauts, 0.8 s / 2 = 0.4 s for
CineBrain, 2.0 s / 4 = 0.5 s for HAD). Sample i belongs to bin i // ratio,
which is exactly what `brainvista.data.alignment.pool_to_fmri_grid` assumes.
`onset_offset` is the scan time at which the stimulus starts.
"""
import csv
import json
import os
from typing import Dict, List

import numpy as np


def sample_end_times(n_bins: int, tr: float, ratio: int, onset_offset: float = 0.0) -> np.ndarray:
    """Window end times in stimulus time (seconds), shape [n_bins * ratio]."""
    t = np.arange(n_bins)[:, None] * tr + (np.arange(ratio)[None, :] + 1) * (tr / ratio)
    return (t.reshape(-1) - onset_offset).astype(np.float64)


def read_manifest(path: str) -> List[Dict[str, str]]:
    """CSV/TSV with columns `rid,path` and optional `n_bins,onset_offset`."""
    with open(path, newline="") as f:
        dialect = csv.Sniffer().sniff(f.read(2048), delimiters=",\t")
        f.seek(0)
        rows = list(csv.DictReader(f, dialect=dialect))
    for r in rows:
        if "rid" not in r or "path" not in r:
            raise ValueError(f"{path}: manifest needs columns rid,path (got {list(r)})")
    return rows


def n_bins_for(row: Dict[str, str], duration: float, tr: float) -> int:
    if row.get("n_bins"):
        return int(row["n_bins"])
    return int(np.floor((duration - float(row.get("onset_offset") or 0.0)) / tr))


def save_features(out_root: str, rid: str, feats: np.ndarray, dtype: str = "float16"):
    os.makedirs(out_root, exist_ok=True)
    np.save(os.path.join(out_root, f"{rid}.npy"), feats.astype(dtype))


def dump_extraction_cfg(out_root: str, cfg: dict):
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "extraction_cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def select_layers(hidden_states, layers: List[int]):
    """Pick encoder layers from a tuple of hidden states (index 0 = embeddings)."""
    n = len(hidden_states)
    return [hidden_states[i if i >= 0 else n + i] for i in layers]
