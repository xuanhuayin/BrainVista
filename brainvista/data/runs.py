# brainvista/data/runs.py
"""Per-run loading of aligned stimulus features, brain region tokens and fMRI.

All streams of a run are aligned to the native fMRI grid and truncated to
their common duration T_r = safeT(V_r, A_r, W_r, Z_r) (Appendix A.6). Forecasting
windows are built within a single run only; nothing is carried across runs.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .alignment import (collapse_time_features, fir_lag_bank, hrf_convolve, pool_to_fmri_grid,
                        reduce_layers, safe_t)
from .io import find_feature_file, find_fmri_file, find_token_file, load_npy
from .normalization import normalize

MODALITIES = ("video", "audio", "text")
_RATIO_ARG = {"video": "ratio_v", "audio": "ratio_a", "text": "ratio_w"}


@dataclass
class RunData:
    subj: str
    rid: str
    V: np.ndarray                    # [T, D_v]  (D_v = 0 if inactive)
    A: np.ndarray                    # [T, D_a]
    W: np.ndarray                    # [T, D_w]
    Z: np.ndarray                    # [T, d_brain] brain region tokens
    X: Optional[np.ndarray] = None   # [T, P] normalized model-space fMRI (ground truth)

    @property
    def T(self) -> int:
        return int(self.Z.shape[0])

    @property
    def dims(self):
        return self.V.shape[1], self.A.shape[1], self.W.shape[1], self.Z.shape[1]


def _modality_spec(args, m: str):
    root = getattr(args, f"{m}_root", None)
    ratio = int(getattr(args, _RATIO_ARG[m], 1))
    active = m in getattr(args, "modalities", MODALITIES) and ratio > 0
    return root, ratio, active


def feature_id(rid: str, regex: Optional[str]) -> str:
    """Stimulus-feature file id for a run (e.g. `task-s01e02a` from `ses-003_task-s01e02a`)."""
    if not regex:
        return rid
    m = re.search(regex, rid)
    return m.group(0) if m else rid


def load_tokens(z_root: str, networks: List[str], subj: str, rid: str) -> np.ndarray:
    parts = []
    for net in networks:
        fp = find_token_file(z_root, net, subj, rid)
        if fp is None:
            raise FileNotFoundError(f"Missing brain region tokens: network={net}, subj={subj}, rid={rid}")
        parts.append(load_npy(fp))
    T = min(p.shape[0] for p in parts)
    return np.concatenate([p[:T] for p in parts], axis=1)


def load_fmri(fmri_root: str, networks: List[str], subj: str, rid: str,
              bundles: Optional[Dict] = None, norm_mode: Optional[str] = None) -> np.ndarray:
    """Model-space fMRI of all networks, normalized with the tokenizers' training-set stats."""
    parts = []
    for net in networks:
        fp = find_fmri_file(fmri_root, net, subj, rid)
        if fp is None:
            raise FileNotFoundError(f"Missing fMRI: network={net}, subj={subj}, rid={rid}")
        X = load_npy(fp)
        stats, mode = None, "run"
        if bundles is not None:
            b = bundles[net]
            stats = b.stats_for(subj)
            mode = norm_mode or (b.norm_mode if stats is not None else "run")
        parts.append(normalize(X, stats, mode))
    T = min(p.shape[0] for p in parts)
    return np.concatenate([p[:T] for p in parts], axis=1)


def load_run(args, subj: str, rid: str, bundles: Optional[Dict] = None,
             need_fmri: bool = False) -> RunData:
    Z = load_tokens(args.z_root, args.parcels, subj, rid)
    T_z = Z.shape[0]

    streams = {}
    fid = feature_id(rid, getattr(args, "feature_id_regex", None))
    for m in MODALITIES:
        root, ratio, active = _modality_spec(args, m)
        if not active:
            streams[m] = np.zeros((T_z, 0), dtype=np.float32)
            continue
        fp = find_feature_file(root, subj, fid)
        if fp is None:
            raise FileNotFoundError(f"Missing {m} features: subj={subj}, id={fid} (root={root})")
        raw = reduce_layers(load_npy(fp), args.layer_pool, args.layer_axis, args.select_layer_idx)
        transform = getattr(args, "stim_transform", "none") or "none"
        if transform == "hrf":
            if not getattr(args, "tr", None):
                raise ValueError("--stim_transform hrf needs --tr")
            raw = hrf_convolve(collapse_time_features(raw, args.collapse), float(args.tr) / ratio)
        streams[m], _ = pool_to_fmri_grid(raw, T_z, ratio, collapse=args.collapse)
        if transform == "fir":
            streams[m] = fir_lag_bank(streams[m], int(getattr(args, "fir_lags", 4)))

    arrs = [streams["video"], streams["audio"], streams["text"], Z]
    X = None
    if need_fmri and getattr(args, "fmri_root", None):
        X = load_fmri(args.fmri_root, args.parcels, subj, rid, bundles,
                      getattr(args, "norm_mode", None))
        arrs.append(X)
    arrs, _T = safe_t(*arrs)
    return RunData(subj, rid, arrs[0], arrs[1], arrs[2], arrs[3], arrs[4] if X is not None else None)


def load_runs(args, run_list, bundles=None, need_fmri=False, verbose=True) -> List[RunData]:
    runs = []
    for subj, rid in run_list:
        try:
            runs.append(load_run(args, subj, rid, bundles, need_fmri))
        except FileNotFoundError as e:
            print(f"[Data] skip {subj}/{rid}: {e}")
    if verbose and runs:
        Dv, Da, Dw, Dz = runs[0].dims
        print(f"[Data] {len(runs)}/{len(run_list)} runs | Dv={Dv} Da={Da} Dw={Dw} d_brain={Dz} "
              f"| frames={sum(r.T for r in runs)}")
    return runs
