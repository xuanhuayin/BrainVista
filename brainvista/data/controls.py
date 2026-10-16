# brainvista/data/controls.py
"""Stimulus-construction variants and negative controls (paper Appendix A.3).

Constructions (applied when runs are loaded, train and test alike):
    hrf   high-rate features are convolved causally with the canonical SPM
          double-gamma HRF before pooling to the fMRI grid
    fir   pooled features are concatenated over a causal lag bank of
          `fir_lags` bins ending at the queried bin

Negative controls (evaluation only) keep the stimulus statistics but break its
temporal correspondence with the fMRI:
    circular   circular shift within the run by at least `min_shift_sec`
    cross_run  stimulus of another run of the same subject and split
"""
from typing import List

import numpy as np

from .alignment import STIM_TRANSFORMS, fir_lag_bank, hrf_convolve  # noqa: F401  (re-exported)
from .runs import RunData

STIM_CONTROLS = ("none", "circular", "cross_run")


def apply_stim_control(runs: List[RunData], mode: str, tr: float, min_shift_sec: float = 30.0,
                       seed: int = 0) -> List[RunData]:
    if mode == "none":
        return runs
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(runs):
        if mode == "circular":
            lo = int(np.ceil(min_shift_sec / tr))
            if r.T <= 2 * lo:
                raise ValueError(f"{r.subj}/{r.rid}: T={r.T} too short for a >= {min_shift_sec}s circular shift")
            s = int(rng.integers(lo, r.T - lo + 1))
            V, A, W = (np.roll(x, s, axis=0) for x in (r.V, r.A, r.W))
        elif mode == "cross_run":
            donors = [d for j, d in enumerate(runs) if j != i and d.subj == r.subj]
            if not donors:
                raise ValueError(f"{r.subj}: cross-run control needs at least two runs")
            d = donors[int(rng.integers(len(donors)))]
            idx = np.arange(r.T) % d.T
            V, A, W = d.V[idx], d.A[idx], d.W[idx]
        else:
            raise ValueError(f"Unknown stimulus control {mode!r}")
        out.append(RunData(r.subj, r.rid, V, A, W, r.Z, r.X))
    return out
