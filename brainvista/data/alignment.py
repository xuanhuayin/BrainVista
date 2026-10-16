# brainvista/data/alignment.py
"""Temporal alignment of high-rate stimulus features to the fMRI grid (paper §3.2, Eq. 3).

Features are expected on a regular grid at `ratio` samples per fMRI bin, where
sample i belongs to bin floor(i / ratio), i.e. its extraction window ends no
later than that bin's end time (end-aligned convention, Appendix A.3). Every
active modality is mean-pooled within each bin:

    tau_m^t = 1/|B_{m,t}| * sum_{i in B_{m,t}} tau'_{m,i}.

A modality whose features are already on the fMRI grid uses ratio = 1.
"""
from typing import List, Tuple

import numpy as np


def reduce_layers(X: np.ndarray, mode: str = "mean", axis: int = 1, idx: int = -1) -> np.ndarray:
    """Collapse a (T, n_layers, D) feature stack to (T, D)."""
    if X.ndim == 2:
        return X
    if mode == "mean":
        return X.mean(axis=axis)
    if mode == "last":
        return np.take(X, -1, axis=axis)
    if mode == "select":
        return np.take(X, idx, axis=axis)
    if mode == "concat":
        return X.reshape(X.shape[0], -1)
    raise ValueError(f"Unknown layer reduction mode: {mode}")


def collapse_time_features(arr: np.ndarray, mode: str = "flatten") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim >= 3:
        if mode != "flatten":
            raise ValueError(f"Unsupported collapse mode: {mode}")
        arr = arr.reshape(arr.shape[0], -1)
    return np.ascontiguousarray(arr)


def pool_to_fmri_grid(M: np.ndarray, T_fmri: int, ratio: int,
                      collapse: str = "flatten") -> Tuple[np.ndarray, int]:
    """Mean-pool `ratio` consecutive feature samples into each fMRI bin.

    Returns the pooled (T, D) array and its length, which is shorter than
    `T_fmri` when the feature stream ends early.
    """
    M2d = collapse_time_features(M, mode=collapse)
    Tm, D = M2d.shape
    ratio = int(max(1, ratio))
    T_eff = min(int(T_fmri), Tm // ratio)
    if T_eff <= 0:
        raise ValueError(f"Feature stream too short: Tm={Tm}, ratio={ratio}")
    pooled = M2d[:T_eff * ratio].reshape(T_eff, ratio, D).mean(axis=1)
    return np.ascontiguousarray(pooled, dtype=np.float32), T_eff


STIM_TRANSFORMS = ("none", "hrf", "fir")


def canonical_hrf(dt: float, length: float = 32.0) -> np.ndarray:
    """SPM canonical double-gamma HRF sampled every dt seconds, normalized to unit sum."""
    from scipy.stats import gamma
    t = np.arange(0.0, length, dt)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return (h / h.sum()).astype(np.float32)


def hrf_convolve(M: np.ndarray, dt: float) -> np.ndarray:
    """Causal convolution along time, y[i] = sum_k h[k] x[i - k] (no future samples)."""
    h = canonical_hrf(dt)
    out = np.zeros_like(M, dtype=np.float32)
    for k, w in enumerate(h[:M.shape[0]]):
        out[k:] += w * M[:M.shape[0] - k]
    return out


def fir_lag_bank(M: np.ndarray, n_lags: int) -> np.ndarray:
    """Causal lag bank [T, D] -> [T, D * n_lags] with lags 0..n_lags-1 (zeros before run start)."""
    T, D = M.shape
    out = np.zeros((T, D * n_lags), dtype=np.float32)
    for lag in range(n_lags):
        out[lag:, lag * D:(lag + 1) * D] = M[:T - lag]
    return out


def safe_t(*arrs: np.ndarray) -> Tuple[List[np.ndarray], int]:
    """safeT: truncate all streams to their common duration T_r."""
    T = min(x.shape[0] for x in arrs)
    return [x[:T] for x in arrs], T
