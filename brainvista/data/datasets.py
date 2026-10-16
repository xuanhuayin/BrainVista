# brainvista/data/datasets.py
"""Datasets for tokenizer pretraining and teacher-forced next-token training.

Anchor convention (Appendix A.6). With context length L (stimulus window L,
brain-history window L-1 plus the target query) and m = L, a query time u is
valid for one-step training iff m <= u <= T_r - 1, giving N^1step_r = T_r - m
anchors per run; an H-step rollout started at u is valid iff
m <= u <= T_r - H, giving N^H_r = T_r - m - H + 1. Windows never cross runs.

`stim_offset` shifts the stimulus window back by that many bins: 0 is the
default s^{<=u} access, 1 the stricter s^{<u} cutoff, negative values the
diagnostic look-ahead conditions of the stimulus lag sweep.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import find_fmri_file, load_npy
from .normalization import normalize
from .runs import RunData


def anchor_bounds(T: int, L: int, H: int = 1, stim_offset: int = 0) -> Tuple[int, int]:
    """Inclusive range [u_min, u_max] of valid start times for an H-step forecast."""
    u_min = max(L, L - 1 + stim_offset)
    u_max = T - H + min(0, stim_offset)
    return u_min, u_max


def run_anchors(T: int, L: int, H: int = 1, stim_offset: int = 0, stride: int = 1) -> List[int]:
    u_min, u_max = anchor_bounds(T, L, H, stim_offset)
    return list(range(u_min, u_max + 1, max(1, stride)))


class WindowDataset(Dataset):
    """Teacher-forced one-step samples (V, A, W, Z_hist, z_target) at every valid anchor u.

    V/A/W cover times u-L+1 .. u (shifted back by stim_offset); Z_hist covers
    u-L+1 .. u-1; z_target = Z[u] is supervision only and never an input.
    """

    def __init__(self, runs: List[RunData], L: int, stim_offset: int = 0, stride: int = 1):
        self.runs = runs
        self.L = int(L)
        self.stim_offset = int(stim_offset)
        self.items = [(ri, u) for ri, r in enumerate(runs)
                      for u in run_anchors(r.T, self.L, 1, self.stim_offset, stride)]
        if runs:
            self.Dv, self.Da, self.Dw, self.Dz = runs[0].dims
        else:
            self.Dv = self.Da = self.Dw = self.Dz = 0
        print(f"  [WindowDataset] runs={len(runs)} anchors={len(self.items)} "
              f"Dv={self.Dv} Da={self.Da} Dw={self.Dw} d_brain={self.Dz}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ri, u = self.items[i]
        r, L = self.runs[ri], self.L
        s0 = u - L + 1 - self.stim_offset
        return (r.V[s0:s0 + L], r.A[s0:s0 + L], r.W[s0:s0 + L],
                r.Z[u - L + 1:u], r.Z[u])


def collate_windows(batch):
    return tuple(torch.from_numpy(np.stack([b[j] for b in batch], 0)).float() for j in range(5))


class TokenizerDataset(Dataset):
    """Normalized fMRI segments of one functional network for reconstruction pretraining (Eq. 14)."""

    def __init__(self, fmri_root: str, network: str, run_list: Sequence[Tuple[str, str]],
                 norm_stats: Dict[str, Dict[str, np.ndarray]], norm_mode: str = "train",
                 seg_len: int = 64, stride: Optional[int] = None):
        self.seg_len = int(seg_len)
        self.frames: List[np.ndarray] = []
        self.items: List[Tuple[int, int]] = []
        self.Db = None
        stride = max(1, int(stride or 1))
        for subj, rid in run_list:
            fp = find_fmri_file(fmri_root, network, subj, rid)
            if fp is None:
                print(f"[TokenizerDataset] skip {subj}/{rid}: no fMRI for {network}")
                continue
            X = normalize(load_npy(fp), norm_stats.get(subj), norm_mode)
            if self.Db is None:
                self.Db = X.shape[1]
            elif X.shape[1] != self.Db:
                raise ValueError(f"{fp}: dimension {X.shape[1]} != {self.Db}")
            ri = len(self.frames)
            self.frames.append(X)
            for t0 in range(0, max(X.shape[0] - self.seg_len + 1, 1), stride):
                self.items.append((ri, t0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ri, t0 = self.items[i]
        x = self.frames[ri][t0:t0 + self.seg_len]
        mask = np.ones(self.seg_len, dtype=bool)
        if x.shape[0] < self.seg_len:
            pad = self.seg_len - x.shape[0]
            mask[-pad:] = False
            x = np.pad(x, ((0, pad), (0, 0)))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(mask)
