# brainvista/rollout.py
"""Conditional autoregressive rollout (paper Eq. 2; Appendix A.5).

From an anchor u0, the brain history is initialized with observed tokens
Z[u0-L+1 : u0]. For h = 0..H-1 the model predicts the token at u = u0 + h from
the stimulus window ending at u (always observed) and the current history; the
prediction is appended to the history, so after initialization no ground-truth
fMRI is ever used. Many anchors of a run are rolled out in parallel.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data.runs import RunData


def run_to_device(run: RunData, device) -> Tuple[torch.Tensor, ...]:
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
    return to(run.V), to(run.A), to(run.W), to(run.Z)


@torch.no_grad()
def rollout(model, V: torch.Tensor, A: torch.Tensor, W: torch.Tensor, Z: torch.Tensor,
            anchors: Sequence[int], H: int, stim_offset: int = 0, batch_size: int = 128,
            use_amp: bool = True, return_attn: bool = False,
            attn_mask: Optional[torch.Tensor] = None):
    """Roll out H steps from each anchor of one run.

    Args:
        V, A, W, Z: per-run tensors [T, D] on the model's device.
        anchors: start times u0 (see `data.datasets.anchor_bounds`).
    Returns:
        Z_pred [N, H, d_brain] (float32) and, if requested, the Spatial Mixer
        attention [N, H, nhead, K, K] (or None when the model has no attention mixer).
    """
    L = model.L
    device = Z.device
    amp_on = bool(use_amp and device.type == "cuda")
    ar_stim = torch.arange(-L + 1, 1, device=device)
    ar_hist = torch.arange(-L + 1, 0, device=device)

    preds, attns = [], []
    for c in range(0, len(anchors), batch_size):
        u0 = torch.as_tensor(anchors[c:c + batch_size], device=device, dtype=torch.long)
        Z_hist = Z[u0[:, None] + ar_hist]                                  # [B, L-1, D_f]
        steps, step_attn = [], []
        for h in range(H):
            sidx = (u0 + h)[:, None] + ar_stim - stim_offset               # [B, L]
            with torch.autocast(device_type="cuda", enabled=amp_on):
                z, aux = model(V[sidx], A[sidx], W[sidx], Z_hist,
                               return_attn=True, attn_mask=attn_mask)
            z = z.float()
            steps.append(z)
            if return_attn and aux.get("spatial_attn") is not None:
                step_attn.append(aux["spatial_attn"][:, 0].float())        # [B, nhead, K, K]
            Z_hist = torch.cat([Z_hist[:, 1:], z.unsqueeze(1)], dim=1)
        preds.append(torch.stack(steps, dim=1))
        if step_attn:
            attns.append(torch.stack(step_attn, dim=1))
    Z_pred = torch.cat(preds, dim=0) if preds else Z.new_zeros(0, H, Z.shape[1])
    if return_attn:
        return Z_pred, (torch.cat(attns, dim=0) if attns else None)
    return Z_pred


def gather_targets(X: torch.Tensor, anchors: Sequence[int], H: int) -> torch.Tensor:
    """Ground-truth frames X[u0 : u0 + H] for every anchor -> [N, H, D]."""
    u0 = torch.as_tensor(list(anchors), device=X.device, dtype=torch.long)
    return X[u0[:, None] + torch.arange(H, device=X.device)]


def select_anchors(T: int, L: int, H: int, stim_offset: int = 0, stride: int = 1,
                   first_only: bool = False) -> List[int]:
    from .data.datasets import run_anchors
    a = run_anchors(T, L, H, stim_offset, stride)
    return a[:1] if first_only else a
