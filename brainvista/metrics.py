# brainvista/metrics.py
"""Forecasting metrics (paper §4.1; Appendix A.10, C.6, C.7).

For predicted / ground-truth rollout segments Y_hat, Y in R^{H x D}:

    r          Pearson correlation of vec(Y_hat) and vec(Y); pooled over all
               evaluated segments of a subject (`r_segment` = per-segment mean)
    p_corr     1/H sum_t corr(Y_hat[t, :], Y[t, :])           (pattern correlation)
    Cls@Topk   the matched ground-truth segment ranks among the top-k of a
               same-subject candidate pool (N - 1 non-overlapping distractors,
               from different runs when available), ranked by correlation
    MSE, rho (Spearman), 2v2 identification, r_traj, mean/std drift, PSD distance

All metrics are computed per subject; summaries average over subjects.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

EPS = 1e-8


def rowcorr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along the last dimension."""
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    num = (a * b).sum(dim=-1)
    den = torch.sqrt((a * a).sum(dim=-1) * (b * b).sum(dim=-1)) + EPS
    return num / den


def _ranks(x: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()


def _unit_rows(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + EPS)


class _PooledCorr:
    """Running sums for a Pearson correlation over all elements seen."""

    def __init__(self, shape=()):
        self.s = {k: torch.zeros(shape, dtype=torch.float64) for k in ("x", "y", "xx", "yy", "xy")}
        self.n = 0

    def update(self, x: torch.Tensor, y: torch.Tensor, dims):
        x, y = x.double(), y.double()
        for k, v in (("x", x), ("y", y), ("xx", x * x), ("yy", y * y), ("xy", x * y)):
            self.s[k] += v.sum(dim=dims).cpu()
        n = 1
        for d in (dims if isinstance(dims, (tuple, list)) else (dims,)):
            n *= x.shape[d]
        self.n += n

    def value(self):
        if self.n == 0:
            return float("nan")
        n, s = self.n, self.s
        cov = s["xy"] / n - (s["x"] / n) * (s["y"] / n)
        vx = s["xx"] / n - (s["x"] / n) ** 2
        vy = s["yy"] / n - (s["y"] / n) ** 2
        r = cov / (torch.sqrt(vx.clamp_min(0) * vy.clamp_min(0)) + 1e-12)
        return r.numpy() if r.ndim else float(r)


class ForecastMetrics:
    """Accumulates metrics over the rollout segments of one subject at one horizon."""

    def __init__(self, network_names: Sequence[str], vox_splits: Sequence[int], H: int,
                 keep_segments: bool = True):
        self.names = list(network_names)
        self.splits = list(vox_splits)
        self.H = int(H)
        self.keep_segments = keep_segments
        P = sum(self.splits)
        self.n_seg = 0
        self.pooled = _PooledCorr()
        self.pooled_net = {n: _PooledCorr() for n in self.names}
        self.pooled_dim = _PooledCorr((P,))
        self.sums = {k: 0.0 for k in ("r_segment", "rho", "r_traj", "drift_mu", "drift_sigma", "psd")}
        self.pc_step = torch.zeros(self.H, dtype=torch.float64)
        self.pc_net = {n: 0.0 for n in self.names}
        self.sq_err, self.n_elem = 0.0, 0
        self.Q: List[torch.Tensor] = []
        self.G: List[torch.Tensor] = []
        self.seg_run: List[int] = []
        self.seg_u: List[int] = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, true: torch.Tensor, run_id: int = 0,
               anchors: Optional[Sequence[int]] = None, keep_mask: Optional[np.ndarray] = None):
        """pred, true: [N, H, P] in the same (normalized model) space."""
        pred, true = pred.float(), true.float()
        N, H, P = pred.shape
        assert H == self.H and P == sum(self.splits), (pred.shape, self.H, sum(self.splits))
        fp, ft = pred.reshape(N, -1), true.reshape(N, -1)

        self.n_seg += N
        self.pooled.update(fp, ft, dims=(0, 1))
        self.pooled_dim.update(pred.reshape(N * H, P), true.reshape(N * H, P), dims=0)
        self.sq_err += float(((pred - true) ** 2).double().sum())
        self.n_elem += pred.numel()
        self.pc_step += rowcorr(pred, true).double().sum(dim=0).cpu()
        self.sums["r_segment"] += float(rowcorr(fp, ft).sum())
        self.sums["rho"] += float(rowcorr(_ranks(fp), _ranks(ft)).sum())

        if H >= 2:
            pc = pred - pred.mean(dim=1, keepdim=True)
            tc = true - true.mean(dim=1, keepdim=True)
            r_dim = (pc * tc).sum(1) / (torch.sqrt((pc * pc).sum(1) * (tc * tc).sum(1)) + EPS)
            self.sums["r_traj"] += float(r_dim.mean(dim=1).sum())
        self.sums["drift_mu"] += float((pred.mean(2) - true.mean(2)).abs().mean(1).sum())
        self.sums["drift_sigma"] += float((pred.std(2, unbiased=False)
                                           - true.std(2, unbiased=False)).abs().mean(1).sum())

        psd = torch.zeros(N, device=pred.device)
        c = 0
        for name, d in zip(self.names, self.splits):
            p_k, t_k = pred[..., c:c + d], true[..., c:c + d]
            self.pc_net[name] += float(rowcorr(p_k, t_k).mean(dim=1).sum())
            self.pooled_net[name].update(p_k, t_k, dims=(0, 1, 2))
            if H >= 4:
                psd += _log_psd_distance(p_k, t_k)
            c += d
        self.sums["psd"] += float(psd.sum()) / max(1, len(self.names))

        if self.keep_segments:
            keep = np.ones(N, dtype=bool) if keep_mask is None else keep_mask
            if keep.any():
                idx = torch.as_tensor(np.nonzero(keep)[0], device=pred.device)
                self.Q.append(_unit_rows(fp[idx]).half().cpu())
                self.G.append(_unit_rows(ft[idx]).half().cpu())
                anchors = list(range(N)) if anchors is None else list(anchors)
                self.seg_run += [run_id] * int(keep.sum())
                self.seg_u += [anchors[i] for i in np.nonzero(keep)[0]]

    def similarity(self, device=None, chunk: int = 1024) -> torch.Tensor:
        """S_ij = corr(pred segment i, ground-truth segment j)."""
        Q, G = torch.cat(self.Q), torch.cat(self.G)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if str(device).startswith("cuda") else torch.float32
        Gd = G.to(device, dt)
        rows = [(Q[i:i + chunk].to(device, dt) @ Gd.T).float().cpu() for i in range(0, Q.shape[0], chunk)]
        return torch.cat(rows)

    def identification(self, pool_size: int = 200, ks: Sequence[int] = (10,), n_pairs: int = 5000,
                       seed: int = 0, device=None) -> Dict[str, float]:
        if not self.Q:
            return {}
        S = self.similarity(device).numpy()
        n = S.shape[0]
        runs, u = np.asarray(self.seg_run), np.asarray(self.seg_u)
        rng = np.random.default_rng(seed)
        diag = np.diag(S)
        hits = {k: 0 for k in ks}
        pools = []
        by_run = {r: np.nonzero(runs != r)[0] for r in np.unique(runs)}
        for i in range(n):
            other = by_run[runs[i]]
            if len(other) < pool_size - 1:
                same = np.nonzero((runs == runs[i]) & (np.abs(u - u[i]) >= self.H))[0]
                other = np.concatenate([other, same])
            m = min(pool_size - 1, len(other))
            cand = rng.choice(other, size=m, replace=False) if m > 0 else np.zeros(0, dtype=int)
            rank = int((S[i, cand] > diag[i]).sum())
            pools.append(m + 1)
            for k in ks:
                hits[k] += rank < k
        out = {f"cls_top{k}": hits[k] / n for k in ks}
        out["cls_pool_size"] = float(np.mean(pools))
        for k in ks:
            out[f"cls_top{k}_chance"] = min(1.0, k / out["cls_pool_size"])

        # Two-versus-two identification on segment pairs that do not overlap in time.
        wins, total = 0.0, 0
        if n >= 2:
            I = rng.integers(0, n, size=4 * n_pairs)
            J = rng.integers(0, n, size=4 * n_pairs)
            ok = (I != J) & ((runs[I] != runs[J]) | (np.abs(u[I] - u[J]) >= self.H))
            I, J = I[ok][:n_pairs], J[ok][:n_pairs]
            left = S[I, I] + S[J, J]
            right = S[I, J] + S[J, I]
            wins = float((left > right).sum() + 0.5 * (left == right).sum())
            total = len(I)
        out["2v2"] = wins / total if total else float("nan")
        out["n_id_segments"] = n
        return out

    def finalize(self, identification: bool = True, **id_kwargs) -> Dict:
        n = max(1, self.n_seg)
        res = {
            "n_segments": self.n_seg,
            "r": self.pooled.value(),
            "p_corr": float(self.pc_step.sum() / (n * self.H)),
            "mse": self.sq_err / max(1, self.n_elem),
            "r_segment": self.sums["r_segment"] / n,
            "rho": self.sums["rho"] / n,
            "r_traj": self.sums["r_traj"] / n if self.H >= 2 else float("nan"),
            "drift_mu": self.sums["drift_mu"] / n,
            "drift_sigma": self.sums["drift_sigma"] / n,
            "psd_dist": self.sums["psd"] / n if self.H >= 4 else float("nan"),
            "p_corr_per_step": (self.pc_step / n).tolist(),
            "per_network": {name: {"p_corr": self.pc_net[name] / n, "r": self.pooled_net[name].value()}
                            for name in self.names},
        }
        if identification and self.keep_segments:
            res.update(self.identification(**id_kwargs))
        return res

    def per_dim_r(self) -> np.ndarray:
        """Pooled temporal correlation per parcel / voxel / component (rollout fidelity map)."""
        return np.asarray(self.pooled_dim.value(), dtype=np.float32)


def _log_psd_distance(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """L1 distance between normalized log power spectra (time axis 1), averaged over dims first."""
    def spec(x):
        x = x - x.mean(dim=1, keepdim=True)
        P = (torch.fft.rfft(x, dim=1).abs() ** 2).mean(dim=2)            # [N, F]
        P = P / (P.sum(dim=1, keepdim=True) + EPS)
        return torch.log(P + 1e-8)
    return (spec(p) - spec(t)).abs().sum(dim=1)


def summarize_subjects(per_subject: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """Mean and std across subjects of every scalar metric."""
    keys = sorted({k for r in per_subject.values() for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)})
    mean, std = {}, {}
    for k in keys:
        vals = np.array([r[k] for r in per_subject.values() if k in r], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        mean[k] = float(vals.mean()) if vals.size else float("nan")
        std[k] = float(vals.std()) if vals.size else float("nan")
    return {"mean": mean, "std": std, "n_subjects": len(per_subject)}
