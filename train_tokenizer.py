#!/usr/bin/env python3
# train_tokenizer.py
"""BrainVista stage 1: network-wise tokenizer pretraining (paper §3.5, Eq. 14).

Trains one deterministic MLP autoencoder (h_phi^k, g_psi^k) for a single
functional network k with reconstruction loss only,

    L_tok = sum_t || x_bar^{k,t} - g_psi^k(h_phi^k(x_bar^{k,t})) ||_2^2 ,

where x_bar is normalized with training-run statistics (Appendix A.4). The
best checkpoint by validation reconstruction error is kept together with the
per-subject normalization statistics; run it once per network (see
scripts/train_tokenizer.sh).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from brainvista.checkpoint import stats_to_torch
from brainvista.data.datasets import TokenizerDataset
from brainvista.data.io import parse_run_list, runs_by_subject
from brainvista.data.normalization import NORM_MODES, compute_train_stats
from brainvista.models.tokenizer import Dec, EncAE, default_token_dim
from brainvista.utils import dump_config, enable_tf32, get_device, save_json, set_seed


def masked_mse(x_hat, x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (((x_hat - x) ** 2) * m).sum() / (m.sum() * x.shape[-1]).clamp_min(1.0)


@torch.no_grad()
def reconstruction_diagnostics(enc, dec, loader, device):
    """Validation recon MSE, mean per-dimension Pearson r and explained variance."""
    n, sse, s1, s2, sx, sy, sxx, syy, sxy = 0, 0.0, 0, 0, 0, 0, 0, 0, 0
    for x, mask in loader:
        x = x.to(device)[mask.to(device)]                     # valid frames [n, P]
        x_hat = dec(enc(x))
        xd, yd = x.double(), x_hat.double()
        n += x.shape[0]
        sse += float(((yd - xd) ** 2).sum())
        s1 = s1 + xd.sum(0); s2 = s2 + (xd ** 2).sum(0)
        sx = sx + xd.sum(0); sy = sy + yd.sum(0)
        sxx = sxx + (xd ** 2).sum(0); syy = syy + (yd ** 2).sum(0); sxy = sxy + (xd * yd).sum(0)
    if n == 0:
        return {"mse": float("nan"), "r": float("nan"), "explained_variance": float("nan")}
    P = s1.numel()
    sst = float((s2 - s1 ** 2 / n).sum())
    cov = sxy / n - (sx / n) * (sy / n)
    r = cov / torch.sqrt((sxx / n - (sx / n) ** 2).clamp_min(0) * (syy / n - (sy / n) ** 2).clamp_min(0) + 1e-12)
    return {"mse": sse / (n * P), "r": float(r.mean()), "explained_variance": 1.0 - sse / max(sst, 1e-12),
            "n_frames": n}


def train(args):
    device = get_device(args.cpu)
    train_list = parse_run_list(args.train_list_file, args.subjects)
    val_list = parse_run_list(args.val_list_file, args.subjects)
    print(f"[Tokenizer:{args.network}] {len(train_list)} train / {len(val_list)} val runs")

    # Training-set normalization statistics per subject (never from val/test runs).
    stats = {}
    for subj, rids in runs_by_subject(train_list).items():
        stats[subj] = compute_train_stats(args.fmri_root, args.network, subj, rids)
        print(f"  [norm] {subj}: {stats[subj]['n_frames']} training frames")

    ds_tr = TokenizerDataset(args.fmri_root, args.network, train_list, stats, args.norm_mode, args.seg_len)
    ds_va = TokenizerDataset(args.fmri_root, args.network, val_list, stats, args.norm_mode, args.seg_len,
                             stride=args.val_stride)
    if len(ds_tr) == 0:
        raise RuntimeError("No training segments; check --fmri_root and the run lists.")
    Db = ds_tr.Db
    z_dim = args.z_dim if args.z_dim > 0 else default_token_dim(Db, args.z_ratio)
    print(f"[Tokenizer:{args.network}] P_k={Db} -> d_k={z_dim} | "
          f"{len(ds_tr)} train / {len(ds_va)} val segments")

    loader_kw = dict(num_workers=args.workers, pin_memory=True,
                     persistent_workers=args.workers > 0,
                     prefetch_factor=4 if args.workers > 0 else None)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       drop_last=len(ds_tr) > args.batch_size, **loader_kw)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, **loader_kw)

    enc = EncAE(Db, z_dim, args.hidden, args.dropout).to(device)
    dec = Dec(z_dim, Db, args.hidden, args.dropout).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    cfg = dict(vars(args), z_dim=z_dim, Db=Db)
    dump_config(cfg, os.path.join(args.save_dir, "tokenizer_cfg.json"))

    best, best_diag = float("inf"), None
    for ep in range(1, args.epochs + 1):
        enc.train(); dec.train()
        tr_loss, nstep = 0.0, 0
        for x, mask in tqdm(dl_tr, desc=f"[Tok:{args.network}] ep{ep:02d}", ncols=110, leave=False, disable=None):
            x, mask = x.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = masked_mse(dec(enc(x)), x, mask)                 # Eq. 14
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optim)
            scaler.update()
            tr_loss += loss.item(); nstep += 1
        sched.step()

        enc.eval(); dec.eval()
        diag = reconstruction_diagnostics(enc, dec, dl_va, device)
        print(f"[Tok:{args.network}] ep{ep:02d} | train_rec={tr_loss / max(1, nstep):.5f} "
              f"| val_rec={diag['mse']:.5f} | val_r={diag['r']:.4f} | val_EV={diag['explained_variance']:.4f}")

        if diag["mse"] < best:
            best, best_diag = diag["mse"], dict(diag, epoch=ep)
            torch.save({
                "enc": enc.state_dict(), "dec": dec.state_dict(),
                "Db": Db, "z_dim": z_dim, "network": args.network, "epoch": ep,
                "args": cfg,
                "norm": {"mode": args.norm_mode, "stats": stats_to_torch(stats)},
                "val_diagnostics": diag,
            }, os.path.join(args.save_dir, "best.pt"))
            print("  -> saved best tokenizer")

    save_json(best_diag, os.path.join(args.save_dir, "recon_diagnostics.json"))
    print(f"[Tokenizer:{args.network}] best val recon: {best_diag}")


def parse():
    p = argparse.ArgumentParser("BrainVista stage 1: network-wise tokenizer (Eq. 14)")
    p.add_argument("--fmri_root", type=str, required=True,
                   help="Model-space fMRI root: FMRI_ROOT/<network>/<subj>/<rid>.npy.")
    p.add_argument("--network", "--parcel_name", dest="network", type=str, required=True,
                   help="Functional network / ROI group k (e.g. Vis).")
    p.add_argument("--subjects", nargs="+", default=["sub-01"])
    p.add_argument("--train_list_file", type=str, required=True)
    p.add_argument("--val_list_file", type=str, required=True)
    p.add_argument("--norm_mode", type=str, default="train", choices=NORM_MODES,
                   help="train: training-set statistics (paper default); run: run-wise diagnostic.")
    p.add_argument("--z_dim", type=int, default=0, help="Token dim d_k; 0 = round(z_ratio * P_k).")
    p.add_argument("--z_ratio", type=float, default=0.8)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seg_len", type=int, default=64)
    p.add_argument("--val_stride", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    set_seed(args.seed)
    enable_tf32()
    train(args)
