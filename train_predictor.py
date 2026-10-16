#!/usr/bin/env python3
# train_predictor.py
"""BrainVista stage 2: next-token whole-brain latent prediction (paper §3.5, Eq. 15).

With the network-wise tokenizers frozen, the temporal Transformer + Spatial
Mixer Head is trained with teacher forcing,

    L_pred = sum_t || tau_f^{t+1} - tau_hat_f^{t+1} ||_2^2 ,

where the input holds brain tokens up to t and stimulus tokens up to t+1 and the
target token is supervision only. After every epoch the model is rolled out
for H steps on the validation runs (never the test runs) and the checkpoint
with the best validation score is kept.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from brainvista.checkpoint import load_all_tokenizers
from brainvista.cli import add_data_args, add_model_args, add_runtime_args, add_stimulus_args
from brainvista.data.datasets import WindowDataset, collate_windows
from brainvista.data.io import parse_run_list, runs_by_subject
from brainvista.data.runs import load_runs
from brainvista.evaluation import evaluate_runs, finalize_all, mean_over_subjects
from brainvista.models.predictor import BrainVista, count_parameters
from brainvista.utils import dump_config, enable_tf32, get_device, set_seed


def build_model(args, Dv, Da, Dw, Dz, z_splits) -> BrainVista:
    return BrainVista(
        Dv=Dv, Da=Da, Dw=Dw, Dz_in=Dz, Dz_out=Dz,
        d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout,
        L=args.L, z_splits=z_splits,
        spatial_nhead=args.spatial_nhead, spatial_nlayers=args.spatial_nlayers,
        spatial_dropout=args.spatial_dropout, spatial_dz=args.spatial_dz,
        mixer=args.mixer, mask=args.mask, input_mode=args.input_mode,
        brain_self_attn=bool(args.brain_self_attn),
    )


def param_groups(model: nn.Module, weight_decay: float):
    """AdamW groups: no weight decay on LayerNorm parameters or biases (Appendix A.9)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def subsample_units(run_list, fraction: float, seed: int):
    """Data-scaling experiments: keep a fraction of complete training units per subject."""
    if fraction >= 1.0:
        return run_list
    rng = np.random.default_rng(seed)
    keep = []
    for subj, rids in runs_by_subject(run_list).items():
        n = max(1, int(round(fraction * len(rids))))
        keep += [(subj, r) for r in sorted(rng.choice(rids, size=n, replace=False))]
    return keep


@torch.no_grad()
def one_step_loss(model, loader, device, use_amp):
    model.eval()
    tot, n = 0.0, 0
    for V, A, W, Z_hist, z_tgt in loader:
        V, A, W, Z_hist, z_tgt = (t.to(device, non_blocking=True) for t in (V, A, W, Z_hist, z_tgt))
        with torch.autocast(device_type="cuda", enabled=use_amp):
            z_pred = model(V, A, W, Z_hist)
        tot += F.mse_loss(z_pred.float(), z_tgt, reduction="sum").item()
        n += z_tgt.numel()
    return tot / max(1, n)


def save_ckpt(path, model, args, epoch, z_splits, dims, val_scores):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "cfg": vars(args),
        "z_dim": sum(z_splits),
        "z_splits": z_splits,
        "dims": dims,
        "parcels": args.parcels,
        "tokenizer_ckpt_dir": args.tokenizer_ckpt_dir,
        "tokenizer_name_template": args.tokenizer_name_template,
        "val": val_scores,
    }, path)


def train(args):
    device = get_device(args.cpu)
    use_amp = not args.no_amp and device.type == "cuda"

    bundles, z_splits, vox_splits = load_all_tokenizers(
        args.parcels, args.tokenizer_ckpt_dir, args.tokenizer_name_template, device)

    train_list = subsample_units(parse_run_list(args.train_list_file, args.subjects),
                                 args.train_fraction, args.seed)
    val_list = parse_run_list(args.val_list_file, args.subjects)
    if args.test_list_file:
        test_runs = set(parse_run_list(args.test_list_file, args.subjects))
        overlap = test_runs & set(val_list)
        if overlap:
            raise ValueError(f"Validation and test runs overlap ({len(overlap)} runs); "
                             "model selection must not use test runs.")
    print(f"[Data] {len(train_list)} train / {len(val_list)} val runs")

    gt_space = args.val_gt_space if args.fmri_root else "recon"
    tr_runs = load_runs(args, train_list)
    va_runs = load_runs(args, val_list, bundles, need_fmri=gt_space == "fmri")
    ds_tr = WindowDataset(tr_runs, args.L, args.stim_offset)
    ds_va = WindowDataset(va_runs, args.L, args.stim_offset, stride=args.val_anchor_stride)
    if len(ds_tr) == 0:
        raise RuntimeError("No training anchors. Check paths, run lists and --L.")
    if ds_tr.Dz != sum(z_splits):
        raise ValueError(f"Token dim mismatch: data d_brain={ds_tr.Dz}, tokenizers sum(d_k)={sum(z_splits)}")
    dims = {"Dv": ds_tr.Dv, "Da": ds_tr.Da, "Dw": ds_tr.Dw, "Dz": ds_tr.Dz,
            "z_splits": z_splits, "vox_splits": vox_splits}

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0,
                     prefetch_factor=4 if args.workers > 0 else None, collate_fn=collate_windows)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       drop_last=len(ds_tr) > args.batch_size, **loader_kw)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size * 4, shuffle=False, **loader_kw)

    model = build_model(args, ds_tr.Dv, ds_tr.Da, ds_tr.Dw, ds_tr.Dz, z_splits).to(device)
    n_params = count_parameters(model)
    print(f"[Model] BrainVista mask={args.mask} mixer={args.mixer} input={args.input_mode} "
          f"| trainable params {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    dump_config(vars(args), os.path.join(args.save_dir, "predictor_cfg.json"),
                dims=dims, trainable_params=n_params, n_train_anchors=len(ds_tr),
                n_val_runs=len(va_runs))
    log_path = os.path.join(args.save_dir, "train_log.jsonl")

    best = -float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n_step = 0.0, 0
        pbar = tqdm(dl_tr, desc=f"[BrainVista] ep{ep:02d}", ncols=110, leave=False, disable=None)
        for V, A, W, Z_hist, z_tgt in pbar:
            V, A, W, Z_hist, z_tgt = (t.to(device, non_blocking=True) for t in (V, A, W, Z_hist, z_tgt))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                z_pred = model(V, A, W, Z_hist)
            loss = F.mse_loss(z_pred.float(), z_tgt)                         # Eq. 15
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item(); n_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        sched.step()

        val_loss = one_step_loss(model, dl_va, device, use_amp) if len(ds_va) else float("nan")
        per_subj = finalize_all(evaluate_runs(
            model, va_runs, bundles, args.parcels, vox_splits, args.val_horizon,
            stim_offset=args.stim_offset, anchor_stride=args.val_anchor_stride,
            batch_size=args.eval_batch_size, use_amp=use_amp, gt_space=gt_space))
        scores = {"loss": -val_loss,
                  "p_corr": mean_over_subjects(per_subj, "p_corr"),
                  "r": mean_over_subjects(per_subj, "r")}
        rec = {"epoch": ep, "train_loss": run_loss / max(1, n_step), "val_loss": val_loss,
               f"val_p_corr_H{args.val_horizon}": scores["p_corr"],
               f"val_r_H{args.val_horizon}": scores["r"], "lr": opt.param_groups[0]["lr"]}
        print(f"[BrainVista] ep{ep:02d} | train_L={rec['train_loss']:.5f} | val_L={val_loss:.5f} "
              f"| val p_corr(H={args.val_horizon})={scores['p_corr']:.4f} | val r={scores['r']:.4f}")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if scores[args.select_metric] > best:
            best = scores[args.select_metric]
            save_ckpt(os.path.join(args.save_dir, "predictor_best.pt"), model, args, ep,
                      z_splits, dims, rec)
            print(f"  -> new best val {args.select_metric} = {best:.5f}; saved")

    save_ckpt(os.path.join(args.save_dir, "predictor_final.pt"), model, args, args.epochs,
              z_splits, dims, rec)
    print("Training finished.")


def parse():
    p = argparse.ArgumentParser("BrainVista stage 2: next-token whole-brain latent prediction")
    add_data_args(p)
    add_stimulus_args(p)
    add_model_args(p)
    add_runtime_args(p)
    g = p.add_argument_group("training")
    g.add_argument("--train_list_file", type=str, required=True)
    g.add_argument("--val_list_file", type=str, required=True,
                   help="Validation runs used for model selection (official split).")
    g.add_argument("--test_list_file", type=str, default=None,
                   help="Optional; only checked for overlap with the validation runs.")
    g.add_argument("--train_fraction", type=float, default=1.0,
                   help="Fraction of complete training units per subject (data scaling).")
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--eval_batch_size", type=int, default=256)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--weight_decay", type=float, default=1e-2)
    g.add_argument("--grad_clip", type=float, default=0.5)
    g.add_argument("--val_horizon", "--K_horizon_val", dest="val_horizon", type=int, default=20)
    g.add_argument("--val_anchor_stride", type=int, default=5,
                   help="Stride over validation anchors for the per-epoch rollout check.")
    g.add_argument("--val_gt_space", type=str, default="fmri", choices=["fmri", "recon"])
    g.add_argument("--select_metric", type=str, default="p_corr", choices=["p_corr", "r", "loss"])
    g.add_argument("--save_dir", type=str, default="./checkpoints/predictor")
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    set_seed(args.seed)
    enable_tf32()
    train(args)
