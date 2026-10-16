#!/usr/bin/env python3
# inference.py
"""BrainVista long-horizon autoregressive rollout inference (paper Eq. 2).

Saves, for every run, the predicted and ground-truth trajectories in the
normalized model space, the K x K Spatial Mixer attention at every step, and
per-run r / p_corr:

    <out_dir>/<subj>/<rid>_pred.npy           [N, H, P]
    <out_dir>/<subj>/<rid>_true.npy           [N, H, P]
    <out_dir>/<subj>/<rid>_anchors.npy        [N]   start times u0 (fMRI bins)
    <out_dir>/<subj>/<rid>_spatial_attn.npy   [N, H, nhead, K, K]
    <out_dir>/inference_summary.json

Anchor selection: `--anchors first` (one rollout from the earliest valid
anchor), `tile` (non-overlapping rollouts covering the run, e.g. for clip-level
analyses) or `all` (every valid anchor, optionally strided).
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from brainvista.checkpoint import decode_tokens
from brainvista.cli import add_data_args, add_eval_data_args, add_runtime_args, add_stimulus_args
from brainvista.data.datasets import run_anchors
from brainvista.evaluation import load_eval_context
from brainvista.metrics import rowcorr
from brainvista.rollout import gather_targets, rollout, run_to_device
from brainvista.utils import save_json, set_seed


@torch.no_grad()
def main(args):
    set_seed(args.seed)
    gt_space = "fmri" if args.fmri_root else "recon"
    ctx = load_eval_context(args, need_fmri=gt_space == "fmri")
    model, H = ctx.model, args.horizon
    use_amp = not args.no_amp and ctx.device.type == "cuda"
    stim_offset = int(args.stim_offset)

    results = []
    for run in tqdm(ctx.runs, desc="[BrainVista inference]", ncols=110, disable=None):
        stride = {"first": 1, "tile": H, "all": args.anchor_stride}[args.anchors]
        anchors = run_anchors(run.T, model.L, H, stim_offset, stride)
        if args.anchors == "first":
            anchors = anchors[:1]
        if not anchors:
            print(f"[Skip] {run.subj}/{run.rid}: T={run.T} too short for L={model.L}, H={H}")
            continue
        V, A, W, Z = run_to_device(run, ctx.device)
        preds, trues, attns = [], [], []
        for c in range(0, len(anchors), args.batch_size):
            a = anchors[c:c + args.batch_size]
            Z_pred, attn = rollout(model, V, A, W, Z, a, H, stim_offset, args.batch_size,
                                   use_amp, return_attn=True)
            preds.append(decode_tokens(Z_pred, ctx.bundles, args.parcels))
            if gt_space == "fmri":
                trues.append(gather_targets(torch.from_numpy(run.X).to(ctx.device), a, H))
            else:
                trues.append(decode_tokens(gather_targets(Z, a, H), ctx.bundles, args.parcels))
            if attn is not None:
                attns.append(attn.cpu())
        X_pred, X_true = torch.cat(preds), torch.cat(trues)

        r = float(rowcorr(X_pred.reshape(len(anchors), -1), X_true.reshape(len(anchors), -1)).mean())
        p_corr = float(rowcorr(X_pred, X_true).mean())
        out = os.path.join(args.out_dir, run.subj)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, f"{run.rid}_pred.npy"), X_pred.cpu().numpy().astype(np.float32))
        np.save(os.path.join(out, f"{run.rid}_true.npy"), X_true.cpu().numpy().astype(np.float32))
        np.save(os.path.join(out, f"{run.rid}_anchors.npy"), np.asarray(anchors, dtype=np.int64))
        if attns:
            np.save(os.path.join(out, f"{run.rid}_spatial_attn.npy"),
                    torch.cat(attns).numpy().astype(np.float32))
        results.append({"subj": run.subj, "rid": run.rid, "n_rollouts": len(anchors), "H": H,
                        "r": r, "p_corr": p_corr})

    summary = {"num_runs": len(results), "H": H, "gt_space": gt_space, "anchors": args.anchors,
               "parcels": args.parcels, "vox_splits": ctx.vox_splits,
               "mean_r": float(np.mean([x["r"] for x in results])) if results else float("nan"),
               "mean_p_corr": float(np.mean([x["p_corr"] for x in results])) if results else float("nan"),
               "runs": results}
    save_json(summary, os.path.join(args.out_dir, "inference_summary.json"))
    print(f"[Summary] {len(results)} runs | mean r={summary['mean_r']:.4f} "
          f"| mean p_corr={summary['mean_p_corr']:.4f} -> {args.out_dir}")


def parse():
    p = argparse.ArgumentParser("BrainVista inference: long-horizon autoregressive rollout")
    add_data_args(p, inherit=True)
    add_stimulus_args(p, inherit=True)
    add_eval_data_args(p)
    add_runtime_args(p)
    g = p.add_argument_group("inference")
    g.add_argument("--horizon", "--K_horizon", dest="horizon", type=int, default=20)
    g.add_argument("--anchors", type=str, default="first", choices=["first", "tile", "all"])
    g.add_argument("--anchor_stride", type=int, default=1, help="Stride for --anchors all.")
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--out_dir", type=str, default="./predictions")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
