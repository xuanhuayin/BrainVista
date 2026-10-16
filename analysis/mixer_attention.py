#!/usr/bin/env python3
# analysis/mixer_attention.py
"""Spatial Mixer attention summary (paper §4.4, Fig. 5b; Appendix F.3).

Rolls the model out on held-out runs and averages the K x K Spatial Mixer
attention over anchors, rollout steps and heads, giving one network-to-network
matrix per subject (row = query network, column = attended network). The
matrix summarizes the fitted model; it is not a connectivity estimate.
Outputs mixer_attention_<subj>.npy, mixer_attention.json and optional heatmaps.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brainvista.cli import add_data_args, add_eval_data_args, add_runtime_args, add_stimulus_args  # noqa: E402
from brainvista.data.datasets import run_anchors  # noqa: E402
from brainvista.evaluation import load_eval_context  # noqa: E402
from brainvista.rollout import rollout, run_to_device  # noqa: E402
from brainvista.utils import save_json  # noqa: E402


@torch.no_grad()
def main(args):
    ctx = load_eval_context(args, need_fmri=False)
    model = ctx.model
    if model.spatial_mixer is None or model.spatial_mixer.mixer != "attn":
        raise ValueError("This checkpoint has no attention Spatial Mixer.")
    use_amp = not args.no_amp and ctx.device.type == "cuda"
    sums, counts = {}, {}
    for run in ctx.runs:
        anchors = run_anchors(run.T, model.L, args.horizon, int(args.stim_offset), args.anchor_stride)
        if not anchors:
            continue
        V, A, W, Z = run_to_device(run, ctx.device)
        _, attn = rollout(model, V, A, W, Z, anchors, args.horizon, int(args.stim_offset),
                          args.batch_size, use_amp, return_attn=True)           # [N, H, nhead, K, K]
        sums[run.subj] = sums.get(run.subj, 0) + attn.double().sum(dim=(0, 1, 2)).cpu()
        counts[run.subj] = counts.get(run.subj, 0) + attn.shape[0] * attn.shape[1] * attn.shape[2]

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"networks": args.parcels, "horizon": args.horizon, "subjects": {}}
    for subj in sums:
        M = (sums[subj] / counts[subj]).numpy()
        np.save(os.path.join(args.out_dir, f"mixer_attention_{subj}.npy"), M)
        out["subjects"][subj] = M.tolist()
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 4.8))
            im = ax.imshow(M, cmap="viridis")
            ax.set_xticks(range(len(args.parcels)), args.parcels, rotation=45, ha="right")
            ax.set_yticks(range(len(args.parcels)), args.parcels)
            ax.set_xlabel("attended network"); ax.set_ylabel("query network")
            ax.set_title(f"Spatial Mixer attention ({subj})")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"mixer_attention_{subj}.png"), dpi=150)
            plt.close(fig)
    save_json(out, os.path.join(args.out_dir, "mixer_attention.json"))
    print(f"[Mixer attention] {len(sums)} subjects -> {args.out_dir}")


def parse():
    p = argparse.ArgumentParser("BrainVista Spatial Mixer attention")
    add_data_args(p, inherit=True)
    add_stimulus_args(p, inherit=True)
    add_eval_data_args(p)
    add_runtime_args(p)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--anchor_stride", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out_dir", default="./analysis_output/mixer_attention")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
