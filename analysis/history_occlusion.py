#!/usr/bin/env python3
# analysis/history_occlusion.py
"""History-token occlusion (paper §4.4, Fig. 5a; Appendix F.2).

For every lag l = 1..L-1, the observed brain token at time u - l is hidden from
all attention (every other history token and the observed stimuli are kept),
and the increase of the one-step L2 prediction error is measured per
functional network, relative to the unoccluded model:

    delta[k, l] = mean_u || x_hat^{u,k}_{occ l} - x^{u,k} ||^2 - || x_hat^{u,k} - x^{u,k} ||^2

This measures the trained model's conditional predictive reliance on each
lag; it is a model-based diagnostic, not a biological intervention.
Outputs occlusion_<subj>.npy ([K, L-1]), occlusion.json and an optional heatmap.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brainvista.checkpoint import decode_tokens  # noqa: E402
from brainvista.cli import add_data_args, add_eval_data_args, add_runtime_args, add_stimulus_args  # noqa: E402
from brainvista.data.datasets import run_anchors  # noqa: E402
from brainvista.evaluation import load_eval_context  # noqa: E402
from brainvista.rollout import gather_targets, rollout, run_to_device  # noqa: E402
from brainvista.utils import save_json  # noqa: E402


def per_network_sq_err(X_pred, X_true, splits):
    out, c = [], 0
    for d in splits:
        out.append(((X_pred[..., c:c + d] - X_true[..., c:c + d]) ** 2).mean(-1))
        c += d
    return torch.stack(out, -1)                                         # [N, 1, K]


@torch.no_grad()
def main(args):
    gt_space = "fmri" if args.fmri_root else "recon"
    ctx = load_eval_context(args, need_fmri=gt_space == "fmri")
    model, L = ctx.model, ctx.model.L
    lags = list(range(1, L))
    masks = {l: model.occlusion_mask(l) for l in lags}
    use_amp = not args.no_amp and ctx.device.type == "cuda"
    stim_offset = int(args.stim_offset)

    sums, counts = {}, {}
    for run in ctx.runs:
        anchors = run_anchors(run.T, L, 1, stim_offset, args.anchor_stride)
        if not anchors:
            continue
        V, A, W, Z = run_to_device(run, ctx.device)
        X = (torch.from_numpy(run.X).to(ctx.device) if gt_space == "fmri" else None)
        X_true = (gather_targets(X, anchors, 1) if X is not None
                  else decode_tokens(gather_targets(Z, anchors, 1), ctx.bundles, args.parcels))

        def err(mask):
            Zp = rollout(model, V, A, W, Z, anchors, 1, stim_offset, args.batch_size, use_amp, attn_mask=mask)
            return per_network_sq_err(decode_tokens(Zp, ctx.bundles, args.parcels), X_true, ctx.vox_splits)

        base = err(None)
        delta = torch.stack([(err(masks[l]) - base).mean(dim=(0, 1)) for l in lags], dim=1)   # [K, L-1]
        sums[run.subj] = sums.get(run.subj, 0) + delta.double().cpu() * len(anchors)
        counts[run.subj] = counts.get(run.subj, 0) + len(anchors)

    os.makedirs(args.out_dir, exist_ok=True)
    result = {"lags": lags, "networks": args.parcels, "gt_space": gt_space, "subjects": {}}
    for subj in sums:
        D = (sums[subj] / counts[subj]).numpy()
        np.save(os.path.join(args.out_dir, f"occlusion_{subj}.npy"), D)
        result["subjects"][subj] = {"n_anchors": counts[subj], "delta": D.tolist()}
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 0.45 * len(args.parcels) + 1.5))
            im = ax.imshow(D, aspect="auto", cmap="magma", extent=[0.5, L - 0.5, len(args.parcels) - 0.5, -0.5])
            ax.set_yticks(range(len(args.parcels)), args.parcels)
            ax.set_xlabel("history lag (TRs before the queried step)")
            ax.set_title(f"History-token occlusion: increase in L2 error ({subj})")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"occlusion_{subj}.png"), dpi=150)
            plt.close(fig)
    save_json(result, os.path.join(args.out_dir, "occlusion.json"))
    print(f"[Occlusion] {len(sums)} subjects -> {args.out_dir}")


def parse():
    p = argparse.ArgumentParser("BrainVista history-token occlusion")
    add_data_args(p, inherit=True)
    add_stimulus_args(p, inherit=True)
    add_eval_data_args(p)
    add_runtime_args(p)
    p.add_argument("--anchor_stride", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out_dir", default="./analysis_output/occlusion")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
