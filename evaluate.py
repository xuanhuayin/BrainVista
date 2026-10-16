#!/usr/bin/env python3
# evaluate.py
"""BrainVista test protocol: conditional autoregressive rollout (paper §4.1, Appendix A).

For every held-out test run and every valid anchor u0 (N^H_r = T_r - m - H + 1
per run), the model is initialized with observed fMRI context, rolled out H
steps with predicted brain tokens fed back, and the stimulus stream observed up
to each queried step. Predictions are decoded by the frozen detokenizers and
scored against the normalized fMRI. Metrics are computed per subject and then
averaged over subjects:

    r, p_corr, Cls@Top10            (main tables)
    MSE, Spearman rho, 2v2          (complementary metrics)
    r_traj, drift, PSD distance     (trajectory / spectral diagnostics)
    per-network p_corr and r        (Yeo-7 breakdown), per-step p_corr curve
"""
import argparse
import os

import numpy as np

from brainvista.cli import add_data_args, add_eval_data_args, add_runtime_args, add_stimulus_args
from brainvista.evaluation import evaluate_runs, load_eval_context
from brainvista.metrics import summarize_subjects
from brainvista.utils import save_json, set_seed

MAIN_KEYS = ("r", "p_corr", "cls_top10", "mse", "rho", "2v2", "r_traj", "drift_mu", "drift_sigma", "psd_dist")


def main(args):
    set_seed(args.seed)
    gt_space = args.gt_space
    if gt_space == "fmri" and not args.fmri_root:
        print("[Eval] --fmri_root not given: scoring against detokenized ground-truth tokens (recon).")
        gt_space = "recon"
    ctx = load_eval_context(args, need_fmri=gt_space == "fmri")
    use_amp = not args.no_amp and ctx.device.type == "cuda"
    stim_offset = int(args.stim_offset)

    summary = {"predictor_ckpt": args.predictor_ckpt, "gt_space": gt_space,
               "stim_offset": stim_offset, "L": ctx.model.L,
               "parcels": args.parcels, "vox_splits": ctx.vox_splits,
               "n_runs": len(ctx.runs), "horizons": {}}
    os.makedirs(args.out_dir, exist_ok=True)

    for H in args.horizons:
        acc = evaluate_runs(ctx.model, ctx.runs, ctx.bundles, args.parcels, ctx.vox_splits, H,
                            stim_offset=stim_offset, anchor_stride=args.anchor_stride,
                            batch_size=args.batch_size, use_amp=use_amp, gt_space=gt_space,
                            keep_segments=not args.no_identification, id_stride=args.id_stride,
                            verbose=True)
        per_subj = {}
        for subj, m in sorted(acc.items()):
            per_subj[subj] = m.finalize(
                identification=not args.no_identification,
                pool_size=args.cls_pool_size, ks=args.cls_ks,
                n_pairs=args.two_v_two_pairs, seed=args.seed)
            if args.save_per_dim:
                np.save(os.path.join(args.out_dir, f"per_dim_r_{subj}_H{H}.npy"), m.per_dim_r())
        agg = summarize_subjects(per_subj)
        summary["horizons"][f"H{H}"] = {"subjects": per_subj, **agg}

        print(f"\n===== H = {H} | {agg['n_subjects']} subjects | gt = {gt_space} =====")
        for k in MAIN_KEYS + tuple(f"cls_top{k}" for k in args.cls_ks if k != 10):
            if k in agg["mean"]:
                print(f"  {k:<12s} {agg['mean'][k]:.4f} +/- {agg['std'][k]:.4f}")
        for subj, r in per_subj.items():
            print(f"  {subj}: r={r['r']:.4f} p_corr={r['p_corr']:.4f} segments={r['n_segments']}"
                  + (f" Cls@Top10={r['cls_top10']:.4f}" if "cls_top10" in r else ""))

    path = os.path.join(args.out_dir, "eval_summary.json")
    save_json(summary, path)
    print(f"\n[Saved] {path}")


def parse():
    p = argparse.ArgumentParser("BrainVista: rollout evaluation on held-out runs")
    add_data_args(p, inherit=True)
    add_stimulus_args(p, inherit=True)
    add_eval_data_args(p)
    add_runtime_args(p)
    p.set_defaults(seed=0)
    g = p.add_argument_group("evaluation")
    g.add_argument("--horizons", nargs="+", type=int, default=[1, 10, 20])
    g.add_argument("--anchor_stride", type=int, default=1, help="1 = every valid anchor (paper protocol).")
    g.add_argument("--batch_size", type=int, default=256, help="Anchors rolled out in parallel.")
    g.add_argument("--gt_space", type=str, default="fmri", choices=["fmri", "recon"],
                   help="fmri: normalized fMRI (paper); recon: detokenized ground-truth tokens.")
    g.add_argument("--no_identification", action="store_true", help="Skip Cls@TopK and 2v2.")
    g.add_argument("--cls_pool_size", type=int, default=200,
                   help="Candidate pool N (Algonauts 200, CineBrain 400, HAD 300).")
    g.add_argument("--cls_ks", nargs="+", type=int, default=[10])
    g.add_argument("--id_stride", type=int, default=1,
                   help="Keep every n-th segment for identification (memory).")
    g.add_argument("--two_v_two_pairs", type=int, default=5000)
    g.add_argument("--save_per_dim", action="store_true",
                   help="Save pooled per-parcel temporal correlation maps.")
    g.add_argument("--out_dir", type=str, default="./eval_output")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
