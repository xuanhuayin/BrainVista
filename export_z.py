#!/usr/bin/env python3
# export_z.py
"""Export brain region tokens tau_f^{t,k} = h_phi^k(x_bar^{k,t}) with a frozen tokenizer.

Each run is normalized with the subject's training-set statistics stored in
the tokenizer checkpoint and written to Z_ROOT/z_<network>/<subj>/<rid>_z.npy.
For a subject the tokenizer was not trained on (cross-subject transfer), the
statistics are estimated from that subject's own training runs
(--train_list_file); validation/test fMRI is never used for normalization.
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from brainvista.checkpoint import load_tokenizer
from brainvista.data.io import find_fmri_file, load_npy, parse_run_list, runs_by_subject
from brainvista.data.normalization import NORM_MODES, compute_train_stats, normalize
from brainvista.utils import get_device


@torch.no_grad()
def export(args):
    device = get_device(args.cpu)
    tok = load_tokenizer(args.ckpt, args.network, device)
    mode = args.norm_mode or tok.norm_mode
    print(f"[Export:{args.network}] P_k={tok.Db} d_k={tok.z_dim} norm={mode}")

    run_list = []
    for path in args.run_list_file:
        run_list += parse_run_list(path, args.subjects)
    train_by_subj = (runs_by_subject(parse_run_list(args.train_list_file, args.subjects))
                     if args.train_list_file else {})

    stats_cache = {}

    def stats_for(subj):
        if subj not in stats_cache:
            st = tok.stats_for(subj)
            if st is None and mode == "train":
                if subj not in train_by_subj:
                    raise KeyError(f"No training-set statistics for {subj} in {args.ckpt}; "
                                   f"pass --train_list_file listing {subj}'s training runs.")
                st = compute_train_stats(args.fmri_root, args.network, subj, train_by_subj[subj])
                print(f"  [norm] {subj}: statistics from {st['n_frames']} of its training frames")
            stats_cache[subj] = st
        return stats_cache[subj]

    out_root = os.path.join(args.out_root, f"z_{args.network}")
    n_done = 0
    for subj, rid in tqdm(sorted(set(run_list)), desc=f"[Export:{args.network}]", ncols=110, disable=None):
        fp = find_fmri_file(args.fmri_root, args.network, subj, rid)
        if fp is None:
            print(f"[WARN] missing fMRI for {subj}/{rid}")
            continue
        X = load_npy(fp)
        if X.ndim != 2 or X.shape[1] != tok.Db:
            raise RuntimeError(f"{fp}: shape {X.shape}, expected (T, {tok.Db})")
        Xn = normalize(X, stats_for(subj), mode)
        Z = tok.enc(torch.from_numpy(Xn).to(device)).cpu().numpy().astype(np.float32)
        os.makedirs(os.path.join(out_root, subj), exist_ok=True)
        np.save(os.path.join(out_root, subj, f"{rid}_z.npy"), Z)
        n_done += 1
    print(f"[Export:{args.network}] {n_done} runs -> {out_root}")


def parse():
    ap = argparse.ArgumentParser("BrainVista: export brain region tokens from a trained tokenizer")
    ap.add_argument("--fmri_root", type=str, required=True)
    ap.add_argument("--network", "--parcel_name", dest="network", type=str, required=True)
    ap.add_argument("--subjects", nargs="+", default=["sub-01"])
    ap.add_argument("--run_list_file", nargs="+", required=True,
                    help="One or more run lists (train / val / test) to export.")
    ap.add_argument("--train_list_file", type=str, default=None,
                    help="Training runs, used only for subjects missing from the checkpoint statistics.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--norm_mode", type=str, default=None, choices=NORM_MODES,
                    help="Override the checkpoint's normalization mode.")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    export(parse())
