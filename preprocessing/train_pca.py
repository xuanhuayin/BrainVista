#!/usr/bin/env python3
# preprocessing/train_pca.py
"""Training-split PCA, fitted per group and subject and frozen for validation/test.

Used for (a) CineBrain visual / auditory ROI voxels (Q_k = 8,405 / 10,541 ->
P_k = 512, Appendix A.7/A.9) and (b) modality-specific compression of stimulus
features (e.g. V-JEPA 2 1408 -> 512, w2v-BERT 1024 -> 256, Llama 3072 -> 512).

    IN_ROOT/<group>/<subj>/<rid>.npy  (T, Q)   ->   OUT_ROOT/<group>/<subj>/<rid>.npy  (T, n_components)

Signals are standardized with training-run statistics before PCA (disable with
--no_standardize). The basis, statistics and retained-variance ratio are saved
to OUT_ROOT/pca/<group>_<subj>.npz and summarized in OUT_ROOT/pca/summary.json.
Use --shared for stimulus features that are stored once for all subjects
(IN_ROOT/<group>/<rid>.npy); multi-layer features (T, n_layers, D) are reduced
with --layer_pool first.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brainvista.data.alignment import reduce_layers  # noqa: E402
from brainvista.data.io import find_run_file, load_npy, parse_run_list, runs_by_subject  # noqa: E402


def _load(args, group, subj, rid):
    root = os.path.join(args.in_root, group)
    fp = find_run_file(root, subj, rid, allow_shared=True)
    if fp is None:
        return None
    return reduce_layers(load_npy(fp), args.layer_pool, 1, -1)


def fit(args, group, subj, rids):
    n, s1, s2 = 0, None, None
    for rid in rids:
        X = _load(args, group, subj, rid)
        if X is None:
            continue
        X = X.astype(np.float64)
        s1 = X.sum(0) if s1 is None else s1 + X.sum(0)
        s2 = (X ** 2).sum(0) if s2 is None else s2 + (X ** 2).sum(0)
        n += X.shape[0]
    if n == 0:
        raise FileNotFoundError(f"No training data for group={group}, subj={subj}")
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6 if args.standardize else np.ones_like(mu)
    C = np.zeros((mu.size, mu.size))
    for rid in rids:
        X = _load(args, group, subj, rid)
        if X is not None:
            Xs = (X.astype(np.float64) - mu) / sd
            C += Xs.T @ Xs
    C /= n
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if dev == "cuda" else torch.float64
    evals, evecs = torch.linalg.eigh(torch.as_tensor(C, device=dev, dtype=dt))
    evals, evecs = evals.flip(0).double().cpu().numpy(), evecs.flip(1).double().cpu().numpy()
    k = min(args.n_components, mu.size)
    ratio = float(evals[:k].clip(min=0).sum() / max(evals.clip(min=0).sum(), 1e-12))
    return {"mean": mu.astype(np.float32), "std": sd.astype(np.float32),
            "components": evecs[:, :k].T.astype(np.float32),
            "explained_variance_ratio": ratio, "n_frames": n}


def main(args):
    train = runs_by_subject(parse_run_list(args.train_list_file, args.subjects))
    every = []
    for p in args.run_list_file:
        every += parse_run_list(p, args.subjects)
    every = runs_by_subject(sorted(set(every)))
    os.makedirs(os.path.join(args.out_root, "pca"), exist_ok=True)
    summary = {}
    for group in args.groups:
        for subj in (["shared"] if args.shared else sorted(train)):
            fit_rids = (sorted({r for rs in train.values() for r in rs}) if args.shared else train[subj])
            src_subj = sorted(train)[0] if args.shared else subj
            pca = fit(args, group, src_subj, fit_rids)
            tag = f"{group}_{subj}"
            np.savez(os.path.join(args.out_root, "pca", f"{tag}.npz"), **pca)
            summary[tag] = {"Q": int(pca["mean"].size), "P": int(pca["components"].shape[0]),
                            "explained_variance_ratio": pca["explained_variance_ratio"],
                            "n_train_frames": pca["n_frames"]}
            print(f"[PCA] {tag}: {summary[tag]}")
            targets = ({r for rs in every.values() for r in rs} if args.shared else every.get(subj, []))
            for rid in sorted(targets):
                X = _load(args, group, src_subj, rid)
                if X is None:
                    print(f"[WARN] missing {group}/{subj}/{rid}")
                    continue
                Y = ((X - pca["mean"]) / pca["std"]) @ pca["components"].T
                d = os.path.join(args.out_root, group) if args.shared else os.path.join(args.out_root, group, subj)
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, f"{rid}.npy"), Y.astype(np.float32))
    with open(os.path.join(args.out_root, "pca", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Training-split PCA (CineBrain voxels / stimulus features)")
    p.add_argument("--in_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--groups", nargs="+", required=True, help="e.g. Visual Auditory, or video audio text")
    p.add_argument("--subjects", nargs="+", default=["sub-01"])
    p.add_argument("--train_list_file", required=True, help="PCA is fitted on these runs only.")
    p.add_argument("--run_list_file", nargs="+", required=True, help="Runs to project (train/val/test).")
    p.add_argument("--n_components", type=int, default=512)
    p.add_argument("--no_standardize", dest="standardize", action="store_false")
    p.add_argument("--shared", action="store_true", help="Subject-independent stimulus features.")
    p.add_argument("--layer_pool", default="mean", choices=["mean", "last", "concat"])
    main(p.parse_args())
