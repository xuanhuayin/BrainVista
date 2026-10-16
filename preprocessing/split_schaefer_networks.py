#!/usr/bin/env python3
# preprocessing/split_schaefer_networks.py
"""Split Schaefer-1000 parcel time series into Yeo-7 networks (Algonauts 2025, HAD).

Network membership is a fixed, non-learned index set taken from the Schaefer
2018 label names ("7Networks_LH_Vis_1", ...). Parcels keep their atlas order
within each network and are not averaged. Inputs:

  --h5_dir   Algonauts 2025 release: *.h5 files holding (T, 1000) datasets per run
             (run id = the `ses-..._task-...` part of the dataset key)
  --npy_dir  generic: NPY_DIR/<subj>/<rid>.npy of shape (T, 1000)

Output: FMRI_ROOT/<network>/<subj>/<rid>.npy of shape (T, P_k), plus
FMRI_ROOT/networks.json with the parcel indices and P_k of every network.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def network_groups(labels, n_networks: int = 7):
    groups = {}
    for col, name in enumerate(labels):
        name = name.decode() if isinstance(name, bytes) else str(name)
        m = re.search(rf"{n_networks}Networks_[LR]H_([A-Za-z]+)", name)
        if m:
            groups.setdefault(m.group(1), []).append(col)
    return groups


def load_labels(args):
    if args.labels_file:
        with open(args.labels_file) as f:
            rows = [ln.strip() for ln in f if ln.strip()]
        # Schaefer *_order.txt / LUT lines: "<idx> <name> r g b a" or bare names.
        names = [next((t for t in ln.split() if "Networks_" in t), ln) for ln in rows]
        return [n for n in names if "Networks_" in n]
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=args.n_rois, yeo_networks=args.n_networks)
    labels = list(atlas["labels"])
    if labels and str(labels[0]).lower() in ("background", "b'background'"):
        labels = labels[1:]
    return labels


def write_run(out_root, groups, subj, rid, X):
    for net, cols in groups.items():
        d = os.path.join(out_root, net, subj)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"{rid}.npy"), X[:, cols].astype(np.float32))


def run_id_from_key(key: str) -> str:
    m = re.search(r"(ses-[A-Za-z0-9_.-]+)", key)
    return m.group(1).split(".")[0] if m else re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip("/"))


def main(args):
    labels = load_labels(args)
    groups = network_groups(labels, args.n_networks)
    counts = {k: len(v) for k, v in groups.items()}
    print(f"[Schaefer-{args.n_rois} -> Yeo-{args.n_networks}] parcels per network: {counts} "
          f"(sum {sum(counts.values())})")
    if sum(counts.values()) != args.n_rois:
        raise ValueError("Label file does not cover every parcel")

    n = 0
    if args.h5_dir:
        import h5py
        for fp in sorted(glob.glob(os.path.join(args.h5_dir, "**", "*.h5"), recursive=True)):
            m = re.search(r"(sub-[A-Za-z0-9]+)", fp)
            subj = m.group(1) if m else "sub-unknown"
            with h5py.File(fp, "r") as f:
                keys = []
                f.visititems(lambda k, o: keys.append(k)
                             if isinstance(o, h5py.Dataset) and o.ndim == 2 and o.shape[1] == args.n_rois else None)
                for key in keys:
                    write_run(args.out_root, groups, subj, run_id_from_key(key), f[key][...])
                    n += 1
    else:
        for fp in sorted(glob.glob(os.path.join(args.npy_dir, "*", "*.npy"))):
            subj = os.path.basename(os.path.dirname(fp))
            X = np.load(fp)
            if X.shape[1] != args.n_rois:
                raise ValueError(f"{fp}: {X.shape[1]} columns, expected {args.n_rois}")
            write_run(args.out_root, groups, subj, os.path.basename(fp)[:-4], X)
            n += 1

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "networks.json"), "w") as f:
        json.dump({"P_k": counts, "parcel_indices": groups}, f, indent=2)
    print(f"[done] {n} runs -> {args.out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Schaefer parcels -> Yeo networks")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5_dir", type=str)
    src.add_argument("--npy_dir", type=str)
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--labels_file", type=str, default=None,
                   help="Schaefer label/LUT text file; default: nilearn fetch_atlas_schaefer_2018.")
    p.add_argument("--n_rois", type=int, default=1000)
    p.add_argument("--n_networks", type=int, default=7, choices=[7, 17])
    main(p.parse_args())
