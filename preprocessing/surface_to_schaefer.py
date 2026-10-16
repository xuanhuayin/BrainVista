#!/usr/bin/env python3
# preprocessing/surface_to_schaefer.py
"""Aggregate fsLR surface fMRI (CIFTI dtseries) to Schaefer-1000 parcels (HAD, Appendix A.1).

Every parcel is the mean over its cortical vertices; vertices are matched
between the dtseries and the Schaefer fsLR-32k dlabel by (hemisphere, vertex).
Output: OUT_DIR/<subj>/<rid>.npy of shape (T, n_parcels), parcels ordered by
label key 1..N, plus OUT_DIR/parcel_labels.txt. Then run
split_schaefer_networks.py --npy_dir OUT_DIR --labels_file OUT_DIR/parcel_labels.txt.

The manifest is a CSV with columns subj,rid,path (path = *.dtseries.nii).
The dlabel is e.g. Schaefer2018_1000Parcels_7Networks_order.dlabel.nii from
the CBIG repository.
"""
import argparse
import csv
import os

import numpy as np


def surface_labels(dlabel_path):
    import nibabel as nib
    img = nib.load(dlabel_path)
    data = np.asarray(img.get_fdata()).reshape(-1).astype(np.int64)
    bm = img.header.get_axis(1)
    names = img.header.get_axis(0).label[0]
    per_struct = {}
    for name, slc, sub in bm.iter_structures():
        if sub.vertex is None or len(sub.vertex) == 0:
            continue
        lab = np.zeros(int(sub.nvertices[name]), dtype=np.int64)
        lab[sub.vertex] = data[slc]
        per_struct[name] = lab
    keys = sorted(k for k in names if k != 0)
    return per_struct, keys, [names[k][0] for k in keys]


def parcellate(dtseries_path, per_struct, keys):
    import nibabel as nib
    img = nib.load(dtseries_path)
    X = np.asarray(img.get_fdata(dtype=np.float32))                     # (T, grayordinates)
    bm = img.header.get_axis(1)
    T = X.shape[0]
    index = {k: i for i, k in enumerate(keys)}
    sums = np.zeros((T, len(keys)), dtype=np.float64)
    counts = np.zeros(len(keys), dtype=np.int64)
    for name, slc, sub in bm.iter_structures():
        if name not in per_struct or sub.vertex is None:
            continue
        lab = per_struct[name][sub.vertex]                               # label of each column
        cols = np.arange(slc.start, slc.stop if slc.stop is not None else X.shape[1])
        for k in np.unique(lab):
            if k == 0 or k not in index:
                continue
            m = lab == k
            sums[:, index[k]] += X[:, cols[m]].sum(1)
            counts[index[k]] += int(m.sum())
    if (counts == 0).any():
        missing = [keys[i] for i in np.nonzero(counts == 0)[0][:5]]
        raise ValueError(f"{dtseries_path}: {int((counts == 0).sum())} parcels without vertices, e.g. {missing}")
    return (sums / counts).astype(np.float32)


def main(args):
    per_struct, keys, names = surface_labels(args.dlabel)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "parcel_labels.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        Y = parcellate(r["path"], per_struct, keys)
        d = os.path.join(args.out_dir, r["subj"])
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"{r['rid']}.npy"), Y)
        print(f"[ok] {r['subj']}/{r['rid']}: {Y.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("fsLR CIFTI dtseries -> Schaefer parcels")
    p.add_argument("--manifest", required=True, help="CSV: subj,rid,path")
    p.add_argument("--dlabel", required=True)
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
