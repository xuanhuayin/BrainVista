#!/usr/bin/env python3
# analysis/emotion_clustering.py
"""Emotion-category clustering of forecasted brain features (paper Appendix G).

Inputs are rollouts written by `inference.py --anchors tile` (non-overlapping
H-step forecasts covering each run) and a clip table with columns
`rid,start,end,label` (seconds; optional `subj`). For every clip, the frames
whose bin centre falls inside [start, end) are averaged, the clip vectors are
reduced with PCA (50 components), clustered with K-means (K = number of labels,
10 initializations, best inertia), and compared with the labels by ARI, NMI and
per-class purity (share of a class in its dominant cluster). `--feature true`
scores the ground-truth trajectories as the upper reference. The labels are
stimulus-derived, so this is an auxiliary representational diagnostic only.
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np


def frame_table(pred_dir, feature):
    """{(subj, rid): {bin_index: vector}} from tiled rollouts."""
    table = {}
    for fp in glob.glob(os.path.join(pred_dir, "*", f"*_{feature}.npy")):
        subj = os.path.basename(os.path.dirname(fp))
        rid = os.path.basename(fp)[: -len(f"_{feature}.npy")]
        Y = np.load(fp)                                            # [N, H, P]
        anchors = np.load(fp.replace(f"_{feature}.npy", "_anchors.npy"))
        frames = {}
        for n, u0 in enumerate(anchors):
            for h in range(Y.shape[1]):
                frames.setdefault(int(u0 + h), Y[n, h])
        table[(subj, rid)] = frames
    return table


def main(a):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    table = frame_table(a.pred_dir, a.feature)
    with open(a.labels_csv, newline="") as f:
        clips = list(csv.DictReader(f))
    X, y = [], []
    for c in clips:
        keys = [k for k in table if k[1] == c["rid"] and (not c.get("subj") or k[0] == c["subj"])]
        for key in keys:
            frames = table[key]
            idx = [t for t in frames if float(c["start"]) <= (t + 0.5) * a.tr < float(c["end"])]
            if idx:
                X.append(np.mean([frames[t] for t in idx], axis=0)); y.append(c["label"])
    if len(set(y)) < 2:
        raise RuntimeError("Fewer than two labelled clips with predicted frames.")
    X = np.stack(X)
    labels = sorted(set(y))
    k = a.k or len(labels)
    Xp = PCA(n_components=min(a.n_pca, X.shape[0], X.shape[1]), random_state=a.seed).fit_transform(X)
    cl = KMeans(n_clusters=k, n_init=a.n_init, random_state=a.seed).fit_predict(Xp)
    purity = {}
    by_label = defaultdict(list)
    for lab, c in zip(y, cl):
        by_label[lab].append(c)
    for lab in labels:
        purity[lab] = Counter(by_label[lab]).most_common(1)[0][1] / len(by_label[lab])
    res = {"feature": a.feature, "n_clips": len(y), "n_clusters": k,
           "ARI": float(adjusted_rand_score(y, cl)), "NMI": float(normalized_mutual_info_score(y, cl)),
           "purity": purity, "mean_purity": float(np.mean(list(purity.values()))),
           "clips_per_label": dict(Counter(y))}
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Emotion-category clustering of forecasted brain features")
    p.add_argument("--pred_dir", required=True, help="inference.py output (run with --anchors tile)")
    p.add_argument("--labels_csv", required=True, help="CSV: rid,start,end,label[,subj]")
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--feature", choices=["pred", "true"], default="pred")
    p.add_argument("--n_pca", type=int, default=50)
    p.add_argument("--k", type=int, default=0, help="Clusters; 0 = number of labels.")
    p.add_argument("--n_init", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    main(p.parse_args())
