#!/usr/bin/env python3
# tests/make_synthetic_data.py
"""Write a small synthetic dataset in the BrainVista on-disk layout (for smoke tests).

A latent brain state follows h_t = a h_{t-1} + B s_t + noise, driven by a smooth
stimulus process s_t that is observed through three modality-specific feature
maps sampled at `ratio` x the fMRI rate; parcel signals are noisy linear
read-outs of h_t split into K networks.
"""
import argparse
import os

import numpy as np


def main(a):
    rng = np.random.default_rng(a.seed)
    nets = {"NetA": 12, "NetB": 8, "NetC": 10}
    P = sum(nets.values())
    d_s, d_h = 6, 5
    feat_dims = {"video": 16, "audio": 8, "text": 12}
    maps = {m: rng.standard_normal((d_s, d)) for m, d in feat_dims.items()}
    B = rng.standard_normal((d_s, d_h)) * 0.6
    C = rng.standard_normal((d_h, P))

    splits = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    lists = {k: [] for k in splits}
    for subj in a.subjects:
        idx = 0
        for split, n in splits.items():
            for _ in range(n):
                rid = f"run-{idx:02d}"; idx += 1
                lists[split].append(f"{subj} {rid}")
                Ts = a.T * a.ratio
                s = np.zeros((Ts, d_s))
                for i in range(1, Ts):
                    s[i] = 0.9 * s[i - 1] + 0.3 * rng.standard_normal(d_s)
                s_bin = s.reshape(a.T, a.ratio, d_s).mean(1)
                h = np.zeros((a.T, d_h))
                for t in range(1, a.T):
                    h[t] = 0.85 * h[t - 1] + s_bin[t] @ B + 0.2 * rng.standard_normal(d_h)
                X = h @ C + 0.5 * rng.standard_normal((a.T, P)) + 3.0
                c = 0
                for net, Pk in nets.items():
                    d = os.path.join(a.out, "fmri", net, subj)
                    os.makedirs(d, exist_ok=True)
                    np.save(os.path.join(d, rid + ".npy"), X[:, c:c + Pk].astype(np.float32))
                    c += Pk
                for m, W in maps.items():
                    d = os.path.join(a.out, "features", m, subj)
                    os.makedirs(d, exist_ok=True)
                    F = s @ W + 0.1 * rng.standard_normal((Ts, W.shape[1]))
                    np.save(os.path.join(d, rid + ".npy"), F.astype(np.float32))
    for split, rows in lists.items():
        with open(os.path.join(a.out, f"{split}.txt"), "w") as f:
            f.write("\n".join(rows) + "\n")
    print(f"synthetic data -> {a.out} | networks={list(nets)} | subjects={a.subjects}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--subjects", nargs="+", default=["sub-01", "sub-02"])
    p.add_argument("--T", type=int, default=150)
    p.add_argument("--ratio", type=int, default=3)
    p.add_argument("--n_train", type=int, default=6)
    p.add_argument("--n_val", type=int, default=2)
    p.add_argument("--n_test", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
