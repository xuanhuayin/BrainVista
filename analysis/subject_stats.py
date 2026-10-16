#!/usr/bin/env python3
# analysis/subject_stats.py
"""Paired subject-level comparison of two evaluated models (paper Appendix C.3-C.5).

Reads two eval_summary.json files (model A, e.g. BrainVista, and reference
model B) and, for one horizon and metric, reports per-subject differences
d_i = A_i - B_i, their mean and SD, a percentile bootstrap 95% CI obtained by
resampling subjects with replacement, the win count, and planned one-sided
paired tests (exact sign test and Wilcoxon signed-rank, H1: A > B). With few
subjects these tests have coarse resolution (n = 4 gives p >= 0.0625).
"""
import argparse
import json

import numpy as np
from scipy import stats


def load(path, horizon, metric):
    with open(path) as f:
        subj = json.load(f)["horizons"][f"H{horizon}"]["subjects"]
    return {s: r[metric] for s, r in subj.items() if metric in r}


def main(a):
    A, B = load(a.a, a.horizon, a.metric), load(a.b, a.horizon, a.metric)
    subjects = sorted(set(A) & set(B))
    if not subjects:
        raise RuntimeError("No subjects in common.")
    d = np.array([A[s] - B[s] for s in subjects])
    rng = np.random.default_rng(a.seed)
    boot = d[rng.integers(0, len(d), size=(a.n_boot, len(d)))].mean(1)
    wins = int((d > 0).sum())
    nonzero = d[d != 0]
    sign_p = float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="greater").pvalue) \
        if len(nonzero) else float("nan")
    try:
        wil_p = float(stats.wilcoxon(d, alternative="greater").pvalue)
    except ValueError:
        wil_p = float("nan")
    res = {"metric": a.metric, "horizon": a.horizon, "n": len(d),
           "A_mean": float(np.mean([A[s] for s in subjects])), "A_sd": float(np.std([A[s] for s in subjects], ddof=1)) if len(d) > 1 else 0.0,
           "B_mean": float(np.mean([B[s] for s in subjects])), "B_sd": float(np.std([B[s] for s in subjects], ddof=1)) if len(d) > 1 else 0.0,
           "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
           "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "wins": f"{wins}/{len(d)}", "sign_test_p_one_sided": sign_p,
           "wilcoxon_p_one_sided": wil_p,
           "per_subject": {s: {"A": A[s], "B": B[s], "delta": A[s] - B[s]} for s in subjects}}
    print(json.dumps({k: v for k, v in res.items() if k != "per_subject"}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Paired subject-level statistics between two eval summaries")
    p.add_argument("--a", required=True, help="eval_summary.json of model A (e.g. BrainVista)")
    p.add_argument("--b", required=True, help="eval_summary.json of model B (reference)")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--metric", default="p_corr")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    main(p.parse_args())
