#!/usr/bin/env python3
# analysis/merge_eval_summaries.py
"""Merge per-subject eval_summary.json files (one model per subject) into one summary.

    python analysis/merge_eval_summaries.py --inputs outputs/algonauts/eval/*/eval_summary.json \
        --out outputs/algonauts/eval_summary_all.json

Every horizon gets the union of subjects and recomputed across-subject mean/std,
which is how the paper's tables average over subjects.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brainvista.metrics import summarize_subjects  # noqa: E402
from brainvista.utils import save_json  # noqa: E402


def main(a):
    merged = {"inputs": a.inputs, "horizons": {}}
    for path in a.inputs:
        with open(path) as f:
            s = json.load(f)
        for h, block in s["horizons"].items():
            subj = merged["horizons"].setdefault(h, {"subjects": {}})["subjects"]
            for name, res in block["subjects"].items():
                if name in subj:
                    raise ValueError(f"{name} appears twice at {h} ({path})")
                subj[name] = res
    for h, block in merged["horizons"].items():
        block.update(summarize_subjects(block["subjects"]))
        m = block["mean"]
        print(f"{h}: n={block['n_subjects']} r={m.get('r', float('nan')):.4f} "
              f"p_corr={m.get('p_corr', float('nan')):.4f} Cls@Top10={m.get('cls_top10', float('nan')):.4f}")
    save_json(merged, a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Merge per-subject BrainVista evaluation summaries")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
