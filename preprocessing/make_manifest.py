#!/usr/bin/env python3
# preprocessing/make_manifest.py
"""Build a feature-extraction manifest (rid,path[,n_bins]) from files on disk.

    python preprocessing/make_manifest.py --glob "movies/friends/**/*.mkv" \
        --rid_regex "s\\d{2}e\\d{2}[a-d]" --rid_format "task-{}" --out video_manifest.csv

`--rid_regex` picks the stimulus id out of the file name and `--rid_format`
turns it into the feature id used by the run lists (see --feature_id_regex in
the training scripts). For text manifests, `--n_bins_from FMRI_DIR` reads n_bins
from the fMRI file with the same id (first match below FMRI_DIR).
"""
import argparse
import csv
import glob
import os
import re

import numpy as np


def main(a):
    files = sorted(glob.glob(a.glob, recursive=True))
    fmri_index = {}
    if a.n_bins_from:
        for fp in glob.glob(os.path.join(a.n_bins_from, "**", "*.npy"), recursive=True):
            fmri_index.setdefault(os.path.basename(fp)[:-4], fp)
    rows = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        m = re.search(a.rid_regex, stem) if a.rid_regex else None
        rid = a.rid_format.format(m.group(0) if m else stem)
        row = {"rid": rid, "path": os.path.abspath(fp)}
        if fmri_index:
            hit = next((p for k, p in fmri_index.items() if rid in k), None)
            if hit is None:
                print(f"[WARN] no fMRI run for {rid}; skipped")
                continue
            row["n_bins"] = int(np.load(hit, mmap_mode="r").shape[0])
        rows.append(row)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["rid", "path"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} entries -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Feature-extraction manifest builder")
    p.add_argument("--glob", required=True)
    p.add_argument("--rid_regex", default=None)
    p.add_argument("--rid_format", default="{}")
    p.add_argument("--n_bins_from", default=None)
    p.add_argument("--out", required=True)
    main(p.parse_args())
