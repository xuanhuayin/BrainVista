#!/usr/bin/env python3
# preprocessing/algonauts_transcripts_to_events.py
"""Convert Algonauts 2025 transcripts (one row per TR) into word-event tables.

The release stores, per TR row, `words_per_tr`, `onsets_per_tr` and
`durations_per_tr` as Python list literals. Each word becomes one event with
offset = onset + duration; extract_text_llama.py makes a word visible only once
it has ended. Onsets are taken as seconds from run start; use
--onsets_relative_to_tr if your copy stores them relative to the TR row.

    python preprocessing/algonauts_transcripts_to_events.py \
        --transcript_dir stimuli/transcripts --out_dir text_events --tr 1.49
writes text_events/<stem>.tsv for every *.tsv found (e.g. friends_s01e02a.tsv).
"""
import argparse
import ast
import csv
import glob
import os


def _list(v):
    if v is None or str(v).strip() in ("", "[]", "nan"):
        return []
    return list(ast.literal_eval(str(v)))


def convert(path, out_path, tr, relative):
    events = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            words = _list(row.get("words_per_tr"))
            onsets = _list(row.get("onsets_per_tr"))
            durs = _list(row.get("durations_per_tr"))
            for w, on, du in zip(words, onsets, durs):
                on = float(on) + (i * tr if relative else 0.0)
                events.append((on, on + float(du), str(w)))
    events.sort()
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["onset", "offset", "text"])
        for on, off, w in events:
            wr.writerow([f"{on:.3f}", f"{off:.3f}", w])
    return len(events)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Algonauts per-TR transcripts -> word events")
    p.add_argument("--transcript_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tr", type=float, default=1.49)
    p.add_argument("--onsets_relative_to_tr", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for fp in sorted(glob.glob(os.path.join(a.transcript_dir, "**", "*.tsv"), recursive=True)):
        out = os.path.join(a.out_dir, os.path.basename(fp))
        print(f"{os.path.basename(fp)}: {convert(fp, out, a.tr, a.onsets_relative_to_tr)} words")
