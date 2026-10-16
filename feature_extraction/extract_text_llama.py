#!/usr/bin/env python3
# feature_extraction/extract_text_llama.py
"""Frozen Llama 3.2 3B text features with causal prefixes (paper §4.1, Appendix A.8).

Input per run is an event table (TSV/CSV) with columns `onset`, `offset` (or
`duration`) and `text`, in stimulus seconds. Two visibility rules:

  words        (subtitles / transcripts) a word becomes visible when it ends.
               Every word is embedded by the causal LM conditioned only on the
               preceding words (at most --context_tokens tokens); the feature at
               sample time tau is the hidden state of the last word that ended
               by tau (`--pool carry`, the causal-prefix state) or the mean of
               the words ending in (tau - stride, tau] (`--pool window`).
               No token after tau is ever used.
  annotations  (HAD clip descriptions) an annotation is visible during
               [onset, offset), i.e. only after its clip started; each is
               embedded on its own (mean over its tokens).

Samples before any visible text are zeros. Output:

    OUT_ROOT/<rid>.npy   [n_bins * ratio, n_layers, 3072]   (float16)

The manifest has columns rid,path,n_bins[,onset_offset] where `path` is the
event table of the run (n_bins is required since text has no duration).
"""
import argparse
import csv

import numpy as np
import torch
from tqdm import tqdm

from common import dump_extraction_cfg, read_manifest, sample_end_times, save_features, select_layers


def read_events(path: str):
    with open(path, newline="") as f:
        dialect = csv.Sniffer().sniff(f.read(2048), delimiters=",\t")
        f.seek(0)
        rows = list(csv.DictReader(f, dialect=dialect))
    ev = []
    for r in rows:
        text = (r.get("text") or r.get("word") or "").strip()
        if not text:
            continue
        on = float(r["onset"])
        off = float(r["offset"]) if r.get("offset") not in (None, "") else on + float(r.get("duration") or 0.0)
        ev.append((on, off, text))
    ev.sort(key=lambda e: (e[1], e[0]))
    return ev


@torch.no_grad()
def embed_words_causal(model, tok, words, layers, context_tokens, device):
    """Contextual embedding of every word given only the preceding words (sliding windows)."""
    ids, owner = [], []
    for wi, w in enumerate(words):
        piece = tok(w if wi == 0 else " " + w, add_special_tokens=False)["input_ids"]
        ids += piece
        owner += [wi] * len(piece)
    ids, owner = np.asarray(ids), np.asarray(owner)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    n_layers = len(layers)
    out = np.zeros((len(words), n_layers, model.config.hidden_size), dtype=np.float32)
    last_tok = np.zeros(len(words), dtype=np.int64)
    for wi in range(len(words)):
        last_tok[wi] = np.nonzero(owner == wi)[0].max() if (owner == wi).any() else -1

    # Each window predicts states for tokens in its second half; the first half is context.
    ctx = max(16, context_tokens - len(bos))
    step = ctx // 2
    start = 0
    while start < len(ids):
        lo = max(0, start - (ctx - step))
        hi = min(len(ids), start + step)
        x = torch.as_tensor(bos + ids[lo:hi].tolist(), device=device)[None]
        hs = model(input_ids=x, output_hidden_states=True).hidden_states
        sel = torch.stack([h[0, len(bos):] for h in select_layers(hs, layers)], dim=1)   # [n, n_layers, D]
        for wi in np.unique(owner[start:hi]):
            t = last_tok[wi]
            if start <= t < hi:
                out[wi] = sel[t - lo].float().cpu().numpy()
        start = hi
    return out


@torch.no_grad()
def embed_sentences(model, tok, texts, layers, device):
    out = []
    for t in texts:
        x = tok(t, return_tensors="pt").to(device)
        hs = model(**x, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0].mean(0) for h in select_layers(hs, layers)], 0).float().cpu().numpy())
    return np.stack(out) if out else np.zeros((0, len(layers), model.config.hidden_size), np.float32)


def features_for_run(model, tok, events, ends, stride, args, device):
    D = model.config.hidden_size
    feats = np.zeros((len(ends), len(args.layers), D), dtype=np.float32)
    if not events:
        return feats
    if args.visibility == "annotations":
        E = embed_sentences(model, tok, [e[2] for e in events], args.layers, device)
        for i, tau in enumerate(ends):
            act = [k for k, (on, off, _) in enumerate(events) if on <= tau < off]
            if act:
                feats[i] = E[act].mean(0)
        return feats
    E = embed_words_causal(model, tok, [e[2] for e in events], args.layers, args.context_tokens, device)
    word_end = np.array([e[1] for e in events])
    for i, tau in enumerate(ends):
        n_vis = int(np.searchsorted(word_end, tau, side="right"))     # words with end <= tau
        if n_vis == 0:
            continue
        if args.pool == "carry":
            feats[i] = E[n_vis - 1]
        else:
            k = (word_end > tau - stride) & (word_end <= tau)
            if k.any():
                feats[i] = E[k].mean(0)
    return feats


def main(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()
    dump_extraction_cfg(args.out_root, vars(args))

    for row in tqdm(read_manifest(args.manifest), desc="[Llama 3.2]", ncols=110, disable=None):
        if not row.get("n_bins"):
            raise ValueError(f"{row['rid']}: text manifest needs n_bins")
        ends = sample_end_times(int(row["n_bins"]), args.tr, args.ratio, float(row.get("onset_offset") or 0.0))
        feats = features_for_run(model, tok, read_events(row["path"]), ends, args.tr / args.ratio, args, device)
        save_features(args.out_root, row["rid"], feats)


def parse():
    p = argparse.ArgumentParser("Frozen Llama 3.2 text features (causal prefixes)")
    p.add_argument("--manifest", required=True, help="CSV: rid,path,n_bins[,onset_offset]")
    p.add_argument("--out_root", required=True)
    p.add_argument("--model_id", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--ratio", type=int, default=3)
    p.add_argument("--visibility", choices=["words", "annotations"], default="words")
    p.add_argument("--pool", choices=["carry", "window"], default="carry")
    p.add_argument("--context_tokens", type=int, default=1024)
    p.add_argument("--layers", nargs="+", type=int, default=[-1])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
