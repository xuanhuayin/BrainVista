#!/usr/bin/env python3
# feature_extraction/extract_audio_w2vbert.py
"""Frozen Wav2Vec2-BERT 2.0 audio features on the TR-locked end-aligned grid (paper §4.1, Table 4).

Audio is decoded to 16 kHz mono with ffmpeg. For every sample end time tau the
window [tau - window, tau) (zero-padded before stimulus onset) is encoded and
the time-averaged hidden states of the selected layers are stored:

    OUT_ROOT/<rid>.npy   [n_bins * ratio, n_layers, 1024]   (float16)

Example (Algonauts 2025): --tr 1.49 --ratio 3 --window 1.0
"""
import argparse
import subprocess

import numpy as np
import torch
from tqdm import tqdm

from common import (dump_extraction_cfg, n_bins_for, read_manifest, sample_end_times,
                    save_features, select_layers)


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def windows(wave: np.ndarray, end_times: np.ndarray, window: float, sr: int) -> np.ndarray:
    n = int(round(window * sr))
    out = np.zeros((len(end_times), n), dtype=np.float32)
    for i, tau in enumerate(end_times):
        e = min(int(np.floor(tau * sr)), len(wave))
        s = max(e - n, 0)
        if e > 0:
            seg = wave[s:e]
            out[i, n - len(seg):] = seg
    return out


@torch.no_grad()
def encode_windows(model, feature_extractor, segs, layers, sr, device, dtype):
    inputs = feature_extractor(list(segs), sampling_rate=sr, return_tensors="pt", padding=True)
    feats = inputs["input_features"].to(device, dtype)
    mask = inputs.get("attention_mask")
    mask = mask.to(device) if mask is not None else None
    out = model(input_features=feats, attention_mask=mask, output_hidden_states=True)
    pooled = []
    for h in select_layers(out.hidden_states, layers):
        if mask is not None and mask.shape[1] == h.shape[1]:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled.append((h * m).sum(1) / m.sum(1).clamp_min(1))
        else:
            pooled.append(h.mean(1))
    return torch.stack(pooled, dim=1).float().cpu().numpy()


def main(args):
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    fe = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = Wav2Vec2BertModel.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()
    dump_extraction_cfg(args.out_root, vars(args))

    for row in tqdm(read_manifest(args.manifest), desc="[w2v-BERT]", ncols=110, disable=None):
        wave = load_audio(row["path"], args.sr)
        n_bins = n_bins_for(row, len(wave) / args.sr, args.tr)
        ends = sample_end_times(n_bins, args.tr, args.ratio, float(row.get("onset_offset") or 0.0))
        segs = windows(wave, ends, args.window, args.sr)
        feats = np.concatenate([encode_windows(model, fe, segs[c:c + args.batch_size], args.layers,
                                               args.sr, device, dtype)
                                for c in range(0, len(segs), args.batch_size)], axis=0)
        save_features(args.out_root, row["rid"], feats)


def parse():
    p = argparse.ArgumentParser("Frozen Wav2Vec2-BERT audio features (end-aligned)")
    p.add_argument("--manifest", required=True, help="CSV: rid,path[,n_bins,onset_offset]")
    p.add_argument("--out_root", required=True)
    p.add_argument("--model_id", default="facebook/w2v-bert-2.0")
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--ratio", type=int, default=3)
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--layers", nargs="+", type=int, default=[-1])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
