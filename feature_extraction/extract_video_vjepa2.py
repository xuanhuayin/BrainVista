#!/usr/bin/env python3
# feature_extraction/extract_video_vjepa2.py
"""Frozen V-JEPA 2 video features on the TR-locked end-aligned grid (paper §4.1, Table 4).

For every sample end time tau, `--num_frames` frames are taken uniformly from
the window [tau - window, tau) (only frames that have finished by tau), passed
through the V-JEPA 2 encoder (predictor skipped), and the token-averaged hidden
states of the selected layers are stored:

    OUT_ROOT/<rid>.npy   [n_bins * ratio, n_layers, D]   (float16)

Example (Algonauts 2025: TR 1.49 s, 3 samples per TR -> ~0.5 s stride, 2.0 s clips):
    python feature_extraction/extract_video_vjepa2.py --manifest video_manifest.csv \
        --tr 1.49 --ratio 3 --window 2.0 --out_root features_raw/video
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from common import (dump_extraction_cfg, n_bins_for, read_manifest, sample_end_times,
                    save_features, select_layers)


def clip_indices(end_time: float, fps: float, n_frames_total: int, window: float, num_frames: int):
    """Frame indices in [end_time - window, end_time) whose display interval has ended by end_time."""
    last = int(np.floor(end_time * fps)) - 1
    first = int(np.floor(max(0.0, end_time - window) * fps))
    last = min(max(last, 0), n_frames_total - 1)
    first = min(max(first, 0), last)
    return np.linspace(first, last, num_frames).round().astype(np.int64)


@torch.no_grad()
def encode_clips(model, processor, clips, layers, device, dtype):
    inputs = processor(clips, return_tensors="pt")
    pv = inputs["pixel_values_videos"].to(device, dtype)
    out = model(pixel_values_videos=pv, skip_predictor=True, output_hidden_states=True)
    hs = out.hidden_states if getattr(out, "hidden_states", None) else (out.last_hidden_state,)
    return torch.stack([h.mean(dim=1) for h in select_layers(hs, layers)], dim=1).float().cpu().numpy()


def extract_run(model, processor, video_path, end_times, args, device, dtype):
    import decord
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps, n_total = float(vr.get_avg_fps()), len(vr)
    h, w = vr[0].shape[:2]
    if args.short_side > 0 and min(h, w) > args.short_side:
        s = args.short_side / min(h, w)
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0),
                                width=int(round(w * s / 2) * 2), height=int(round(h * s / 2) * 2))
    feats = []
    for c in range(0, len(end_times), args.batch_size):
        clips = [vr.get_batch(clip_indices(tau, fps, n_total, args.window, args.num_frames)).asnumpy()
                 for tau in end_times[c:c + args.batch_size]]
        feats.append(encode_clips(model, processor, clips, args.layers, device, dtype))
    return np.concatenate(feats, axis=0)


def main(args):
    from transformers import AutoModel, AutoVideoProcessor
    import decord

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoVideoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()
    dump_extraction_cfg(args.out_root, vars(args))

    for row in tqdm(read_manifest(args.manifest), desc="[V-JEPA 2]", ncols=110, disable=None):
        vr = decord.VideoReader(row["path"], ctx=decord.cpu(0))
        duration = len(vr) / float(vr.get_avg_fps())
        del vr
        n_bins = n_bins_for(row, duration, args.tr)
        ends = sample_end_times(n_bins, args.tr, args.ratio, float(row.get("onset_offset") or 0.0))
        feats = extract_run(model, processor, row["path"], ends, args, device, dtype)
        save_features(args.out_root, row["rid"], feats)


def parse():
    p = argparse.ArgumentParser("Frozen V-JEPA 2 video features (end-aligned)")
    p.add_argument("--manifest", required=True, help="CSV: rid,path[,n_bins,onset_offset]")
    p.add_argument("--out_root", required=True)
    p.add_argument("--model_id", default="facebook/vjepa2-vitg-fpc64-256")
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--ratio", type=int, default=3, help="Samples per fMRI bin.")
    p.add_argument("--window", type=float, default=2.0, help="Clip length in seconds.")
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--layers", nargs="+", type=int, default=[-1])
    p.add_argument("--short_side", type=int, default=320, help="Decode-time resize (0 = off).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse())
