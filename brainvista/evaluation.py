# brainvista/evaluation.py
"""Rollout evaluation shared by validation (model selection) and the final test protocol."""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .checkpoint import TokenizerBundle, decode_tokens, load_all_tokenizers, load_predictor, torch_load
from .cli import resolve_inherited
from .data.controls import apply_stim_control
from .data.datasets import run_anchors
from .data.io import parse_run_list, runs_by_subject
from .data.normalization import compute_train_stats
from .data.runs import RunData, load_runs
from .metrics import ForecastMetrics
from .rollout import gather_targets, rollout, run_to_device
from .utils import get_device


@dataclass
class EvalContext:
    device: torch.device
    model: torch.nn.Module
    cfg: dict
    bundles: Dict[str, TokenizerBundle]
    z_splits: List[int]
    vox_splits: List[int]
    runs: List[RunData]


def add_missing_norm_stats(args, bundles: Dict[str, TokenizerBundle], subjects: List[str]):
    """Training-set statistics for subjects a tokenizer was not trained on (cross-subject transfer)."""
    path = getattr(args, "norm_train_list_file", None)
    missing = [(n, s) for n, b in bundles.items() for s in subjects
               if b.norm_mode == "train" and b.stats_for(s) is None]
    if not missing:
        return
    if not path or not getattr(args, "fmri_root", None):
        raise ValueError(f"No training-set normalization statistics for {sorted({s for _, s in missing})}; "
                         "pass --norm_train_list_file (their training runs) and --fmri_root.")
    train = runs_by_subject(parse_run_list(path, subjects))
    for net, subj in missing:
        bundles[net].norm_stats[subj] = compute_train_stats(args.fmri_root, net, subj, train.get(subj, []))
    print(f"[Norm] training-set statistics estimated for {sorted({s for _, s in missing})}")


def load_eval_context(args, need_fmri: bool) -> EvalContext:
    """Checkpoint-consistent setup: inherit data settings, load tokenizers, runs and predictor."""
    device = get_device(args.cpu)
    cfg = torch_load(args.predictor_ckpt).get("cfg", {})
    resolve_inherited(args, cfg)
    bundles, z_splits, vox_splits = load_all_tokenizers(
        args.parcels, args.tokenizer_ckpt_dir, args.tokenizer_name_template, device)
    run_list = parse_run_list(args.run_list_file, args.subjects)
    add_missing_norm_stats(args, bundles, sorted({s for s, _ in run_list}))
    runs = load_runs(args, run_list, bundles, need_fmri=need_fmri)
    if not runs:
        raise RuntimeError("No evaluation runs could be loaded.")
    control = getattr(args, "stim_control", "none")
    if control != "none":
        if not args.tr:
            raise ValueError("--stim_control needs --tr")
        runs = apply_stim_control(runs, control, float(args.tr), args.min_shift_sec, args.seed)
        print(f"[Control] stimulus stream replaced by the '{control}' negative control")
    Dv, Da, Dw, Dz = runs[0].dims
    if Dz != sum(z_splits):
        raise ValueError(f"Token dim mismatch: data d_brain={Dz}, tokenizers sum(d_k)={sum(z_splits)}")
    model, cfg = load_predictor(args.predictor_ckpt, Dv, Da, Dw, Dz, z_splits, device)
    return EvalContext(device, model, cfg, bundles, z_splits, vox_splits, runs)


@torch.no_grad()
def evaluate_runs(model, runs: List[RunData], bundles: Dict[str, TokenizerBundle],
                  networks: List[str], vox_splits: List[int], H: int,
                  stim_offset: int = 0, anchor_stride: int = 1, batch_size: int = 256,
                  use_amp: bool = True, gt_space: str = "fmri", keep_segments: bool = False,
                  id_stride: int = 1, verbose: bool = False) -> Dict[str, ForecastMetrics]:
    """H-step conditional rollout from every valid anchor of every run.

    Predictions are decoded by the frozen detokenizers into the normalized model
    space and compared with the normalized fMRI (gt_space="fmri") or, as a
    diagnostic, with the detokenized ground-truth tokens (gt_space="recon").
    Returns one ForecastMetrics accumulator per subject.
    """
    model.eval()
    device = next(model.parameters()).device
    out: Dict[str, ForecastMetrics] = {}
    for ri, run in enumerate(runs):
        anchors = run_anchors(run.T, model.L, H, stim_offset, anchor_stride)
        if not anchors:
            if verbose:
                print(f"[Eval] {run.subj}/{run.rid}: T={run.T} too short for L={model.L}, H={H}")
            continue
        acc = out.setdefault(run.subj, ForecastMetrics(networks, vox_splits, H, keep_segments))
        V, A, W, Z = run_to_device(run, device)
        X = None
        if gt_space == "fmri":
            if run.X is None:
                raise ValueError("gt_space='fmri' needs --fmri_root (normalized fMRI ground truth)")
            X = torch.from_numpy(run.X).float().to(device)
        for c in range(0, len(anchors), batch_size):
            a = anchors[c:c + batch_size]
            Z_pred = rollout(model, V, A, W, Z, a, H, stim_offset, batch_size, use_amp)
            X_pred = decode_tokens(Z_pred, bundles, networks)
            X_true = gather_targets(X, a, H) if X is not None else \
                decode_tokens(gather_targets(Z, a, H), bundles, networks)
            keep = (np.arange(c, c + len(a)) % max(1, id_stride)) == 0
            acc.update(X_pred, X_true, run_id=ri, anchors=a, keep_mask=keep)
    return out


def finalize_all(acc: Dict[str, ForecastMetrics], identification: bool = False,
                 **id_kwargs) -> Dict[str, dict]:
    return {subj: m.finalize(identification=identification, **id_kwargs) for subj, m in acc.items()}


def mean_over_subjects(per_subject: Dict[str, dict], key: str) -> float:
    vals = [r[key] for r in per_subject.values() if key in r]
    return float(np.mean(vals)) if vals else float("nan")
