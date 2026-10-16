# brainvista/checkpoint.py
"""Loading of tokenizer and predictor checkpoints.

Tokenizer checkpoint (`ckpt_tokenizer_<network>/best.pt`):
    enc, dec      state dicts of EncAE / Dec
    Db, z_dim     P_k and d_k
    args          resolved training configuration
    norm          {"mode": "train", "stats": {subj: {"mu": Tensor[P_k], "std": Tensor[P_k]}}}

Predictor checkpoint (`predictor_best.pt`):
    state_dict, cfg, z_splits, parcels, epoch

Checkpoints written by the original supplementary code (no `norm` entry, no
`mask` key in cfg) are still accepted: they fall back to run-wise
normalization and to the strictly-past brain -> brain S2B rule they were
trained with.
"""
import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .data.io import subject_variants
from .models.predictor import BrainVista
from .models.tokenizer import Dec, EncAE


def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def torch_load(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class TokenizerBundle:
    network: str
    enc: EncAE
    dec: Dec
    Db: int
    z_dim: int
    cfg: dict
    norm_mode: str = "run"
    norm_stats: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    def stats_for(self, subj: str) -> Optional[Dict[str, np.ndarray]]:
        for v in subject_variants(subj):
            if v in self.norm_stats:
                return self.norm_stats[v]
        return None


def stats_to_numpy(stats: dict) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for subj, st in (stats or {}).items():
        out[subj] = {k: (v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                     for k, v in st.items() if k in ("mu", "std")}
    return out


def stats_to_torch(stats: Dict[str, Dict[str, np.ndarray]]) -> dict:
    return {subj: {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in st.items()
                   if k in ("mu", "std")}
            for subj, st in stats.items()}


def load_tokenizer(ckpt_path: str, network: str, device) -> TokenizerBundle:
    ck = torch_load(ckpt_path)
    cfg = ck.get("args", ck.get("cfg", {}))
    if isinstance(cfg, argparse.Namespace):
        cfg = vars(cfg)
    Db = int(ck["Db"])
    z_dim = int(ck.get("z_dim", cfg.get("z_dim", 128)))
    hidden = int(cfg.get("hidden", 768))
    dropout = float(cfg.get("dropout", 0.1))

    enc = EncAE(Db, z_dim, hidden, dropout)
    dec = Dec(z_dim, Db, hidden, dropout)
    enc.load_state_dict(_strip_prefixes(ck["enc"]), strict=True)
    dec.load_state_dict(_strip_prefixes(ck["dec"]), strict=True)
    enc.to(device).eval()
    dec.to(device).eval()
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)

    norm = ck.get("norm", {})
    return TokenizerBundle(network=network, enc=enc, dec=dec, Db=Db, z_dim=z_dim, cfg=cfg,
                           norm_mode=norm.get("mode", "run"),
                           norm_stats=stats_to_numpy(norm.get("stats", {})))


def tokenizer_path(ckpt_dir: str, template: str, network: str) -> str:
    try:
        name = template.format(parcel=network, network=network)
    except (IndexError, KeyError):
        name = template.format(network)
    return os.path.join(ckpt_dir, name)


def load_all_tokenizers(networks: List[str], ckpt_dir: str, template: str,
                        device) -> Tuple[Dict[str, TokenizerBundle], List[int], List[int]]:
    """Load the K frozen tokenizer--detokenizer pairs; returns (bundles, z_splits, vox_splits)."""
    bundles: Dict[str, TokenizerBundle] = {}
    print(f"--- Loading {len(networks)} network-wise tokenizers ---")
    for net in networks:
        path = tokenizer_path(ckpt_dir, template, net)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer checkpoint for network {net!r} not found: {path}")
        b = load_tokenizer(path, net, device)
        bundles[net] = b
        print(f"  > {net:<12s} P_k={b.Db:<5d} d_k={b.z_dim:<5d} norm={b.norm_mode}")
    z_splits = [bundles[n].z_dim for n in networks]
    vox_splits = [bundles[n].Db for n in networks]
    print(f"--- d_brain = {sum(z_splits)}, P = {sum(vox_splits)} ---")
    return bundles, z_splits, vox_splits


@torch.no_grad()
def decode_tokens(Z: torch.Tensor, bundles: Dict[str, TokenizerBundle],
                  networks: List[str]) -> torch.Tensor:
    """Frozen detokenization (Eq. 13): [..., d_brain] -> [..., P] in normalized model space."""
    parts, c = [], 0
    for net in networks:
        b = bundles[net]
        parts.append(b.dec(Z[..., c:c + b.z_dim].float()))
        c += b.z_dim
    return torch.cat(parts, dim=-1)


# ------------------------------------------------------------------ #
#  Predictor                                                         #
# ------------------------------------------------------------------ #

PREDICTOR_ARCH_KEYS = ("d_model", "nhead", "nlayers", "dropout", "L",
                       "spatial_nhead", "spatial_nlayers", "spatial_dropout", "spatial_dz",
                       "mixer", "mask", "input_mode", "brain_self_attn")


def predictor_kwargs(cfg: dict) -> dict:
    return dict(
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        nlayers=int(cfg.get("nlayers", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        L=int(cfg.get("L", 40)),
        spatial_nhead=int(cfg.get("spatial_nhead", 4)),
        spatial_nlayers=int(cfg.get("spatial_nlayers", 1)),
        spatial_dropout=float(cfg.get("spatial_dropout", 0.1)),
        spatial_dz=int(cfg.get("spatial_dz", 128)),
        mixer=str(cfg.get("mixer", "attn")),
        mask=str(cfg.get("mask", "s2b")),
        input_mode=str(cfg.get("input_mode", "both")),
        brain_self_attn=bool(cfg.get("brain_self_attn", True)),
    )


def _infer_from_state_dict(sd: Dict[str, torch.Tensor], cfg: dict) -> dict:
    """Recover architecture details that legacy checkpoints did not store in cfg."""
    cfg = dict(cfg)
    legacy = "mask" not in cfg
    if legacy:
        cfg["brain_self_attn"] = False
    layer_ids = [int(m.group(1)) for k in sd
                 for m in [re.match(r"^spatial_mixer\.layers\.(\d+)\.", k)] if m]
    if layer_ids:
        cfg["spatial_nlayers"] = max(layer_ids) + 1
        cfg.setdefault("mixer", "linear" if any(k.endswith(".mix") for k in sd) else "attn")
    elif not any(k.startswith("spatial_mixer.") for k in sd):
        cfg["mixer"] = "none"
    if "spatial_mixer.in_projs.0.weight" in sd:
        cfg["spatial_dz"] = int(sd["spatial_mixer.in_projs.0.weight"].shape[0])
    if "pos_embed.weight" in sd:
        cfg["L"] = int(sd["pos_embed.weight"].shape[0]) // 2
    return cfg


def load_predictor(ckpt_path: str, Dv: int, Da: int, Dw: int, Dz: int,
                   z_splits: List[int], device) -> Tuple[BrainVista, dict]:
    ckpt = torch_load(ckpt_path)
    sd = _strip_prefixes(ckpt.get("state_dict", ckpt))
    cfg = _infer_from_state_dict(sd, ckpt.get("cfg", {}))
    kw = predictor_kwargs(cfg)
    stim_in = int(sd["stim_proj.weight"].shape[1])
    if stim_in != Dv + Da + Dw:
        raise ValueError(f"Stimulus dim mismatch: checkpoint expects {stim_in}, "
                         f"data provides Dv+Da+Dw={Dv}+{Da}+{Dw}")
    model = BrainVista(Dv=Dv, Da=Da, Dw=Dw, Dz_in=Dz, Dz_out=Dz, z_splits=z_splits, **kw)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print("[Model] " + ", ".join(f"{k}={kw[k]}" for k in PREDICTOR_ARCH_KEYS if k in kw))
    return model, cfg
