# brainvista/cli.py
"""Command-line options shared by the training, evaluation and inference entry points."""
import argparse

from .data.runs import MODALITIES
from .models.predictor import INPUT_MODES, MASK_TYPES, MIXER_TYPES

YEO7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

# Settings that must match between training and evaluation; evaluation and
# inference inherit them from the predictor checkpoint unless given explicitly.
INHERITED_DEFAULTS = {
    "parcels": YEO7, "modalities": list(MODALITIES),
    "ratio_v": 3, "ratio_a": 3, "ratio_w": 3,
    "layer_pool": "mean", "layer_axis": 1, "select_layer_idx": -1, "collapse": "flatten",
    "stim_offset": 0, "tokenizer_name_template": "ckpt_tokenizer_{parcel}/best.pt",
    "feature_id_regex": None, "stim_transform": "none", "fir_lags": 4, "tr": None,
}


def _default(name, inherit):
    return None if inherit else INHERITED_DEFAULTS[name]


def resolve_inherited(args, cfg: dict):
    """Fill options left unset (None) from the checkpoint config, then from the defaults."""
    for k, v in INHERITED_DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, cfg.get(k, v))
    return args


def add_data_args(p: argparse.ArgumentParser, need_z: bool = True, inherit: bool = False):
    g = p.add_argument_group("data")
    g.add_argument("--subjects", nargs="+", default=["sub-01"],
                   help="Subjects to use; bare run ids in run lists are expanded to these.")
    g.add_argument("--parcels", "--networks", dest="parcels", nargs="+", default=_default("parcels", inherit),
                   help="Functional networks / ROI groups k = 1..K, in token concatenation order.")
    g.add_argument("--fmri_root", type=str, default=None,
                   help="Model-space fMRI: FMRI_ROOT/<network>/<subj>/<rid>.npy, shape (T, P_k).")
    if need_z:
        g.add_argument("--z_root", type=str, required=True,
                       help="Brain region tokens: Z_ROOT/z_<network>/<subj>/<rid>_z.npy.")
    g.add_argument("--tokenizer_ckpt_dir", type=str, required=True)
    g.add_argument("--tokenizer_name_template", type=str,
                   default=_default("tokenizer_name_template", inherit))
    return g


def add_stimulus_args(p: argparse.ArgumentParser, inherit: bool = False):
    d = lambda k: _default(k, inherit)
    g = p.add_argument_group("stimulus features")
    g.add_argument("--video_root", type=str, default=None, help="V-JEPA 2 features.")
    g.add_argument("--audio_root", type=str, default=None, help="Wav2Vec2-BERT features.")
    g.add_argument("--text_root", type=str, default=None, help="Llama 3.2 3B features.")
    g.add_argument("--modalities", nargs="+", default=d("modalities"), choices=MODALITIES,
                   help="Active modality set M_D (e.g. `video text` for HAD).")
    g.add_argument("--ratio_v", type=int, default=d("ratio_v"), help="Video samples per fMRI bin (0 disables).")
    g.add_argument("--ratio_a", type=int, default=d("ratio_a"), help="Audio samples per fMRI bin (0 disables).")
    g.add_argument("--ratio_w", type=int, default=d("ratio_w"), help="Text samples per fMRI bin (0 disables).")
    g.add_argument("--layer_pool", type=str, default=d("layer_pool"),
                   choices=["mean", "last", "select", "concat"])
    g.add_argument("--layer_axis", type=int, default=d("layer_axis"))
    g.add_argument("--select_layer_idx", type=int, default=d("select_layer_idx"))
    g.add_argument("--collapse", type=str, default=d("collapse"))
    g.add_argument("--feature_id_regex", type=str, default=None,
                   help="Map run ids to shared stimulus-feature ids, e.g. 'task-[A-Za-z0-9]+'.")
    g.add_argument("--tr", type=float, default=None, help="fMRI repetition time in seconds.")
    g.add_argument("--stim_transform", type=str, default=d("stim_transform"),
                   choices=["none", "hrf", "fir"],
                   help="Stimulus construction: bin-wise pooling (paper default), canonical-HRF "
                        "convolution, or a causal FIR lag bank.")
    g.add_argument("--fir_lags", type=int, default=d("fir_lags"))
    g.add_argument("--stim_offset", type=int, default=d("stim_offset"),
                   help="Shift the stimulus window back by this many bins (1 = s^{<u}; <0 = look-ahead).")
    return g


def add_model_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("BrainVista predictor")
    g.add_argument("--d_model", type=int, default=256)
    g.add_argument("--nhead", type=int, default=8)
    g.add_argument("--nlayers", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--L", type=int, default=40, help="Context length L (sequence length 2L).")
    g.add_argument("--mask", type=str, default="s2b", choices=MASK_TYPES)
    g.add_argument("--input_mode", type=str, default="both", choices=INPUT_MODES)
    g.add_argument("--brain_self_attn", type=int, default=1,
                   help="S2B brain->brain rule t_j <= t_i (1, paper) or t_j < t_i (0).")
    g.add_argument("--mixer", type=str, default="attn", choices=MIXER_TYPES)
    g.add_argument("--spatial_dz", type=int, default=128, help="Shared mixer dimension d_z.")
    g.add_argument("--spatial_nhead", type=int, default=4)
    g.add_argument("--spatial_nlayers", type=int, default=1)
    g.add_argument("--spatial_dropout", type=float, default=0.1)
    return g


def add_eval_data_args(p: argparse.ArgumentParser):
    """Options of the held-out evaluation entry points (evaluate / inference / analysis)."""
    g = p.add_argument_group("held-out data")
    g.add_argument("--predictor_ckpt", type=str, required=True)
    g.add_argument("--run_list_file", "--test_list_file", dest="run_list_file", type=str, required=True)
    g.add_argument("--norm_train_list_file", type=str, default=None,
                   help="Training runs of subjects absent from the tokenizer statistics "
                        "(cross-subject / leave-one-subject-out evaluation).")
    g.add_argument("--stim_control", type=str, default="none", choices=["none", "circular", "cross_run"],
                   help="Negative control: circularly shifted or cross-run replaced stimulus stream.")
    g.add_argument("--min_shift_sec", type=float, default=30.0)
    return g


def add_runtime_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("runtime")
    g.add_argument("--no_amp", action="store_true", help="Disable mixed precision.")
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--seed", type=int, default=2025)
    return g
