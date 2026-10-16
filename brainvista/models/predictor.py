# brainvista/models/predictor.py
"""BrainVista temporal predictor (paper §3.4, Fig. 3; Appendix A.9, D.2).

Input sequence at queried prediction time u (length S = 2L):

    S_u = (tau_s^{u-L+1}, tau_f^{u-L+1}, ..., tau_s^{u-1}, tau_f^{u-1}, tau_s^u, q_target)

The stimulus window ends at the target-aligned token tau_s^u. The final brain
slot is a learnable target query q_target, never an encoding of x^u, so the
target and future fMRI states are never model inputs.

Stimulus-to-Brain (S2B) mask (Appendix Eq. for M_u), with token time t and
type a in {s, b}:

    M_ij = 1  <=>  (a_i = s, a_j = s, t_j <= t_i)    stimulus-only causal stream
                   (a_i = b, a_j = s, t_j <= t_i)    brain reads causal stimulus
                   (a_i = b, a_j = b, t_j <= t_i)    brain reads past brain

Stimulus queries never read brain tokens, so (predicted) brain states cannot
leak into the stimulus representations during rollout.

The hidden state at q_target is projected by Proj_b^out to a preliminary
whole-brain latent, which the Spatial Mixer Head refines across the K
functional networks (Eqs. 11-13) before frozen detokenization.

Ablation variants used in the paper are exposed as constructor options:
    mask        in {"s2b", "causal", "bidirectional"}           (Table: mask ablation)
    mixer       in {"attn", "linear", "none"}                   (Table: Spatial Mixer ablation)
    input_mode  in {"both", "brain_only", "stim_only"}          (Table: input-source ablation)
"""
from typing import List, Optional

import torch
import torch.nn as nn

STIM, BRAIN = 0, 1

MASK_TYPES = ("s2b", "causal", "bidirectional")
MIXER_TYPES = ("attn", "linear", "none")
INPUT_MODES = ("both", "brain_only", "stim_only")


# ------------------------------------------------------------------ #
#  Attention masks                                                   #
# ------------------------------------------------------------------ #

def build_attention_mask(L: int, mask: str = "s2b", input_mode: str = "both",
                         brain_self_attn: bool = True,
                         key_block: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Additive attention mask (0 = allowed, -inf = blocked) of shape [2L, 2L].

    Position 2k holds the stimulus token at time k+1 and position 2k+1 the brain
    token at time k+1; position 2L-1 is the target query q_target (time u = L).

    Args:
        mask: "s2b" (paper), "causal" (vanilla lower-triangular over the
            interleaved order) or "bidirectional" (no temporal constraint).
        input_mode: "brain_only" blocks brain -> stimulus attention (past fMRI
            only); "stim_only" blocks attention to observed brain tokens
            (stimulus-to-brain prediction).
        brain_self_attn: S2B brain -> brain rule uses t_j <= t_i (paper) when
            True, or the strictly-past t_j < t_i rule of earlier checkpoints.
        key_block: optional bool [2L] marking key positions that no other
            position may attend to (used for history-token occlusion).
    """
    if mask not in MASK_TYPES:
        raise ValueError(f"mask must be one of {MASK_TYPES}, got {mask!r}")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")

    S = 2 * L
    pos = torch.arange(S)
    t = pos // 2 + 1                                   # token time 1..L
    a = pos % 2                                        # 0 = stim, 1 = brain
    ti, tj = t[:, None], t[None, :]
    ai, aj = a[:, None], a[None, :]

    if mask == "bidirectional":
        allowed = torch.ones(S, S, dtype=torch.bool)
    elif mask == "causal":
        allowed = pos[None, :] <= pos[:, None]
    else:
        brain_past = (tj <= ti) if brain_self_attn else (tj < ti)
        allowed = (((ai == STIM) & (aj == STIM) & (tj <= ti))
                   | ((ai == BRAIN) & (aj == STIM) & (tj <= ti))
                   | ((ai == BRAIN) & (aj == BRAIN) & brain_past))

    if input_mode == "brain_only":
        allowed &= ~((ai == BRAIN) & (aj == STIM))
    elif input_mode == "stim_only":
        observed_brain = (a == BRAIN) & (pos != S - 1)
        allowed &= ~observed_brain[None, :]

    if key_block is not None:
        allowed &= ~key_block.to(torch.bool).view(1, S)

    # A query with no admissible key would produce NaNs; let it attend to itself.
    empty = ~allowed.any(dim=1)
    allowed[empty, pos[empty]] = True

    out = torch.zeros(S, S)
    out.masked_fill_(~allowed, float("-inf"))
    return out


# ------------------------------------------------------------------ #
#  Spatial Mixer Head (paper Eqs. 11-13)                             #
# ------------------------------------------------------------------ #

class _SpatialSelfAttnBlock(nn.Module):
    """Multi-head self-attention over the K network tokens."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        y, attn = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + y)
        x = self.norm2(x + self.ffn(x))
        return x, attn


class _SpatialLinearMixBlock(nn.Module):
    """Linear-mixer ablation: a learned, state-independent K x K token mixing."""

    def __init__(self, K: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(K, K))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm1(x + torch.einsum("kj,bjd->bkd", self.mix.to(x.dtype), x))
        x = self.norm2(x + self.ffn(x))
        return x, None


class _SpatialMixerHead(nn.Module):
    """Per-step cross-network refinement of the preliminary whole-brain latent.

    The latent is split into its K network components, each mapped to a shared
    mixer dimension d_z by a network-specific W_in^k, mixed across networks,
    and mapped back by a network-specific W_out^k (Eqs. 11-13).
    """

    def __init__(self, D_f: int, z_splits: List[int], d_z: int,
                 nhead: int = 4, nlayers: int = 1, dropout: float = 0.1,
                 mixer: str = "attn"):
        super().__init__()
        assert sum(z_splits) == D_f, f"sum(z_splits)={sum(z_splits)} != D_f={D_f}"
        self.D_f = D_f
        self.z_splits = list(z_splits)
        self.K = len(z_splits)
        self.d_z = d_z
        self.nhead = nhead
        self.mixer = mixer

        self.in_projs = nn.ModuleList([nn.Linear(d, d_z) for d in z_splits])
        if mixer == "attn":
            self.layers = nn.ModuleList([
                _SpatialSelfAttnBlock(d_z, nhead, dropout) for _ in range(nlayers)
            ])
        elif mixer == "linear":
            self.layers = nn.ModuleList([
                _SpatialLinearMixBlock(self.K, d_z, dropout) for _ in range(nlayers)
            ])
        else:
            raise ValueError(f"Unknown mixer type: {mixer!r}")
        self.out_projs = nn.ModuleList([nn.Linear(d_z, d) for d in z_splits])
        self.last_attn = None

    def forward(self, z_seq: torch.Tensor, return_attn: bool = False):
        """z_seq: [B, T, D_f] -> [B, T, D_f] (and attn [B, T, nhead, K, K] if available)."""
        B, T, D_f = z_seq.shape
        assert D_f == self.D_f
        z_flat = z_seq.reshape(B * T, D_f)
        chunks = torch.split(z_flat, self.z_splits, dim=-1)

        tokens = [self.in_projs[k](chunks[k]) for k in range(self.K)]
        x = torch.stack(tokens, dim=1)  # [B*T, K, d_z]

        attn_last = None
        for layer in self.layers:
            x, attn_last = layer(x)

        outs = [self.out_projs[k](x[:, k, :]) for k in range(self.K)]
        z_ref = torch.cat(outs, dim=-1).view(B, T, D_f)

        self.last_attn = (attn_last.view(B, T, self.nhead, self.K, self.K)
                          if attn_last is not None else None)
        if return_attn:
            return z_ref, self.last_attn
        return z_ref


# ------------------------------------------------------------------ #
#  BrainVista predictor                                              #
# ------------------------------------------------------------------ #

class BrainVista(nn.Module):
    """Temporal causal Transformer with S2B mask + Spatial Mixer Head.

    forward(V, A, W, Z_hist) predicts the whole-brain latent token at time u:
        V, A, W : [B, L, D_m]       stimulus features at times u-L+1 .. u
                                    (target-aligned tau_s^u is the last row);
                                    a modality with D_m = 0 is inactive.
        Z_hist  : [B, L-1, D_f]     brain region tokens at times u-L+1 .. u-1
                                    (observed, or previously predicted during rollout).
    Returns tau_hat_f^u : [B, D_f].
    """

    def __init__(self,
                 Dv: int, Da: int, Dw: int,
                 Dz_in: int, Dz_out: int,
                 d_model: int = 256, nhead: int = 8, nlayers: int = 4,
                 dropout: float = 0.1, L: int = 40,
                 z_splits: Optional[List[int]] = None,
                 spatial_nhead: int = 4,
                 spatial_nlayers: int = 1,
                 spatial_dropout: float = 0.1,
                 spatial_dz: int = 128,
                 mixer: str = "attn",
                 mask: str = "s2b",
                 input_mode: str = "both",
                 brain_self_attn: bool = True):
        super().__init__()
        if mixer not in MIXER_TYPES:
            raise ValueError(f"mixer must be one of {MIXER_TYPES}, got {mixer!r}")
        self.Dv, self.Da, self.Dw = max(Dv, 0), max(Da, 0), max(Dw, 0)
        self.Dz_in, self.Dz_out = Dz_in, Dz_out
        self.d_model = d_model
        self.L = L
        self.z_splits = list(z_splits) if z_splits is not None else None
        self.mask_type = mask
        self.input_mode = input_mode
        self.brain_self_attn = brain_self_attn

        # Shared stimulus token (Eq. 4): tau_s = Proj_s([tau_vis; tau_aud; tau_text]).
        stim_in = self.Dv + self.Da + self.Dw
        if stim_in <= 0:
            raise ValueError("At least one stimulus modality must have a positive dimension")
        self.stim_proj = nn.Linear(stim_in, d_model)

        # Brain input projection Proj_b^in (d_brain -> d_model) and output Proj_b^out.
        self.z_proj_in = nn.Linear(Dz_in, d_model)
        self.out_proj = nn.Linear(d_model, Dz_out)

        # Learnable target query q_target occupying the brain slot at time u.
        self.target_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.target_query, std=0.02)

        # Token-type and absolute positional embeddings over the 2L positions.
        self.type_embed = nn.Embedding(2, d_model)
        S = 2 * L
        self.S = S
        self.pos_embed = nn.Embedding(S, d_model)
        self.register_buffer("type_ids", torch.tensor([STIM, BRAIN] * L, dtype=torch.long),
                             persistent=False)
        self.register_buffer("pos_ids", torch.arange(S, dtype=torch.long), persistent=False)
        self.register_buffer("attn_mask",
                             build_attention_mask(L, mask, input_mode, brain_self_attn),
                             persistent=False)

        # GPT-style Pre-LN Transformer (GELU, 4x FFN).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=False, activation="gelu", norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=nlayers,
                                              enable_nested_tensor=False)

        self.spatial_mixer = None
        if (mixer != "none" and z_splits is not None and len(z_splits) > 1
                and spatial_nlayers > 0 and (mixer == "linear" or spatial_nhead > 0)):
            self.spatial_mixer = _SpatialMixerHead(
                D_f=Dz_out, z_splits=z_splits, d_z=spatial_dz,
                nhead=spatial_nhead, nlayers=spatial_nlayers,
                dropout=spatial_dropout, mixer=mixer,
            )

    def _prep_stim_tokens(self, V, A, W):
        parts = []
        if V is not None and self.Dv > 0: parts.append(V)
        if A is not None and self.Da > 0: parts.append(A)
        if W is not None and self.Dw > 0: parts.append(W)
        return self.stim_proj(torch.cat(parts, dim=-1))

    def occlusion_mask(self, brain_lag: int) -> torch.Tensor:
        """Mask that hides the observed brain token `brain_lag` steps before u (1..L-1)."""
        if not 1 <= brain_lag <= self.L - 1:
            raise ValueError(f"brain_lag must be in [1, {self.L - 1}], got {brain_lag}")
        key_block = torch.zeros(self.S, dtype=torch.bool)
        key_block[2 * (self.L - 1 - brain_lag) + 1] = True
        return build_attention_mask(self.L, self.mask_type, self.input_mode,
                                    self.brain_self_attn, key_block).to(self.attn_mask.device)

    def forward(self, V, A, W, Z_hist, return_attn: bool = False,
                attn_mask: Optional[torch.Tensor] = None):
        B, L_v = V.shape[0], V.shape[1]
        if L_v != self.L:
            raise ValueError(f"Expected stimulus window length {self.L}, got {L_v}")
        if Z_hist.shape[1] != self.L - 1:
            raise ValueError(f"Expected brain history length {self.L - 1}, got {Z_hist.shape[1]}")

        stim_tokens = self._prep_stim_tokens(V, A, W)                      # [B, L,   d]
        brain_obs = self.z_proj_in(Z_hist)                                 # [B, L-1, d]
        target_q = self.target_query.expand(B, 1, self.d_model).to(brain_obs.dtype)
        brain_tokens = torch.cat([brain_obs, target_q], dim=1)             # [B, L,   d]

        seq = torch.stack([stim_tokens, brain_tokens], dim=2).view(B, self.S, self.d_model)
        seq = seq + self.type_embed(self.type_ids)[None] + self.pos_embed(self.pos_ids)[None]

        mask = self.attn_mask if attn_mask is None else attn_mask
        h = self.backbone(seq.transpose(0, 1), mask).transpose(0, 1)       # [B, S, d]
        z_pred = self.out_proj(h[:, -1, :])                                # [B, D_f]

        spatial_attn = None
        if self.spatial_mixer is not None:
            z_seq, spatial_attn = self.spatial_mixer(z_pred.unsqueeze(1), return_attn=True)
            z_pred = z_seq.squeeze(1)
        if return_attn:
            return z_pred, {"spatial_attn": spatial_attn}
        return z_pred


# Name used by the original release; checkpoints are interchangeable.
InterleavedStim2BrainSpatialPredictor = BrainVista


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
