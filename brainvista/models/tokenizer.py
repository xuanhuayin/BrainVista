# brainvista/models/tokenizer.py
"""Network-wise tokenizer--detokenizer pairs (paper §3.3, Eq. 6; Appendix A.9).

For every functional network k, an independent two-layer MLP autoencoder maps
the normalized model-space signal x^{k,t} in R^{P_k} to a brain region token
tau_f^{t,k} in R^{d_k} and back. Parameters are never shared across networks.
The pairs are pretrained with reconstruction only (Eq. 14) and frozen before
the temporal predictor is trained.
"""
import math

import torch.nn as nn


def default_token_dim(P_k: int, ratio: float = 0.8) -> int:
    """Latent size d_k = ceil(ratio * P_k) (e.g. 160 -> 128, 512 -> 410)."""
    return max(1, int(math.ceil(ratio * P_k - 1e-9)))


class EncAE(nn.Module):
    """Tokenizer h_{phi_k}^k: x^{k,t} (P_k) -> tau_f^{t,k} (d_k)."""

    def __init__(self, D_in, z_dim=128, hidden=768, dropout=0.1):
        super().__init__()
        self.D_in = D_in
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.LayerNorm(D_in),
            nn.Linear(D_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class Dec(nn.Module):
    """Detokenizer g_{psi_k}^k: tau_f^{t,k} (d_k) -> x_tilde^{k,t} (P_k)."""

    def __init__(self, z_dim=128, D_out=512, hidden=768, dropout=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.D_out = D_out
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, D_out),
        )

    def forward(self, z):
        return self.net(z)
