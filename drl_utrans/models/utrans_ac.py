# drl_utrans/models/utrans_ac.py
from __future__ import annotations
import torch
import torch.nn as nn

class _NormLinearReLU(nn.Module):
    """LayerNorm ➜ Linear ➜ ReLU."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.fc = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        return self.relu(self.fc(x))

class UTransActorCritic(nn.Module):
    """
    U-Net style encoder/decoder with Transformer bottleneck and three heads:
    - action logits   (3: Buy, Sell, Hold)
    - weight logits   (n_weight_bins in [0,1])
    - state value     (scalar)

    The policy is factorized:  π(a, w) = π_a(a | s) * π_w(w | s). If a==Hold,
    we ignore π_w in the loss (masking its log-prob & entropy).
    """
    def __init__(
        self,
        input_dim: int,
        n_actions: int = 3,
        n_weight_bins: int = 11,
        n_transformer_heads: int = 8,
        n_transformer_layers: int = 1,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_weight_bins = n_weight_bins

        # Encoder
        self.enc1 = _NormLinearReLU(input_dim, 64)
        self.enc2 = _NormLinearReLU(64, 128)
        self.enc3 = _NormLinearReLU(128, 256)

        # Transformer bottleneck
        t_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=n_transformer_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(t_layer, num_layers=n_transformer_layers)

        # Decoder
        self.dec1 = _NormLinearReLU(256, 128)
        self.dec2 = _NormLinearReLU(128, 64)

        # Heads
        self.action_head = nn.Linear(64, n_actions)
        self.weight_head = nn.Linear(64, n_weight_bins)
        self.value_head  = nn.Linear(64, 1)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        t = self.transformer(e3)
        d1 = self.dec1(t + e3)
        d2 = self.dec2(d1 + e2)
        final = d2 + e1
        pooled = final[:, -1, :]  # use last time step
        return pooled

    def forward(self, x: torch.Tensor):
        h = self._trunk(x)
        action_logits = self.action_head(h)
        weight_logits = self.weight_head(h)
        value = self.value_head(h).squeeze(-1)
        return action_logits, weight_logits, value