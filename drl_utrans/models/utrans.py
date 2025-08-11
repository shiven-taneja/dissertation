import torch
import torch.nn as nn


class _NormLinearReLU(nn.Module):
    """Red arrow in the figure:  LayerNorm ➜ Linear ➜ ReLU."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # self.bn   = nn.BatchNorm1d(in_dim, affine=True)
        self.ln = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.fc   = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_dim)
        # return self.block(x)
        # x = self.bn(x.transpose(1,2)).transpose(1,2)  # (B,T,F) -> BN -> (B,T,F)
        x = self.ln(x)  # (B, T, F)
        return self.relu(self.fc(x))


class UTransNet(nn.Module):
    """
    U-Net-style encoder/decoder with a Transformer bottleneck.

    ── Input  ➜  1×64  ➜  1×128  ➜  1×256  ➜  Transformer  ─┐
                     ↑        ↑         ↑                 │
                     └────────┴─────────┴─────────────────┘  (skip-adds)
                               ↓         ↓         ↓
                          1×256 ▸ 1×128 ▸ 1×64 ▸ heads
    """
    def __init__(
        self,
        input_dim: int,          # feature dimension per time step
        n_actions: int = 3,      # Buy / Sell / Hold
        n_transformer_heads: int = 8,
        n_transformer_layers: int = 1,
    ):
        super().__init__()

        # --- Encoder ---------------------------------------------------------
        self.enc1 = _NormLinearReLU(input_dim, 64)    # 1 × 64
        self.enc2 = _NormLinearReLU(64, 128)          # 1 × 128
        self.enc3 = _NormLinearReLU(128, 256)         # 1 × 256

        # --- Transformer bottleneck -----------------------------------------
        t_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=n_transformer_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            t_layer, num_layers=n_transformer_layers
        )

        # --- Decoder (add-skip ➜ Norm-Linear-ReLU) ---------------------------
        self.dec1 = _NormLinearReLU(256, 128)         # 1 × 128
        self.dec2 = _NormLinearReLU(128, 64)          # 1 × 64

        # --- Dual heads ------------------------------------------------------
        # global average over the sequence length → 1 × 64 vector
        self.act_head   = nn.Linear(65, n_actions)    # categorical action
        self.weight_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()                              # 0-1 weighting
        )

    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_dim)
        Returns:
            q_values   : (batch, n_actions)
            act_weight : (batch,) ─ scalar in [0, 1]
        """
        # Encoder -------------------------------------------------------------
        e1 = self.enc1(x)     # 1 × 64
        e2 = self.enc2(e1)    # 1 × 128
        e3 = self.enc3(e2)    # 1 × 256

        # Transformer ---------------------------------------------------------
        t = self.transformer(e3)  # (batch, seq_len, 256)

        # Decoder with additive skips ----------------------------------------
        d1_in = t + e3            # add 1 × 256 skip
        d1    = self.dec1(d1_in)  # → 1 × 128

        d2_in = d1 + e2           # add 1 × 128 skip
        d2    = self.dec2(d2_in)  # → 1 × 64

        final = d2 + e1           # last skip (no further reduction)

        # Heads ---------------------------------------------------------------
        # pooled = final.mean(dim=1)        # global average over sequence
        # pooled = final[:, -1, :]  # take the last time step (as in the paper)
        # pooled = final.squeeze(1)  # (batch, 64)
        pooled = final.mean(dim=1)  # global average over sequence (changed from last timestep)
        act_weight = self.weight_head(pooled).squeeze(-1)
        pooled_with_weight = torch.cat([pooled, act_weight.unsqueeze(1)], dim=1)  # (batch, 65)
        q_values   = self.act_head(pooled_with_weight)

        # q_values   = self.act_head(pooled)


        return q_values, act_weight
