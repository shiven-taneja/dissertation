import torch
import torch.nn as nn

class _NormLinearReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.fc = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)          # (B, T, F)
        return self.relu(self.fc(x))

class UTransNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int = 3,
        n_transformer_heads: int = 8,
        n_transformer_layers: int = 1,
    ):
        super().__init__()
        self.enc1 = _NormLinearReLU(input_dim, 64)
        self.enc2 = _NormLinearReLU(64, 128)
        self.enc3 = _NormLinearReLU(128, 256)

        t_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=n_transformer_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(t_layer, num_layers=n_transformer_layers)

        self.dec1 = _NormLinearReLU(256, 128)
        self.dec2 = _NormLinearReLU(128, 64)

        self.act_head = nn.Linear(64, n_actions)  # action logits only
        self.weight_head = nn.Sequential(         # independent weight head
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        t  = self.transformer(e3)

        d1 = self.dec1(t + e3)
        d2 = self.dec2(d1 + e2)
        final = d2 + e1

        pooled = final.mean(dim=1)               # (B, 64)
        q_values = self.act_head(pooled)         # (B, n_actions)
        act_weight = self.weight_head(pooled).squeeze(-1)  # (B,)
        return q_values, act_weight
