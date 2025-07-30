import torch
import torch.nn as nn
import torch.nn.functional as F

class UTransModel(nn.Module):
    """
    Exact topology of U‑Trans (Yang et al., 2023).
    Input : (B, window_size, feature_dim)
    Output: action_logits (B,3), action_weight (B,1 in [0,1])
    """
    def __init__(self,
                 window_size: int = 12,
                 feature_dim: int = 1,
                 d_model: int   = 256,
                 nhead: int     = 8,
                 ff_dim: int    = 512,
                 dropout: float = 0.1):
        super().__init__()
        flat_dim = window_size * feature_dim        # 1 × T

        # ---------- Encoder ----------
        self.enc0_norm = nn.LayerNorm(flat_dim)     # Embedding norm
        self.enc1 = nn.Linear(flat_dim, 64)
        self.enc1_norm = nn.LayerNorm(64)

        self.enc2 = nn.Linear(64, 128)
        self.enc2_norm = nn.LayerNorm(128)

        self.enc3 = nn.Linear(128, d_model)
        self.enc3_norm = nn.LayerNorm(d_model)

        # ---------- Transformer bottleneck ----------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

        # ---------- Decoder ----------
        self.dec1_norm = nn.LayerNorm(d_model)      # input: z + skip3
        self.dec1 = nn.Linear(d_model, 128)

        self.dec2_norm = nn.LayerNorm(128)          # input: y1 + skip2
        self.dec2 = nn.Linear(128, 64)

        self.dec3_norm = nn.LayerNorm(64)           # input: y2 + skip1

        # ---------- Heads ----------
        self.head_action = nn.Linear(64, 3)
        self.head_weight = nn.Linear(64, 1)

        self._init_weights()

    # ---------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------
    def forward(self, x):
        # x: (B, L, F)  → flatten to (B, L·F)
        B = x.size(0)
        v  = self.enc0_norm(x.reshape(B, -1))

        # Encoder 64
        s1 = F.relu(self.enc1_norm(self.enc1(v)))     # skip‑1

        # Encoder 128
        s2 = F.relu(self.enc2_norm(self.enc2(s1)))    # skip‑2

        # Encoder 256
        s3 = F.relu(self.enc3_norm(self.enc3(s2)))    # skip‑3

        # Transformer (sequence length = 1 token)
        z = self.transformer(s3.unsqueeze(1)).squeeze(1)  # (B,256)

        # Decoder: 256 → 128 (+ skip‑3)
        y1 = F.relu(self.dec1(self.dec1_norm(z + s3)))

        # Decoder: 128 → 64  (+ skip‑2)
        y2 = F.relu(self.dec2(self.dec2_norm(y1 + s2)))

        # Fusion: add skip‑1
        fused = F.relu(self.dec3_norm(y2 + s1))       # (B,64)

        # Heads
        action_logits = self.head_action(fused)
        action_weight = torch.sigmoid(self.head_weight(fused))

        return action_logits, action_weight