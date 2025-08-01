import torch
import torch.nn as nn

class UTransNet(nn.Module):
    def __init__(self, input_dim: int, window_size: int):
        super(UTransNet, self).__init__()
        
        # --- Encoder Path ---
        self.enc1 = nn.Sequential(nn.Conv1d(input_dim, 64, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool1d(2)

        # --- Bottleneck with Transformer Layer ---
        transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        # --- Decoder Path ---
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        # For the skip connection cat([d2, p1]), the input to dec2 is 64+64=128 channels
        self.dec2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        # **BUG 2 FIX**: For cat([d1, e1]), the input is 32+64=96 channels.
        # The original code incorrectly had 64 here.
        self.dec1 = nn.Sequential(nn.Conv1d(96, 32, kernel_size=3, padding=1), nn.ReLU())
        
        # --- Dual Output Heads ---
        # The flattened dimension comes from the output of the final decoder layer (32 channels * window_size length)
        self.action_head = nn.Linear(32 * window_size, 3) # Q-values for Buy, Sell, Hold
        self.weight_head = nn.Sequential(
            nn.Linear(32 * window_size, 1),
            nn.Sigmoid() # To constrain weight between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input x has shape (batch_size, window_size, feature_dim)
        
        # **BUG 1 FIX**: Permute to (batch_size, feature_dim, window_size) for Conv1d
        # The original code `x.unsqueeze(1)` was incorrect and caused the runtime error.
        x = x.permute(0, 2, 1)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        # Transformer expects (batch, seq_len, features)
        p2_permuted = p2.permute(0, 2, 1) 
        trans_out = self.transformer_encoder(p2_permuted)
        trans_out_permuted = trans_out.permute(0, 2, 1)

        # Decoder with Skip Connections
        d2 = self.up2(trans_out_permuted)
        d2 = torch.cat([d2, p1], dim=1) # Shape becomes (B, 128, L/2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # Shape becomes (B, 96, L)
        d1 = self.dec1(d1)

        # Flatten for final layers
        flat = d1.view(d1.size(0), -1)

        # Get outputs
        q_values = self.action_head(flat)
        action_weight = self.weight_head(flat)

        return q_values, action_weight.squeeze(-1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class UTransModel(nn.Module):
#     """
#     Exact topology of U‑Trans (Yang et al., 2023).
#     Input : (B, window_size, feature_dim)
#     Output: action_logits (B,3), action_weight (B,1 in [0,1])
#     """
#     def __init__(self,
#                  window_size: int = 12,
#                  feature_dim: int = 1,
#                  d_model: int   = 256,
#                  nhead: int     = 8,
#                  ff_dim: int    = 512,
#                  dropout: float = 0.1):
#         super().__init__()
#         flat_dim = window_size * feature_dim        # 1 × T

#         # ---------- Encoder ----------
#         self.enc0_norm = nn.LayerNorm(flat_dim)     # Embedding norm
#         self.enc1 = nn.Linear(flat_dim, 64)
#         self.enc1_norm = nn.LayerNorm(64)

#         self.enc2 = nn.Linear(64, 128)
#         self.enc2_norm = nn.LayerNorm(128)

#         self.enc3 = nn.Linear(128, d_model)
#         self.enc3_norm = nn.LayerNorm(d_model)

#         # ---------- Transformer bottleneck ----------
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead,
#             dim_feedforward=ff_dim,
#             dropout=dropout,
#             batch_first=True)
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

#         # ---------- Decoder ----------
#         self.dec1_norm = nn.LayerNorm(d_model)      # input: z + skip3
#         self.dec1 = nn.Linear(d_model, 128)

#         self.dec2_norm = nn.LayerNorm(128)          # input: y1 + skip2
#         self.dec2 = nn.Linear(128, 64)

#         self.dec3_norm = nn.LayerNorm(64)           # input: y2 + skip1

#         # ---------- Heads ----------
#         self.head_action = nn.Linear(64, 3)
#         self.head_weight = nn.Linear(64, 1)

#         self._init_weights()

#     # ---------------------------------------------------------------
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
#                 nn.init.zeros_(m.bias)

#     # ---------------------------------------------------------------
#     def forward(self, x):
#         # x: (B, L, F)  → flatten to (B, L·F)
#         B = x.size(0)
#         v  = self.enc0_norm(x.reshape(B, -1))

#         # Encoder 64
#         s1 = F.relu(self.enc1_norm(self.enc1(v)))     # skip‑1

#         # Encoder 128
#         s2 = F.relu(self.enc2_norm(self.enc2(s1)))    # skip‑2

#         # Encoder 256
#         s3 = F.relu(self.enc3_norm(self.enc3(s2)))    # skip‑3

#         # Transformer (sequence length = 1 token)
#         z = self.transformer(s3.unsqueeze(1)).squeeze(1)  # (B,256)

#         # Decoder: 256 → 128 (+ skip‑3)
#         y1 = F.relu(self.dec1(self.dec1_norm(z + s3)))

#         # Decoder: 128 → 64  (+ skip‑2)
#         y2 = F.relu(self.dec2(self.dec2_norm(y1 + s2)))

#         # Fusion: add skip‑1
#         fused = F.relu(self.dec3_norm(y2 + s1))       # (B,64)

#         # Heads
#         action_logits = self.head_action(fused)
#         action_weight = torch.sigmoid(self.head_weight(fused))

#         w_exp = action_weight.expand_as(action_logits[:, :1])  # (B,1) → (B,1)
#         buy_q  = action_logits[:, 0:1] * w_exp     # (B,1)
#         sell_q = action_logits[:, 1:2] * w_exp     # (B,1)
#         hold_q = action_logits[:, 2:3]             # (B,1) unchanged
#         q_mod  = torch.cat([buy_q, sell_q, hold_q], dim=1)

#         return q_mod, action_weight


