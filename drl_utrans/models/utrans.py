import torch
import torch.nn as nn
import torch.nn.functional as F

class UTransModel(nn.Module):
    """
    UTrans Model: Combines U-Net style encoder–decoder with a Transformer bottleneck.
    Outputs action logits for 3 actions (buy, sell, hold) and an action weight (0–1).
    Input: tensor of shape (batch_size, window_size, feature_dim).
    """
    def __init__(self, window_size: int = 12, feature_dim: int = 14,
                 d_model: int = 128, nhead: int = 4):
        super(UTransModel, self).__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        # Encoder: U-Net downsampling path
        self.enc_conv1a = nn.Conv1d(feature_dim, 32, kernel_size=3, padding=1)
        self.enc_conv1b = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # length 12 -> 6
        self.enc_conv2a = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_conv2b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # length 6 -> 3
        # Bottleneck: Transformer layer on compressed sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=256,
                                                  batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # Project downsampled features to d_model and back for Transformer
        self.enc_project = nn.Conv1d(64, d_model, kernel_size=1)
        self.dec_project = nn.Conv1d(d_model, 64, kernel_size=1)
        # Decoder: U-Net upsampling path with skip connections
        self.upconv2 = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2)  # 3 -> 6
        self.dec_conv2a = nn.Conv1d(64 + 64, 64, kernel_size=3, padding=1)  # concat skip2  # skip connection:contentReference[oaicite:0]{index=0}
        self.dec_conv2b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)  # 6 -> 12
        self.dec_conv1a = nn.Conv1d(32 + 32, 32, kernel_size=3, padding=1)  # concat skip1  # skip connection:contentReference[oaicite:1]{index=1}
        self.dec_conv1b = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        # Output heads: flatten and fully-connected layers
        self.flatten = nn.Flatten()
        self.out_action = nn.Linear(32 * window_size, 3)
        self.out_weight = nn.Linear(32 * window_size, 1)
        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize Conv and Linear weights (Kaiming for conv, Xavier for linear)."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, window_size, feature_dim)
        # Permute to (batch, channels, length) for Conv1d
        x = x.permute(0, 2, 1)  # -> (batch, feature_dim, window_size)
        # Encoder
        x1 = F.relu(self.enc_conv1a(x))
        x1 = F.relu(self.enc_conv1b(x1))
        skip1 = x1  # (batch, 32, 12), skip connection 1
        x2 = self.pool1(x1)      # -> (batch, 32, 6)
        x2 = F.relu(self.enc_conv2a(x2))
        x2 = F.relu(self.enc_conv2b(x2))
        skip2 = x2  # (batch, 64, 6), skip connection 2
        x3 = self.pool2(x2)      # -> (batch, 64, 3)
        # Transformer bottleneck on compressed sequence
        x3 = self.enc_project(x3)          # (batch, d_model, 3)
        x3 = x3.permute(0, 2, 1)           # (batch, 3, d_model)
        x3 = self.transformer(x3)          # (batch, 3, d_model)
        x3 = x3.permute(0, 2, 1)           # (batch, d_model, 3)
        x3 = self.dec_project(x3)          # (batch, 64, 3)
        # Decoder with skip connections
        x4 = self.upconv2(x3)              # -> (batch, 64, 6)
        # Align shapes if needed (in case of any rounding issues)
        if x4.shape[2] != skip2.shape[2]:
            diff = skip2.shape[2] - x4.shape[2]
            if diff > 0:
                x4 = F.pad(x4, (0, diff))
            else:
                x4 = x4[:, :, :skip2.shape[2]]
        x4 = torch.cat([x4, skip2], dim=1)  # concat skip2 features
        x4 = F.relu(self.dec_conv2a(x4))
        x4 = F.relu(self.dec_conv2b(x4))    # (batch, 64, 6)
        x5 = self.upconv1(x4)              # -> (batch, 32, 12)
        if x5.shape[2] != skip1.shape[2]:
            diff = skip1.shape[2] - x5.shape[2]
            if diff > 0:
                x5 = F.pad(x5, (0, diff))
            else:
                x5 = x5[:, :, :skip1.shape[2]]
        x5 = torch.cat([x5, skip1], dim=1)  # concat skip1 features
        x5 = F.relu(self.dec_conv1a(x5))
        x5 = F.relu(self.dec_conv1b(x5))    # (batch, 32, 12)
        # Flatten and produce outputs
        flat = self.flatten(x5)            # (batch, 32*12)
        action_logits = self.out_action(flat)       # (batch, 3) raw logits for actions
        action_weight = torch.sigmoid(self.out_weight(flat))  # (batch, 1) in [0,1]
        return action_logits, action_weight
