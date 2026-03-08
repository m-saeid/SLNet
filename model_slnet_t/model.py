# model/model.py
"""
Clean model wrapper.
  - Simple input: (B, F, N) → output: (B, N, num_classes)
"""
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class SegModel(nn.Module):
    """
    Hierarchical 3D segmentation model for S3DIS.

    Input contract:
        x: (B, F, N)   where F = cfg.in_channels (default 6: rgb + norm_xyz)
        channel-first, as produced by the training loop after .permute(0,2,1)

    Output:
        logits: (B, N, num_classes)
    """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F, N) → logits: (B, N, C)"""
        xyz_list, feat_list = self.encoder(x)
        return self.decoder(xyz_list, feat_list)