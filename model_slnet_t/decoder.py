# model/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .encoder import knn, gather_nd, gn, conv_bn_relu


# ─── Feature Propagation ──────────────────────────────────────────────────────

class FP(nn.Module):
    """
    Inverse-distance-weighted interpolation + MLP.
    Propagates features from a coarser level to a finer level.
    """
    def __init__(self, coarse_c: int, fine_c: int, out_c: int, k: int = 3):
        super().__init__()
        in_c = coarse_c + fine_c
        mid  = max(in_c // 2, out_c)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, mid, 1, bias=False),
            gn(mid),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, out_c, 1, bias=False),
            gn(out_c),
            nn.ReLU(inplace=True),
        )
        self.k = k

    def forward(self, fine_xyz: torch.Tensor, coarse_xyz: torch.Tensor,
                fine_feat: torch.Tensor, coarse_feat: torch.Tensor) -> torch.Tensor:
        """
        fine_xyz    : (B, N, 3)
        coarse_xyz  : (B, S, 3)   S < N
        fine_feat   : (B, N, C_fine)
        coarse_feat : (B, S, C_coarse)
        → (B, N, out_c)
        """
        B, N, _ = fine_xyz.shape
        S = coarse_xyz.shape[1]

        if S == 1:
            interp = coarse_feat.expand(B, N, -1)
        else:
            idx = knn(coarse_xyz, fine_xyz, self.k)          # (B,N,k)
            neighbors = gather_nd(coarse_xyz, idx)            # (B,N,k,3)
            # squared distances for inverse-distance weighting
            diff = fine_xyz.unsqueeze(2) - neighbors          # (B,N,k,3)
            dist2 = (diff ** 2).sum(-1).clamp(min=1e-8)      # (B,N,k)
            w = 1.0 / dist2                                   # (B,N,k)
            w = w / w.sum(dim=-1, keepdim=True)               # normalize
            # gather + weighted sum
            n_feat = gather_nd(coarse_feat, idx)              # (B,N,k,C_coarse)
            interp = (w.unsqueeze(-1) * n_feat).sum(dim=2)   # (B,N,C_coarse)

        combined = torch.cat([fine_feat, interp], dim=-1)    # (B,N,C_fine+C_coarse)
        combined = combined.permute(0, 2, 1).contiguous()    # (B,C,N)
        out = self.mlp(combined)                              # (B,out_c,N)
        return out.permute(0, 2, 1).contiguous()              # (B,N,out_c)


# ─── Full Decoder ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    4-stage FP decoder, channel-deterministic from config.
    """
    def __init__(self, cfg):
        super().__init__()
        # Encoder dims: [d0, d1, d2, d3, d4]  (shallow→deep)
        enc_dims   = [cfg.encoder_dims[0]] + list(cfg.encoder_dims)
        # Decoder output dims (deep→shallow direction)
        dec_dims   = list(cfg.decoder_dims)   # e.g. [256, 128, 64, 64]
        num_cls    = cfg.num_classes

        # Build FP blocks deep→shallow
        # FP[0]: enc[4] + enc[3] → dec[0]
        # FP[1]: dec[0] + enc[2] → dec[1]
        # FP[2]: dec[1] + enc[1] → dec[2]
        # FP[3]: dec[2] + enc[0] → dec[3]
        n_stages = len(enc_dims) - 1  # 4
        self.fp_blocks = nn.ModuleList()

        coarse_c = enc_dims[-1]     # deepest encoder dim
        for i in range(n_stages):
            fine_c  = enc_dims[n_stages - 1 - i]
            out_c   = dec_dims[i]
            self.fp_blocks.append(FP(coarse_c, fine_c, out_c))
            coarse_c = out_c        # next FP uses this output as "coarse"

        # Classification head
        final_c = dec_dims[-1]      # 64
        self.head = nn.Sequential(
            nn.Conv1d(final_c, final_c, 1, bias=False),
            gn(final_c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Conv1d(final_c, num_cls, 1),
        )

        self.out_dims = enc_dims    # expose for inspection

    def forward(self, xyz_list: List[torch.Tensor],
                feat_list: List[torch.Tensor]) -> torch.Tensor:
        """
        xyz_list  : len 5, each (B, N_i, 3)  shallow→deep
        feat_list : len 5, each (B, N_i, C_i) shallow→deep
        → logits : (B, N, num_classes)  — N is original resolution
        """
        # Start from deepest
        cur_feat = feat_list[-1]    # (B, N_deep, C_deep)
        cur_xyz  = xyz_list[-1]

        n_stages = len(self.fp_blocks)
        for i, fp in enumerate(self.fp_blocks):
            level = n_stages - 1 - i   # fine level index: 3, 2, 1, 0
            fine_xyz  = xyz_list[level]
            fine_feat = feat_list[level]
            cur_feat  = fp(fine_xyz, cur_xyz, fine_feat, cur_feat)
            cur_xyz   = fine_xyz

        # cur_feat: (B, N, final_c)
        out = cur_feat.permute(0, 2, 1).contiguous()    # (B,C,N)
        logits = self.head(out)                         # (B,num_classes,N)
        return logits.permute(0, 2, 1).contiguous()     # (B,N,num_classes)