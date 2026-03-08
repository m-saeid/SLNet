# model/encoder.py

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

import torch.library



# ─── Optional fast ops ────────────────────────────────────────────────────────
try:
    from pointnet2_ops import pointnet2_utils as _pn2
    _HAS_PN2 = True
except ImportError:
    _HAS_PN2 = False

# pytorch3d is no longer used for knn (not cudagraph-safe).
# Keep the import only if you need it elsewhere; otherwise drop it.


# ─── Geometry primitives ──────────────────────────────────────────────────────
#
# Both ops are registered as opaque custom ops so that torch.compile /
# torch.inductor can:
#   1. Skip tracing into them (Dynamo sees them as black boxes).
#   2. Use the register_fake shape-function for AOT-autograd / fake-tensor
#      propagation.
#   3. Record CUDA graphs safely (no host↔device syncs inside).


# ── FPS ───────────────────────────────────────────────────────────────────────

@torch.library.custom_op("custom::fps", mutates_args=())
def _fps_op(xyz: torch.Tensor, n: int) -> torch.Tensor:
    """Furthest-point sampling.  xyz (B,N,3) → idx (B,n) long."""
    if _HAS_PN2:
        # Fast CUDA kernel from pointnet2_ops — cudagraph-safe.
        return _pn2.furthest_point_sample(xyz.contiguous(), n).long()
    # Pure-torch fallback — O(N·n) but correct and always available.
    B, N, _ = xyz.shape
    idx  = torch.zeros(B, n, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    for i in range(n):
        idx[:, i] = farthest
        c    = xyz[torch.arange(B, device=xyz.device), farthest].unsqueeze(1)
        d    = ((xyz - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(1)
    return idx

@_fps_op.register_fake
def _fps_fake(xyz: torch.Tensor, n: int) -> torch.Tensor:
    B = xyz.shape[0]
    return torch.zeros(B, n, dtype=torch.long, device=xyz.device)


# ── KNN ───────────────────────────────────────────────────────────────────────

@torch.library.custom_op("custom::knn", mutates_args=())
def _knn_op(query: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
    """K-nearest neighbours.  query (B,S,3), xyz (B,N,3) → idx (B,S,k) long.

    Uses torch.cdist + topk — a single ATen/CUDA kernel call with no
    host↔device synchronisation, so it is safe inside CUDA-graph capture.
    pytorch3d's knn_points does a .min() host-sync and is NOT safe here.
    """
    d = torch.cdist(query, xyz)                             # (B, S, N)
    return d.topk(k, dim=-1, largest=False)[1].long()      # (B, S, k)

@_knn_op.register_fake
def _knn_fake(query: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
    B, S = query.shape[0], query.shape[1]
    return torch.zeros(B, S, k, dtype=torch.long, device=query.device)


# ── Public wrappers (keep the original call signatures) ───────────────────────

def fps(xyz: torch.Tensor, n: int) -> torch.Tensor:
    """xyz (B,N,3) → idx (B,n) long"""
    return torch.ops.custom.fps(xyz, n)

def knn(xyz: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    """xyz (B,N,3), query (B,S,3) → idx (B,S,k) long"""
    return torch.ops.custom.knn(query, xyz, k)


def gather_nd(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x  : (B, N, C)
    idx: (B, M)     → (B, M, C)
         (B, M, k)  → (B, M, k, C)
    """
    B, N, C = x.shape
    if idx.dim() == 2:
        M = idx.shape[1]
        return x.gather(1, idx.unsqueeze(-1).expand(B, M, C))
    M, k = idx.shape[1], idx.shape[2]
    flat = idx.reshape(B, M * k)
    out = x.gather(1, flat.unsqueeze(-1).expand(B, M * k, C))
    return out.reshape(B, M, k, C)


# ─── Normalization helpers ─────────────────────────────────────────────────────

def gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(max(1, g), channels)


def conv_bn_relu(in_c: int, out_c: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, 1, bias=False),
        gn(out_c, groups),
        nn.ReLU(inplace=True),
    )


# ─── Local operators ──────────────────────────────────────────────────────────

class LocalMLPBlock(nn.Module):
    """
    PointNet2-style local MLP aggregation.
    For each sampled point: gather k neighbors, apply shared MLP, max-pool.
    Fast and effective for early stages where local geometry details matter.
    """
    def __init__(self, in_c: int, out_c: int, k: int):
        super().__init__()
        mid = max(out_c // 2, in_c)
        # input: features(C) + relative_xyz(3)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_c + 3, mid, 1, bias=False),
            nn.GroupNorm(max(1, min(8, mid)), mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_c, 1, bias=False),
            nn.GroupNorm(max(1, min(8, out_c)), out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, center_xyz: torch.Tensor, center_feat: torch.Tensor,
                neigh_xyz: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
        """
        center_xyz  : (B, S, 3)
        center_feat : (B, S, C)  — unused here, kept for API consistency
        neigh_xyz   : (B, S, k, 3)
        neigh_feat  : (B, S, k, C)
        → out: (B, S, out_c)
        """
        rel = neigh_xyz - center_xyz.unsqueeze(2)      # (B,S,k,3)
        x = torch.cat([neigh_feat, rel], dim=-1)       # (B,S,k,C+3)
        x = x.permute(0, 3, 1, 2).contiguous()        # (B,C+3,S,k)
        x = self.mlp(x)                                # (B,out_c,S,k)
        x = x.max(dim=-1)[0]                           # (B,out_c,S)  max pool over k
        return x.permute(0, 2, 1).contiguous()         # (B,S,out_c)


class LocalAttentionBlock(nn.Module):
    """
    Point Transformer v1 local attention.
    Subtraction-based attention (not dot-product) — more stable for point clouds.
    Position encoding adds relative geometry into both attention weights and values.

    Reference: Zhao et al. "Point Transformer" ICCV 2021.
    """
    def __init__(self, in_c: int, out_c: int, k: int):
        super().__init__()
        self.out_c = out_c
        self.q = nn.Linear(in_c, out_c, bias=False)
        self.k = nn.Linear(in_c, out_c, bias=False)
        self.v = nn.Linear(in_c, out_c, bias=False)
        # Position encoding: relative xyz → out_c (shared for attn weights + values)
        self.pos_enc = nn.Sequential(
            nn.Linear(3, out_c, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_c, out_c, bias=False),
        )
        # Attention weight MLP (applied after subtraction + pos_enc)
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_c, max(out_c // 4, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(out_c // 4, 8), out_c, bias=False),
        )
        # Output projection + residual
        self.proj = nn.Linear(out_c, out_c, bias=False)
        self.norm = nn.LayerNorm(out_c)
        self.res = nn.Linear(in_c, out_c, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, center_xyz: torch.Tensor, center_feat: torch.Tensor,
                neigh_xyz: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
        """
        center_xyz  : (B, S, 3)
        center_feat : (B, S, C)
        neigh_xyz   : (B, S, k, 3)
        neigh_feat  : (B, S, k, C)
        → out: (B, S, out_c)
        """
        B, S, _ = center_xyz.shape
        k = neigh_xyz.shape[2]

        q = self.q(center_feat).unsqueeze(2)                  # (B,S,1,out_c)
        kk = self.k(neigh_feat)                               # (B,S,k,out_c)
        v  = self.v(neigh_feat)                               # (B,S,k,out_c)

        rel = neigh_xyz - center_xyz.unsqueeze(2)             # (B,S,k,3)
        pe = self.pos_enc(rel)                                # (B,S,k,out_c)

        # Subtraction attention (PT v1): w = MLP(q - k + pe)
        w = self.attn_mlp(q - kk + pe)                       # (B,S,k,out_c)
        w = F.softmax(w, dim=2)                               # (B,S,k,out_c)  per-neighbor weights

        # Weighted sum of (v + pe)
        out = (w * (v + pe)).sum(dim=2)                       # (B,S,out_c)
        out = self.proj(out)
        out = self.norm(out + self.res(center_feat))
        return out                                            # (B,S,out_c)


# ─── One encoder stage ────────────────────────────────────────────────────────

class EncoderStage(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int,
                 downsample_ratio: int, use_attention: bool = False):
        super().__init__()
        self.k = k
        self.ratio = downsample_ratio
        block_cls = LocalAttentionBlock if use_attention else LocalMLPBlock
        self.block = block_cls(in_c, out_c, k)
        # Post-block MLP to refine features
        self.post = conv_bn_relu(out_c, out_c)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz   : (B, N, 3)
        feats : (B, N, C_in)
        → (sampled_xyz (B,S,3), out_feats (B,S,C_out))
        """
        B, N, _ = xyz.shape
        S = max(8, N // self.ratio)

        # FPS sampling
        idx_fps = fps(xyz, S)                               # (B,S)
        s_xyz = gather_nd(xyz, idx_fps)                     # (B,S,3)
        s_feat = gather_nd(feats, idx_fps)                  # (B,S,C_in)

        # kNN grouping relative to sampled points
        idx_knn = knn(xyz, s_xyz, self.k)                   # (B,S,k)
        n_xyz = gather_nd(xyz, idx_knn)                     # (B,S,k,3)
        n_feat = gather_nd(feats, idx_knn)                  # (B,S,k,C_in)

        # Local op
        out = self.block(s_xyz, s_feat, n_xyz, n_feat)      # (B,S,C_out)

        # post MLP — expects (B,C,N), convert
        out_t = out.permute(0, 2, 1).contiguous()           # (B,C_out,S)
        out_t = self.post(out_t)
        out = out_t.permute(0, 2, 1).contiguous()           # (B,S,C_out)

        return s_xyz, out


# ─── Full Encoder ─────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    4-stage hierarchical encoder.
    Stages 0-1: MLP (fast), Stages 2-3: LocalAttention (context).
    Returns (xyz_list, feat_list) both shallow→deep.
    """
    def __init__(self, cfg):
        super().__init__()
        in_c   = cfg.in_channels        # 6
        dims   = cfg.encoder_dims       # [64, 128, 256, 512]
        ratios = [4, 4, 4, 4]           # downsample ratios
        ks     = cfg.k_neighbors        # [16, 16, 24, 24]


        self.input_proj = nn.Sequential(
            nn.Linear(in_c, dims[0], bias=False),
            nn.LayerNorm(dims[0]),
            nn.ReLU(inplace=True),
        )

        use_attn_from = {'hybrid': 2, 'mlp': 999, 'attn': 0}
        threshold = use_attn_from.get(cfg.encoder_type, 2)
        self.stages = nn.ModuleList()
        in_dim = dims[0]
        for i, (out_dim, ratio, k) in enumerate(zip(dims, ratios, ks)):
            use_attn = (i >= threshold)
            # use_attn = (i >= 2)
            self.stages.append(EncoderStage(in_dim, out_dim, k, ratio, use_attn))
            in_dim = out_dim

        self.out_dims = [dims[0]] + list(dims)   # [32(input), 64, 128, 256, 512] after input proj

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        x: (B, F, N)  [channel-first, as fed by training loop]
        Returns:
          xyz_list  : list len=5, each (B, N_i, 3)  — includes original xyz at idx 0
          feat_list : list len=5, each (B, N_i, C_i)
        """
        # x is (B, F, N) → split xyz and features
        xyz   = x[:, :3, :].permute(0, 2, 1).contiguous()   # (B,N,3)
        feats = x.permute(0, 2, 1).contiguous()              # (B,N,F)

        # Project all F features (including xyz) to first embedding dim
        feats = self.input_proj(feats)                       # (B,N,dims[0])

        xyz_list  = [xyz]
        feat_list = [feats]

        cur_xyz  = xyz
        cur_feat = feats

        for stage in self.stages:
            cur_xyz, cur_feat = stage(cur_xyz, cur_feat)
            xyz_list.append(cur_xyz)
            feat_list.append(cur_feat)

        return xyz_list, feat_list