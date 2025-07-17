import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.encoder_util import mlp, square_distance, index_points
from encoder.m6_block2 import Block2


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, de_fp_fuse='mlp', de_fp_block='mlp', blocks=1, res_expansion=1.0, bias=True):
        super(FeaturePropagation, self).__init__()

        self.fuse = mlp(in_channel, out_channel, bias=bias)

        input_dim, output_dim = in_channel, out_channel

        if de_fp_fuse == 'mlp':
            self.embd = mlp(input_dim, output_dim, bias=bias)
        else:
            raise Exception(f"de_fp_fuse!!! {de_fp_fuse}")   

        self.extraction = Block2(out_channel, de_fp_block, blocks, res_dim_ratio=res_expansion, bias=bias)

    def forward(self, xyz1, xyz2, points1, points2):
        # (1)      2,256,3  2,128,3  2,128,256  2,128,128
        # (2)      ...
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1) # 2,128,128
        B, N, C = xyz1.shape               # 2,256,3
        _, S, _ = xyz2.shape               # _,128,_      2 8 3

        if S == 1:                         # False
            interpolated_points = points2.repeat(1, N, 1)  #  2 8 128 > 
        else:
            dists = square_distance(xyz1, xyz2)            # 2,256,128
            dists, idx = dists.sort(dim=-1)                # 2,256,128  2,256,128
            dists, idx = dists[:, :, :3], idx[:, :, :3]    # k=3  2,256,3  2,256,3

            dist_recip = 1.0 / (dists + 1e-8)                   # 2,256,3
            norm = torch.sum(dist_recip, dim=2, keepdim=True)   # 2,256,1
            weight = dist_recip / norm                          # 2,256,3
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) # 2,256,128
            # points2:2,128,128    idx:2,256,3    weight:2,256,3    weight.view(B, N, 3, 1):2,256,3,1
        if points1 is not None:                     # True
            points1 = points1.permute(0, 2, 1)      # 2,256,128
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # 2,256,256
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)    # 2,256,256

        new_points = self.fuse(new_points)          # MLP(256>512): 2,512,256
        # MLP1(256>512; [2,256,32]>[2,512,32])     MLP2(576>256; [2,576,128]>[2,256,128])
        # MLP3(288>128; [2,288,512]>[2,128,512])   MLP4(144>128; [2,144,2048]>[2,128,2048])
        new_points = self.extraction(new_points)    # Residual: 2,512,256
        # 2 512 32 > 2 512 32  -  2 256 128 > 2 256 128  -  2 128 512 > 2 128 512  -  2 128 2048 > 2 128 2048
        return new_points                           # 2,512,256
        # 2 512 32