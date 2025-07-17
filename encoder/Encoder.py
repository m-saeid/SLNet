import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from encoder.imports import *
except:
    from imports import *

class Encoder(nn.Module):
    def __init__(self,
                 n=1024,
                 embed=[3,16,'no',True],
                 res_dim_ratio=1.0,
                 bias=True,
                 use_xyz=True,
                 norm_mode="anchor",
                 std_mode="BN1D",
                 dim_ratio=[2, 2, 2, 1],

                 num_blocks1=[1, 1, 2, 1],
                 transfer_mode = ['mlp', 'mlp', 'mlp', 'mlp'],
                 block1_mode = ['mlp', 'mlp', 'mlp', 'mlp'],

                 num_blocks2=[1, 1, 2, 1],
                 block2_mode = ['mlp', 'mlp', 'mlp', 'mlp'],

                 k_neighbors=[32, 32, 32, 32],
                 sampling_mode=['fps', 'fps', 'fps', 'fps'],
                 sampling_ratio=[2, 2, 2, 2],

                 fps_method='pointops2',     # ['pointops2', 'pytorch3d', 'pytorch']
                 knn_method='pytorch3d',     # ['pytorch3d', 'pytorch']
                 **kwargs):
        super(Encoder, self).__init__()

        # print(num_blocks1, k_neighbors, sampling_ratio, num_blocks2, dim_ratio)

        assert len(num_blocks1) == len(k_neighbors) == len(sampling_ratio) == len(num_blocks2) == len(dim_ratio), \
            "The number of stages must be equal. [num_blocks1, num_blocks2 k_neighbors, sampling_ratio]"
        
        self.embedding = Embedding(embed[0], embed[1], mode=embed[2], alpha_beta=embed[3])

        self.sampling_list = nn.ModuleList()
        self.grouping_list = nn.ModuleList()
        self.normalization_list = nn.ModuleList()
        self.blocks1_list = nn.ModuleList()
        self.agg_list = nn.ModuleList()
        self.blocks2_list = nn.ModuleList()

        s = n
        last_ch = embed[1]
        self.stages = len(num_blocks1)

        for i in range(self.stages):
            out_ch = last_ch * dim_ratio[i]
            s = s // sampling_ratio[i]

            self.sampling_list.append(Sampling(sampling_mode[i], s, fps_method))

            self.grouping_list.append(Grouping(k_neighbors[i], use_xyz, knn_method))

            self.normalization_list.append(Normalization(d=last_ch, norm_mode=norm_mode, std_mode=std_mode, use_xyz=use_xyz))

            self.blocks1_list.append(Block1(last_ch, out_ch, transfer_mode[i], block1_mode[i], num_blocks1[i], res_dim_ratio=res_dim_ratio,
                                             bias=bias, use_xyz=use_xyz))
            
            self.agg_list.append(Agg("adaptive_max_pool1d"))

            self.blocks2_list.append(Block2(out_ch, block2_mode[i], num_blocks2[i], res_dim_ratio=res_dim_ratio, bias=bias))

            last_ch = out_ch


    def forward(self, xyz, x):            # xyz: [B, C, N] (2,3,1024)       x: [B, D, 1024] (2,32,1024)
        
        f = self.embedding(x)             # f: [B, D, N] (2,32,1024)

        xyz = xyz.permute(0, 2, 1)        # [B, N, C] (2,1024,3)

        xyz_list = [xyz]
        f_list = [f]

        for i in range(self.stages):  # 4 stages
            f = f.permute(0,2,1)      # [B, N/S, D]     (2, 1024>512, 16>32)

            # D : 16>32>64>128

            xyz_sampled, f_sampled = self.sampling_list[i](xyz, f)                                  # [B, N, C/D] > [B, S, C/D]  (2, 1024>512>256>128>64, 3)  (2, 1024>512>256>128>64, 16>32>64>128)
            xyz_grouped, f_grouped = self.grouping_list[i](xyz, f, xyz_sampled, f_sampled)          # [B, S, K, C]  [B, S, K, D]  (2, 512>256>128>64, 24,3)  (2, 512>256>128>64, 24, 16>32>64>128)
            f_grouped = self.normalization_list[i](xyz_sampled, f_sampled, xyz_grouped, f_grouped)  # [B, S, K, 2D+C]  (2, 512>256>128>64, 24, 35>67>131>259)

            b, s, _ = xyz_sampled.shape                                                             # B, S, C  2, 512>256>128>64, 3

            f = self.blocks1_list[i](f_grouped)    # [B*S, 2D|D, K]  (1024>512>256>128, 32>64>128>128, 24)
            f = self.agg_list[i](f, b, s)          # [B, 2D, S]    (2, 32>64>128>128, 512>256>128>64)
            f = self.blocks2_list[i](f)            # [B, 2D, S]    (2, 32>64>128>128, 512>64>128,64)

            xyz = xyz_sampled                      # [B, S, C]     (2, 512>256>128>64, 3)

            xyz_list.append(xyz)
            f_list.append(f)

        return xyz_list, f_list
    

encoder = Encoder(n=1024, embed=[3,16,'no',False], res_dim_ratio=0.25, bias=False, use_xyz=True, norm_mode="anchor", std_mode="BN1D",
                            dim_ratio=[2, 2, 2, 1], num_blocks1=[1, 1, 2, 1], num_blocks2=[1, 1, 2, 1],
                            k_neighbors=[24,24,24,24], sampling_mode=['fps', 'fps', 'fps', 'fps'],
                            sampling_ratio=[2, 2, 2, 2],
                            fps_method='pointops2', knn_method='pytorch3d')


if __name__ == '__main__':
    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    xyz = torch.rand(2, 3, 1024)
    x = torch.rand(2, 16, 1024)
    print("===> testing Model ...")
    #model_encoder = encoder()
    print(f'number of params: {trainable_params(encoder)}')
    xyz_list, f_list = encoder(xyz, x)
    print('xyz', [print(xyz.shape, end=', ') for xyz in xyz_list])
    print('f', [print(f.shape, end=', ') for f in f_list])
