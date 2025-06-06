import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.Encoder import Encoder
from encoder.encoder_util import mlp

from decoder.m0_fpropagation import FeaturePropagation


class Decoder(nn.Module):
    def __init__(self,
                 
                 task="partseg_shapenet",

                 # Encoder:
                 n=1024,
                 embed=[3,16,'no',True],
                 res_dim_ratio=1.0,
                 bias=True,
                 use_xyz=True,
                 norm_mode="center",
                 std_mode="BN1D",
                 dim_ratio=[2, 2, 2, 2],

                 num_blocks1=[2, 2, 2, 2],
                 transfer_mode = ['mlp', 'mlp', 'mlp', 'mlp'],
                 block1_mode = ['mlp', 'mlp', 'gaussian', 'mlp'],

                 num_blocks2=[2, 2, 2, 2],
                 block2_mode = ['mlp', 'mlp', 'mlp', 'mlp'],

                 k_neighbors=[32, 32, 32, 32],
                 sampling_mode=['fps', 'fps', 'fps', 'fps'],
                 sampling_ratio=[2, 2, 2, 2],

                 # Decoder:
                 de_dims=[512, 256, 128, 128],
                 de_blocks=[2, 2, 2, 2],

                 de_fp_fuse=['mlp', 'mlp', 'mlp', 'mlp'],
                 de_fp_block=['mlp', 'mlp', 'mlp', 'mlp'],

                 gmp_dim=64,
                 gmp_dim_mode = 'mlp',

                 cls_dim=64,
                 cls_map_mode = 'mlp',
                 gmp_map_end_mode = 'mlp',

                 num_cls = 50,
                 classifier_mode = 'mlp',
                 fps_method = 'pointops2',
                 knn_method = 'pytorch3d',
                 **kwargs):
        
        super(Decoder, self).__init__()

        self.task = task

        self.encoder = Encoder(
                            n=n,
                            embed=embed,
                            res_dim_ratio=res_dim_ratio,
                            bias=bias,
                            use_xyz=use_xyz,
                            norm_mode=norm_mode,
                            std_mode=std_mode,
                            dim_ratio=dim_ratio,

                            num_blocks1=num_blocks1,
                            transfer_mode = transfer_mode,
                            block1_mode = block1_mode,

                            num_blocks2=num_blocks2,
                            block2_mode=block2_mode,

                            k_neighbors=k_neighbors,
                            sampling_mode=sampling_mode,
                            sampling_ratio=sampling_ratio,
                            fps_method=fps_method,
                            knn_method=knn_method,
                            )


        # en_dims = [16,32,64,128,128]
        en_dims = [embed[1]]
        for i in dim_ratio:
            en_dims.append(en_dims[-1]*i)

        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                FeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1], de_fp_fuse=de_fp_fuse[i], de_fp_block=de_fp_block[i],
                                           blocks=de_blocks[i], res_expansion=res_dim_ratio,
                                           bias=bias)
            )

        # class label mapping
        if task == 'partseg_shapenet':
            if cls_map_mode == "mlp":
                self.cls_map = nn.Sequential(
                    mlp(16, cls_dim, bias=bias),
                    mlp(cls_dim, cls_dim, bias=bias)
                    )
            else:
                raise Exception(f"cls_map_mode!!! {cls_map_mode}")

        elif task == "semseg_s3dis":
            cls_dim = 0

        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            if gmp_dim_mode == "mlp":
                self.gmp_map_list.append(mlp(en_dim, gmp_dim, bias=bias))

            else:
                raise Exception(f"gmp_dim_mode!!! {gmp_dim_mode}")

        if gmp_map_end_mode == "mlp":
            self.gmp_map_end = mlp(gmp_dim*len(en_dims), gmp_dim, bias=bias)
        else:
            raise Exception(f"gmp_map_end_mode!!! {gmp_map_end_mode}")


        # classifier
        if classifier_mode == "mlp":
            self.classifier = nn.Sequential(
                nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
                nn.BatchNorm1d(128), nn.Dropout(),
                nn.Conv1d(128, num_cls, 1, bias=bias)
            )
        else:
            raise Exception(f"classifier_mode!!! {classifier_mode}")


        self.en_dims = en_dims

    def forward(self, xyz, f, norm_plt, cls_label): 
        # x:[B,C,N]  norm_plt:[B,C,N]  cls_label:[B,num_cls]  dataset=shapenet
        x = torch.cat([f, norm_plt],dim=1)          # [B, 6, N]  (2,16/32,2048)

        xyz_list, x_list = self.encoder(xyz, f)
        # xyz_list                                  # [B, S, C]  (2, 2048,1024,512,256,128, 3)
        # x_list                                    # [B, D, S]  (2, 16,32,64,128,128, 2048,1024,512,256,128)

        trans_feat = x_list[-1]

        # Decoder
        xyz_list.reverse()              # len = 5     [B, S, C]  (2, 128,256,512,1024,2048, 3)
        x_list.reverse()                # len = 5     [B, D, S]  (2, 128,128,64,32,16, 128,256,512,1024,2048)
        x = x_list[0]                   #             [B, D, S]  (2 128 8)
        for i in range(len(self.decode_list)): # 4
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)        # > [B, D, S]
            #                       2 256 3   , 2 128 3   , 2 128 256  , 2 512 256      > 2 512 256
            #                       2 512 3   , 2 256 3   , 2 64 512   , 2 256 512      > 2 256 512
            #                       2 1024 3  , 2 512 3   , 2 32 1024  , 2 128 1024     > 2 128 1024
            #                       2 2048 3  , 2 1024 3  , 2 16 2048  , 2 128 2048     > 2 128 2048

        # Global Context
        # x_list    : 2 128 128  -  2 128 256  -  2 64 512  -  2 32 1024  -  2 16 2048
        gmp_list = []
        for i in range(len(x_list)):    # 5
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
            #print(x_list[i].shape, self.gmp_map_list[i], self.gmp_map_list[i](x_list[i]).shape, F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1).shape)
            #print()
            # 1: 2 128 128   MLP(128>64) 2 64 128      pool    2 64 1
            # 2: 2 128 256   MLP(128>64) 2 64 256      pool    2 64 1
            # 3: 2 64 512    MLP(128>64) 2 64 512      pool    2 64 1
            # 4: 2 32 1024   MLP(32>64)  2 64 1024     pool    2 64 1
            # 5: 2 16 2048   MLP(16>64)  2 64 2048     pool    2 64 1
            # gmp_list: 2 64 1, 2 64 1, 2 64 1, 2 64 1, 2 64 1

        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # 2,320,1  MLP(320>64)  2,64,1

        # cls_token
        if cls_label is not None: #shapenet cls_label:[B,num_cls] (32,16)
            cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]  2,16 > 2,16,1  MLP(16>64) MLP(64>64)  2,64,1
            x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1) # 2 256 2048
            # 2 128 2048  -  2 64 2048  -  2 64 2048  >  2 256 2048
        else:
            x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]])], dim=1) # 2 256 2048


        x = self.classifier(x)      # 2 256 2048 MLP(256>128>50) 2 50 2048
        x = F.log_softmax(x, dim=1) # 2 50 2048
        x = x.permute(0, 2, 1)      # 2 2048 50
        return x, trans_feat        # 2 4096 13     2 128 128  s3dis


if __name__ == '__main__':

    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 2048) # 2 3 2048
    norm = torch.rand(2, 3, 2048) # 2 3 2048
    cls_label = torch.rand([2, 16]) # 2 16
    print("===> testing modelD ...")
    decoder = Decoder(task="partseg_shapenet")

    print(f'params: {trainable_params(decoder)}')

    out = decoder(data, norm, cls_label)  # [2,2048,50]
    
    print(out[0].shape, out[1].shape)