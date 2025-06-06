import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import *
except:
    from encoder_util import *


class Block1(nn.Module):    # reduce d  and  residual
    def __init__(self, ch, out_ch, transfer_mode='mlp', block1_mode='mlp',  blocks=1, res_dim_ratio=1, bias=True, use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        """

        super().__init__()
        in_ch = 3+2*ch if use_xyz else 2*ch

        if transfer_mode == 'mlp':
            self.transfer = mlp(in_ch, out_ch, bias=bias)
        else:
               raise Exception(f"transfer_mode!!! {transfer_mode}")
               
        operation = []
        for _ in range(blocks):
            operation.append(
                Residual(ch=out_ch, res_mode=block1_mode, res_dim_ratio=res_dim_ratio, bias=bias)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, k, d = x.size()       # [b,s,k,d] 2,512,24,35
        x = x.permute(0, 1, 3, 2)   # [b,s,d,k] 2,512,35,24
        x = x.reshape(-1, d, k)     # [b*s,d,k] 1024,35,24
        x = self.transfer(x)        # [b*s,d,k] 1024,32,24   transfer: Conv1d(35>32)+BN+ReLU
        #batch_size, _, _ = x.size() # [b*s,d,k] 1024,32,24
        x = self.operation(x)       # [b*s,d,k] 1024,32,24   RESIDUAL BLOCKS
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
        # x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
        return x          # 1024,32,24          #[b,d,s] 2,32,512