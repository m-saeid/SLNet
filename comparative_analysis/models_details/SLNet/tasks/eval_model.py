import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]
sys.path.append(BASE_DIR)

import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# import models as models
from data.modelnet import ModelNet40
from utils.helper import cal_loss
import numpy as np
import torch.nn.functional as F

from encoder.Encoder import Encoder
from utils.util import Classifier
from time import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1

import torch
from fvcore.nn import FlopCountAnalysis


class Classification(nn.Module):
    def __init__(self,
                 n=1024,
                 embed=[3, 16, 'no', 'yes'],
                 res_dim_ratio=0.25,
                 bias=False,
                 use_xyz=True,
                 norm_mode="anchor",
                 std_mode="BN1D",
                 dim_ratio=[2, 2, 2, 1],
                 num_blocks1=[1, 1, 2, 1],
                 transfer_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 block1_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 num_blocks2=[1, 1, 2, 1],
                 block2_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 k_neighbors=[32, 32, 32, 32],
                 sampling_mode=['fps', 'fps', 'fps', 'fps'],
                 sampling_ratio=[2, 2, 2, 2],
                 classifier_mode='mlp_very_large',
                 fps_method='pointops2',
                 knn_method='pytorch3d',
                 ):
        super(Classification, self).__init__()
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
            transfer_mode=transfer_mode,
            block1_mode=block1_mode,
            num_blocks2=num_blocks2,
            block2_mode=block2_mode,
            k_neighbors=k_neighbors,
            sampling_mode=sampling_mode,
            sampling_ratio=sampling_ratio,
            fps_method=fps_method,
            knn_method=knn_method, 
        )
        
        last_dim = embed[1]
        for d in dim_ratio:
            last_dim *= d
        
        self.classifier = Classifier(last_dim, classifier_mode, 40)
    
    def forward(self, xyz, feature):
        _, f_list = self.encoder(xyz, feature)
        x = F.adaptive_max_pool1d(f_list[-1], 1).squeeze(dim=-1)
        return self.classifier(x)


def compute_flops_fvcore(model, inputs):
    """
    Computes GFLOPs of a model using fvcore, supporting one or two inputs.

    Args:
        model: PyTorch model.
        inputs: torch.Tensor or tuple of torch.Tensor.

    Returns:
        total_flops_g: float, total GFLOPs.
    """
    model.eval()
    # FlopCountAnalysis accepts a tensor or tuple of tensors as inputs :contentReference[oaicite:5]{index=5}.
    flops = FlopCountAnalysis(model, inputs)
    total_flops = flops.total()  # in FLOPs
    total_flops_g = total_flops / 1e9
    print(f"Total FLOPs: {total_flops_g:.3f} GFLOPs")
    return total_flops_g


def measure_memory_cuda(model: torch.nn.Module, inputs):
    """
    Measures peak GPU memory usage during inference.

    Args:
        model: PyTorch model.
        inputs: torch.Tensor or tuple of torch.Tensor.

    Returns:
        peak_mem_mb: float, peak memory in MB.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    # Move inputs to device
    if isinstance(inputs, tuple):
        inp = [inp.to(device) for inp in inputs]
        args = tuple(inp)
    else:
        args = (inputs.to(device),)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(*args) if len(args) > 1 else model(args[0])
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB :contentReference[oaicite:7]{index=7}
    print(f"Peak GPU Memory: {peak:.2f} MB")
    return peak


def trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params: {trainable_params}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--embd_dim', type=int, default=16, help='embd_dim')
    parser.add_argument('--model', default='', help='model name [default: SLNet]')
    parser.add_argument('--num_classes', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    device = 'cuda'

    embd_dim = args.embd_dim

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, in_d=3, out_d=embd_dim), num_workers=4,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)
    print('==> Building model..')

    net = Classification(embed=[3, embd_dim, 'no', 'yes'],)

    criterion = cal_loss
    net = net.to(device)
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    validate(net, test_loader, criterion, device, args.batch_size)

    # Single-input example
    #data = torch.randn(1, 3, 1024).cuda()
    # Two-input example
    data2 = torch.randn(1, 3, 1024).cuda().contiguous()
    feat2 = torch.randn(1, embd_dim, 1024).cuda().contiguous()

    # GFLOPs
    # compute_flops_fvcore(net, data)             # Single input
    compute_flops_fvcore(net, (data2, feat2))     # Two inputs

    # Memory
    # measure_memory_cuda(net, data)               # GPU mem, single
    measure_memory_cuda(net, (data2, feat2))     # GPU mem, two

    trainable_params(net)


def validate(net, testloader, criterion, device, bs):
    net.eval()
    with torch.no_grad():
        t0 = time()
        for batch_idx, (data, feature, _) in enumerate(testloader):
            #data, feature, label = data.to(device), feature.to(device), label.to(device).squeeze()
            data, feature = data.to(device), feature.to(device)
            data = data.permute(0, 2, 1)
            #feature = feature.permute(0, 2, 1)
            net(data, feature)
            
        t1 = time()
        t = (t1 - t0) * 1e3
        print(f"time {t} - {t/ 2480} per sample")


if __name__ == '__main__':
    main()
