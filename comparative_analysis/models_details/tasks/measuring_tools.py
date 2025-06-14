import torch
import argparse
from fvcore.nn import FlopCountAnalysis
from torch.autograd import Variable
import numpy as np

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('eval')
    parser.add_argument('--model', type=str,
                        help='model: [pointnet, pointnet2_ssg, pointnet2_msg, dgcnn, curvenet, pointmlp_elite, pointmlp, apes_global, apes_local, slnet_embed_dim_16, slnet_embed_dim_32]')
    parser.add_argument('--num_points', type=int, default=2048, help='Point Number')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--workers', default=6, type=int, help='workers')
    return parser.parse_args()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y

def params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")
    return trainable


def gflops(model: torch.nn.Module, inputs) -> float:
    """
    Compute GFLOPs using fvcore.
    """
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        total = flops.total()
    gflops = total / 1e9
    print(f"Total FLOPs: {gflops:.3f} GFLOPs")
    return gflops


def memory(model: torch.nn.Module, inputs) -> float:
    """
    Measure peak GPU memory during a single forward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if isinstance(inputs, tuple):
        args = tuple(inp.to(device) for inp in inputs)
    else:
        args = (inputs.to(device),)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(*args) if len(args) > 1 else model(args[0])

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak GPU memory: {peak_mb:.2f} MB")
    return peak_mb



def inference_time(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    bs,
    model_name,
    num_warmup_batches: int = 5,
    max_batches: int = None,
    ) -> float:
    """
    Measures average inference time per batch over a DataLoader.
    Benchmark with different batch sizes: Data loading time grows with batch size, so test what works best for your hardware
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Warm-up phase to stabilize performance
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_warmup_batches:
                break
            if model_name[:5] == 'slnet':
                data = (data[0].to(device).permute(0,2,1), data[1].to(device))
            else:
                data = (data.to(device).permute(0,2,1), )
            _ = model(*data)

    timings = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            if model_name[:5] == 'slnet':
                data = (data[0].to(device).permute(0,2,1), data[1].to(device))
            else:
                data = (data.to(device).permute(0,2,1), )

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(*data)
            end.record()
            torch.cuda.synchronize()
            delta = start.elapsed_time(end)

            timings.append(delta)

    avg_ms = sum(timings) / (len(timings)*bs)

    print(f"Avg inference time over {len(timings)} runs, each with {bs} batches: {avg_ms:.3f} ms per sample")
    return avg_ms



def batch_prep(data, model_name):

    device = 'cuda'

    if model_name[:5] == 'slnet':
        points, feature, label, target, norm_plt = data
        points, feature, label, target, norm_plt = Variable(points.float()), Variable(feature.float()), Variable(label.long()), \
                                            Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        feature = feature.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, feature, label, target, norm_plt = points.cuda(non_blocking=True), feature.cuda(non_blocking=True), \
                                                label.squeeze(1).cuda(non_blocking=True), \
                                                target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        return points, feature, norm_plt, to_categorical(label, 16)


    elif model_name == 'pointmlp':
        points, label, target, norm_plt = data
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                        Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                        target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        return points, norm_plt, to_categorical(label, 16)


    elif model_name in ['dgcnn', 'curvenet']:
        data, label, seg, _ = data
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        return data, label_one_hot


    elif model_name in ['pointnet', 'pointnet2_ssg', 'pointnet2_msg']:
        points, label, target, normals = data
        # points = torch.cat([points, normals], 2)    # with normals
        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
        points = points.transpose(2, 1)
        return points, to_categorical(label, 16)

    elif model_name in ['apes_local', 'apes_global']:
        data, label, seg, _ = data
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32)).unsqueeze(-1)
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        return data, label_one_hot





def inference_time_shapenet(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    bs,
    model_name,
    num_warmup_batches: int = 5,
    max_batches: int = None,
    ) -> float:
    """
    Measures average inference time per batch over a DataLoader.
    Benchmark with different batch sizes: Data loading time grows with batch size, so test what works best for your hardware
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Warm-up phase to stabilize performance
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_warmup_batches:  # <-- add this
                break
            data = batch_prep(data, model_name)   
            model(*data)

    timings = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            data = batch_prep(data, model_name)    
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(*data)
            end.record()
            torch.cuda.synchronize()
            delta = start.elapsed_time(end)
            timings.append(delta)
            
    avg_ms = sum(timings) / (len(timings)*bs)

    print(f"Avg inference time over {len(timings)} runs, each with {bs} batches: {avg_ms:.3f} ms per sample")
    return avg_ms
