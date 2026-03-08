import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]
sys.path.append(BASE_DIR)

import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from config import Config, parse_config
from data.dataloader_setup import build_train_loader, build_test_loader
from model_slnet_t.model import SegModel
from tasks.val_epoch import val_epoch

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import wandb
import platform
import psutil

from rich.logging import RichHandler
from rich.console import Console
from rich import inspect

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Optional, Dict



def profile_model(model, cfg, device, logger):
    """Compute GFLOPs, inference time, peak GPU memory for WandB."""
    import time
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, cfg.in_channels, cfg.num_points, device=device)
        flops = FlopCountAnalysis(model, dummy)
        gflops = flops.total() / 1e9
    except Exception:
        gflops = None
        logger.warning("fvcore not installed — GFLOPs not computed. pip install fvcore")

    # Inference time (median over 50 runs after 10 warmup)
    dummy = torch.randn(1, cfg.in_channels, cfg.num_points, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(dummy)
        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    inference_ms = float(np.median(times))

    # Peak GPU memory
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    param_count_val = sum(p.numel() for p in model.parameters()) / 1e6

    model.train()
    
    gflops_str = f"{gflops:.2f}" if gflops is not None else "N/A"

    logger.info(
        f"[Profile] GFLOPs={gflops_str}  "
        f"Params={param_count_val:.2f}M  "
        f"InferenceTime={inference_ms:.1f}ms  "
        f"PeakMemory={peak_mb:.0f}MB"
    )

    return {
        'gflops': gflops,
        'params_M': param_count_val,
        'inference_ms': inference_ms,
        'peak_memory_mb': peak_mb,
    }



CLASS_NAMES = ['ceiling','floor','wall','beam','column','window',
               'door','chair','table','bookcase','sofa','board','clutter']


# ─────────────────────────── Helpers ─────────────────────────────────────────

def get_system_info() -> dict:
    """Collect system hardware/software specs for WandB."""
    info = {
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'os': platform.platform(),
        'cpu': platform.processor(),
        'cpu_cores': psutil.cpu_count(logical=False),
        'ram_gb': round(psutil.virtual_memory().total / 1e9, 1),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info['gpu_count'] = torch.cuda.device_count()
    return info



def setup_logger(log_dir: str, name: str = None) -> logging.Logger:
    logger_name = name or f's3dis_{os.path.basename(log_dir)}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    file_fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S')
    
    # train.log — clean training metrics
    fh_train = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh_train.setFormatter(file_fmt)
    fh_train.setLevel(logging.INFO)

    # run.log — everything including DEBUG, warnings, errors
    fh_run = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    fh_run.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S'
    ))
    fh_run.setLevel(logging.DEBUG)

    sh = RichHandler(show_path=False, omit_repeated_times=False,
                     markup=True, rich_tracebacks=True)

    logger.addHandler(fh_train)
    logger.addHandler(fh_run)
    logger.addHandler(sh)
    return logger

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_count(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total"


# ─── 3. Balanced sampler ─────────────────────────────────────────────────────


# ─── Efficient room rarity (computed once, not every epoch) ────────────
def precompute_rare_rooms(dataset) -> set:
    """
    Scans all rooms ONCE to find which contain rare classes.
    Called before the training loop — result cached as rare_rooms set.
    """
    RARE = {3, 4, 6, 7, 8, 9, 10, 11}  # beam,column,door,chair,table,bookcase,sofa,board
    rare_rooms = set()
    for room_id in range(len(dataset.room_files)):
        data = dataset.cache.get(room_id)
        lbls = set(data[:, 6].astype(int).tolist())
        if lbls & RARE:
            rare_rooms.add(room_id)
    return rare_rooms


def make_sampler_from_rare_rooms(dataset, rare_room_weight: float, rare_rooms: set) -> WeightedRandomSampler:
    """
    Builds WeightedRandomSampler from pre-computed rare_rooms set.
    Called every epoch after on_epoch_end() — NO file I/O.
    """
    weights = [rare_room_weight if rid in rare_rooms else 1.0
               for rid, _ in dataset.sample_list]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float),
        num_samples=len(weights),
        replacement=True,
    )


# ─────────────────────────── Losses ──────────────────────────────────────────

# Lovász-Softmax loss for multi-class segmentation (optional auxiliary loss)

# ─── Weighted CE ───────────────────────────────────────────────────────────

def compute_class_weights(data_root: str, test_area: int,
                           num_classes: int = 13,
                           mode: str = 'inv_sqrt',
                           n_sample_rooms: int = None) -> np.ndarray:
    """
    Compute class weights from training rooms.

    Args:
        mode: 'inv_sqrt' = 1/sqrt(freq)   (good balance, industry standard)
              'median_freq' = median/freq  (SegNet-style, more aggressive)
              'inv_log'     = 1/log(1+freq) (softer than inv_sqrt)
    Returns:
        weights: (num_classes,) float32, sum-normalized
    """
    data_root = Path(data_root)
    test_area_str = f'Area_{test_area}'
    train_files = [f for f in sorted(data_root.glob('*.npy'))
                   if test_area_str not in f.stem]
    if n_sample_rooms:
        train_files = train_files[:n_sample_rooms]

    counts = np.zeros(num_classes, dtype=np.int64)
    for f in train_files:
        data = np.load(f, mmap_mode='r')
        lbls = data[:, 6].astype(np.int64)
        for c in range(num_classes):
            counts[c] += int((lbls == c).sum())

    counts = counts.clip(min=1)
    freq = counts / counts.sum()

    if mode == 'inv_sqrt':
        weights = 1.0 / np.sqrt(freq)
    elif mode == 'old_inv_sqrt':
        weights = 1.0 / np.sqrt(counts.astype(np.float64))
    elif mode == 'median_freq':
        med = np.median(freq)
        weights = med / freq
    elif mode == 'inv_log':
        weights = 1.0 / np.log(1.0 + freq)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    weights /= weights.sum()
    print(f"[class_weights] mode={mode}")
    CLASS_NAMES = ['ceiling','floor','wall','beam','column','window',
                   'door','chair','table','bookcase','sofa','board','clutter']
    for i, (n, w, c) in enumerate(zip(CLASS_NAMES, weights, counts)):
        print(f"  {i:2d} {n:<12}  count={c:>10}  weight={w:.4f}")
    return weights.astype(np.float32)


class WeightedCELoss(nn.Module):
    """
    Weighted cross-entropy with optional label smoothing.
    class_weights: (C,) tensor on the correct device.
    """
    def __init__(self, class_weights: torch.Tensor,
                 ignore_index: int = -100,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B*N, C) or (B, C, N) etc.
        return self.ce(logits, targets)


# ─── 2. Lovász-Softmax Auxiliary Loss ────────────────────────────────────────
# Reference: Berman et al., "The Lovász-Softmax Loss" (CVPR 2018)
# This directly optimizes the mIoU metric rather than cross-entropy.
# Add as auxiliary: total_loss = focal_loss + 0.5 * lovász_loss

def _lovász_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient of the Lovász extension with respect to sorted errors.
    Equation 9 in https://arxiv.org/abs/1705.08790.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """
    Multi-class Lovász-Softmax loss.
    Operates on the flattened (N, C) logit tensor.

    per_class='present' — average only over classes present in the batch
    per_class='all'     — average over all 13 classes (stable gradients)
    """
    def __init__(self, per_class: str = 'present', ignore_index: int = -100):
        super().__init__()
        self.per_class = per_class
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, C) after reshaping
        targets: (N,)
        """
        # Filter ignore
        mask = targets != self.ignore_index
        logits  = logits[mask]
        targets = targets[mask]

        if logits.numel() == 0:
            return logits.sum() * 0.0

        C = logits.size(1)
        probs = F.softmax(logits, dim=1)

        losses = []
        for c in range(C):
            if self.per_class == 'present' and (targets == c).sum() == 0:
                continue
            fg      = (targets == c).float()            # (N,) binary
            class_p = probs[:, c]                       # (N,)
            errors  = (fg - class_p).abs()              # (N,)
            errors_sorted, perm = errors.sort(descending=True)
            fg_sorted = fg[perm]
            grad      = _lovász_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


# cfg.loss_type = 'focal+lovasz'  
class FocalPlusLovaszLoss(nn.Module):
    """
    Primary: Focal  (handles class imbalance)
    Auxiliary: Lovász (directly optimizes mIoU)
    Total: focal + lovász_weight * lovász
    """
    def __init__(self, focal_gamma: float = 2.0,
                 class_weights: torch.Tensor = None,
                 lovász_weight: float = 0.5,
                 ignore_index: int = -100,
                 label_smoothing: float = 0.0):
        super().__init__()
        # from tasks.main import FocalLoss

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                class_weights = class_weights.float()
            
        self.focal  = FocalLoss(gamma=focal_gamma, alpha=class_weights,
                                ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.lovász = LovaszSoftmaxLoss(per_class='all', # 'present',
                                         ignore_index=ignore_index)
        self.lw = lovász_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        f_loss = self.focal(logits, targets)
        l_loss = self.lovász(logits, targets)
        return f_loss + self.lw * l_loss



# ───────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None,
                 ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(
            weight=alpha,
            ignore_index=ignore_index,
            reduction='none',
            label_smoothing=label_smoothing,   # ← ADD THIS
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = self.ce(logits, targets)
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        mask = targets != self.ce.ignore_index
        return loss[mask].mean() if mask.any() else loss.mean()




def _compute_class_weights_inv_sqrt_freq(data_root: str, test_area: int,
                                          num_classes: int = 13) -> np.ndarray:
    """Compute inverse-sqrt-frequency class weights from training rooms."""
    data_root = Path(data_root)
    test_area_str = f'Area_{test_area}'
    train_files   = [f for f in sorted(data_root.glob('*.npy'))
                     if test_area_str not in f.stem]
    counts = np.zeros(num_classes, dtype=np.int64)
    for f in train_files:   # ← REMOVE [:20] (only for quick testing)
        data = np.load(f, mmap_mode='r')
        lbls = data[:, 6].astype(np.int64)
        for c in range(num_classes):
            counts[c] += (lbls == c).sum()
    counts = counts.clip(min=1)
    weights = 1.0 / np.sqrt(counts.astype(np.float64))
    weights /= weights.sum()
    return weights.astype(np.float32)


def build_criterion(cfg: Config, class_weights=None, device=None) -> nn.Module:
    w = None
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if cfg.loss_type == 'focal':
        return FocalLoss(gamma=cfg.focal_gamma, alpha=w, ignore_index=-100, label_smoothing=cfg.label_smoothing)
    elif cfg.loss_type == 'weighted_ce':
        if w is None:
            cw = compute_class_weights(
                cfg.data_dir, cfg.test_area,
                mode=cfg.class_weights_mode,   # ← now uses inv_sqrt / median_freq / inv_log
                num_classes=cfg.num_classes,
            )

            w = torch.tensor(cw, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w, ignore_index=-100,
                                label_smoothing=cfg.label_smoothing)
    
    elif cfg.loss_type == 'weighted_ce+lovasz':
        if w is None:
            cw = _compute_class_weights_inv_sqrt_freq(cfg.data_dir, cfg.test_area)
            w = torch.tensor(cw, dtype=torch.float32, device=device)
        wce = nn.CrossEntropyLoss(weight=w, ignore_index=-100,
                                label_smoothing=cfg.label_smoothing)
        lovasz = LovaszSoftmaxLoss(per_class='all', ignore_index=-100)
        class _WCELovasz(nn.Module):
            def forward(self, logits, targets):
                return wce(logits, targets) + 0.5 * lovasz(logits, targets)
        return _WCELovasz()

    elif cfg.loss_type == 'focal+lovasz':
        if w is None:
            if cfg.class_weights_mode == 'inv_sqrt':
                cw = compute_class_weights(cfg.data_dir, cfg.test_area, mode='inv_sqrt')
            else:
                cw = compute_class_weights(
                    cfg.data_dir,
                    cfg.test_area,
                    num_classes=cfg.num_classes
                )

            w = torch.tensor(cw, dtype=torch.float32, device=device)

        return FocalPlusLovaszLoss(
            focal_gamma=cfg.focal_gamma,
            class_weights=w,
            lovász_weight=0.5,
            ignore_index=-100,
            label_smoothing=cfg.label_smoothing,
        )
    else:
        return nn.CrossEntropyLoss(ignore_index=-100,
                                   label_smoothing=cfg.label_smoothing)


def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=0.9, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)


def build_scheduler(cfg: Config, optimizer, steps_per_epoch: int):
    if cfg.scheduler == 'onecycle':
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr * 5,
            total_steps=cfg.epochs * steps_per_epoch,
            pct_start=cfg.warmup_epochs / cfg.epochs,
            anneal_strategy='cos',
        )
        sched.step_per_batch = True
        return sched, True
    else:
        def lr_lambda(epoch):
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / cfg.warmup_epochs
            progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return cfg.min_lr / cfg.lr + 0.5 * (1 - cfg.min_lr / cfg.lr) * (
                1 + np.cos(np.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        sched.step_per_batch = False
        return sched, False

