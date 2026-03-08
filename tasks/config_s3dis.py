# config.py

from dataclasses import dataclass, field
from html import parser
from typing import List, Optional
import argparse


@dataclass
class Config:
    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir: str = "data/dataset/s3dis/processed/Stanford3dDataset_v1.2_Aligned_Version"
    test_area: int = 5
    num_points: int = 16384
    crop_radius: float = 2.5             # was 2.0
    samples_per_room: int = 75          # increase from 50
    num_workers: int = 8
    cache_mode: str = "mmap"          # 'mmap' | 'ram'
    test_stride: float = 0.75    # was 1.0 — more stable val coverage, less variance in val metrics, at the cost of more val batches (1.0 stride → ~1 min val → 0.75 stride → ~1.5 min val)
    batch_size_test: int = 16         # defailt = 1   - # was 1 — GPU processes 16 crops at once
    '''
    Stride 0.5→1.5: 9× fewer crops → val goes from 10,379 to ~1,150 batches
    batch_size 1→16: 16× GPU efficiency
    Combined: ~14 min → ~1 min validation (12-14× speedup)
    '''

    # ── Model ────────────────────────────────────────────────────────────────
    num_classes: int = 13
    in_channels: int = 6              # rgb + norm_xyz
    embed_dim: int = 64
    encoder_type: str = "attn"   # 'mlp' | 'attn' | 'hybrid'
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])
    encoder_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    # k_neighbors: List[int] = field(default_factory=lambda: [16, 24, 32, 48])
    # Pad k to powers of 2 (16, 32, 32, 64 instead of 16, 24, 32, 48). This lets Inductor apply online softmax. 
    k_neighbors: list = field(default_factory=lambda: [16, 16, 32, 64])
    decoder_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 64])
    dropout: float = 0.5          # was 0.3

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_type: str = "focal+lovasz"          # 'ce' | 'weighted_ce' (most reliable first step) | default: 'focal' | 'focal+lovasz'
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1             # was 0.05 — increased smoothing to help stabilize training with focal+lovasz, which has more volatile loss landscape than plain CE. This is a bit of a band-aid and ideally we'd add label smoothing to the focal loss implementation itself, but this is a quick way to get some smoothing effect without code changes.
    class_weights_mode: str = 'old_inv_sqrt' # 'inv_sqrt'
    '''
    8.1  label_smoothing with focal loss has no effect
    Config.py sets label_smoothing=0.05. FocalLoss uses nn.CrossEntropyLoss internally but does NOT pass label_smoothing to it. Only WeightedCELoss passes label_smoothing. This means 0.05 label smoothing is silently inactive in the focal and focal+lovasz runs — not a bug that harms accuracy, but a misleading config:
    # tasks/main.py — FocalLoss — label_smoothing is NOT used here:
    self.ce = nn.CrossEntropyLoss(weight=alpha, ignore_index=ignore_index, reduction='none')
    > Either add label_smoothing to FocalLoss or document in config that label_smoothing only applies to loss_type='weighted_ce'.
    '''

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer: str = "adamw"          # 'adamw' | 'sgd'
    lr: float = 9e-4                    # 6e-4
    weight_decay: float = 2e-3          # 1e-3      # 5e-4        # was 1e-4 
    grad_clip: float = 1.0
    use_gmp: bool = False

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler: str = "cosine"         # 'cosine' | 'onecycle'
    epochs: int = 150                # with early stopping, actual epochs may be less
    warmup_epochs: int = 20
    min_lr: float = 1e-6

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 48
    accum_steps: int = 1
    use_amp: bool = True
    validate_every: int = 5
    seed: int = 42

    use_ema: bool = True             # use EMA model for validation
    ema_decay: float = 0.999         # EMA decay rate (0.999 for 100ep, 0.9999 for 200ep+)

    early_stop_patience: int = 20   # stop if no improvement for 20 val checks

    rare_room_weight: float = 8.0   # was 8.0 — increase to 20.0 to give even more emphasis to rare rooms (conference room, hallway, lobby, stairs). This is a simple way to address the class imbalance in the dataset without modifying the loss function itself. By assigning a higher sampling weight to samples from rare rooms, we can ensure that the model sees more examples of these underrepresented classes during training, which can help improve its performance on them.

    # ── Logging ──────────────────────────────────────────────────────────────
    log_dir: str = "checkpoints"
    log_interval: int = 50
    exp_name: Optional[str] = None

    use_wandb: bool = True
    wandb_project: str = "SLNet-T"
    wandb_entity: str = "m-saeid"


def parse_config() -> Config:
    parser = argparse.ArgumentParser()
    cfg = Config()
    # Expose every field as a CLI arg with its default
    for f_name, f_val in vars(cfg).items():
        
        if isinstance(f_val, bool):
            # parser.add_argument(f'--{f_name}', default=f_val, action='store_true' if not f_val else 'store_false')
            parser.add_argument(f'--{f_name}', default=f_val, type=lambda x: x.lower() != 'false')
        elif isinstance(f_val, list):
            parser.add_argument(f'--{f_name}', default=f_val, type=type(f_val[0]) if f_val else str, nargs='+')
        elif f_val is None:
            parser.add_argument(f'--{f_name}', default=None, type=str)
        else:
            parser.add_argument(f'--{f_name}', default=f_val, type=type(f_val))
    args = parser.parse_args()
    return Config(**vars(args))