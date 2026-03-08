# dataloader_setup.py

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from .s3dis_dataset import S3DISDataset, build_train_augmentations


def build_train_loader(
    data_root: str,
    test_area: int = 5,
    num_points: int = 16384,
    batch_size: int = 8,
    num_workers: int = 8,
    samples_per_room: int = 50,
    distributed: bool = False,
    crop_radius: float = 2.0,
) -> DataLoader:

    dataset = S3DISDataset(
        data_root=data_root,
        split='train',
        test_area=test_area,
        num_points=num_points,
        crop_strategy='sphere',
        crop_radius=crop_radius,
        samples_per_room_per_epoch=samples_per_room,
        cache_mode='mmap',  # switch to 'ram' if you have enough RAM (~50GB)
        augmentations=build_train_augmentations(),
    )

    sampler = DistributedSampler(dataset) if distributed else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
    return loader


def build_test_loader(
    data_root: str,
    test_area: int = 5,
    num_points: int = 16384,
    num_workers: int = 4,
    test_stride: float = 0.5,
    batch_size_test: int = 1,
    crop_radius: float = 2.0,
) -> DataLoader:
    """
    Test loader: batch_size=1, full room coverage via grid centers.
    We override __getitem__ behavior for testing — see note below.
    """
    dataset = S3DISDataset(
        data_root=data_root,
        split='test',
        test_area=test_area,
        num_points=num_points,
        crop_strategy='sphere',
        crop_radius=crop_radius,
        samples_per_room_per_epoch=0,  # not used for test
        cache_mode='ram',   # test rooms fit in RAM, faster repeated access
        augmentations=None,
        test_stride=test_stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size_test,     # test is always bs=1 in standard protocol
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader

'''
# ─── Epoch callback hook ──────────────────────────────────────────────────────
class EpochResamplingCallback:
    """Call dataset.on_epoch_end() after each training epoch."""

    def __init__(self, dataset: S3DISDataset):
        self.dataset = dataset

    def on_epoch_end(self):
        self.dataset.on_epoch_end()
        print(f"Epoch ended: resampled {len(self.dataset)} crops")
'''