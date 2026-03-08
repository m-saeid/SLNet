# data/s3dis_dataset.py

import os
import pickle
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree


NUM_CLASSES = 13
AREA5_TEST = {'Area_5'}


# ─────────────────────────── Room Cache ──────────────────────────────────────

class RoomCache:
    """Manages room data either in RAM or memory-mapped."""

    def __init__(self, room_paths: List[Path], mode: str = 'mmap'):
        assert mode in ('ram', 'mmap')
        self.mode = mode
        self.paths = room_paths
        self._cache: Dict[int, np.ndarray] = {}

        if mode == 'ram':
            print("Preloading all rooms to RAM...")
            for i, p in enumerate(room_paths):
                self._cache[i] = np.load(p)

    def get(self, room_id: int) -> np.ndarray:
        if self.mode == 'ram':
            return self._cache[room_id]
        return np.load(self.paths[room_id], mmap_mode='r')


# ─────────────────────────── Crop Samplers ───────────────────────────────────

class SphereSampler:
    def __init__(self, radius: float = 2.0, min_points: int = 100):
        self.radius = radius
        self.min_points = min_points

    def __call__(self, xyz: np.ndarray, kdtree: cKDTree,
                 center: Optional[np.ndarray] = None) -> np.ndarray:
        if center is None:
            center = xyz[np.random.randint(len(xyz))]
        indices = kdtree.query_ball_point(center, self.radius)
        return np.array(indices, dtype=np.int64)


class BlockSampler:
    def __init__(self, block_size: float = 1.0, min_points: int = 100):
        self.block_size = block_size
        self.min_points = min_points

    def __call__(self, xyz: np.ndarray, kdtree: cKDTree,
                 center: Optional[np.ndarray] = None) -> np.ndarray:
        if center is None:
            center = xyz[np.random.randint(len(xyz))]
        half = self.block_size / 2
        mask = (
            (xyz[:, 0] >= center[0] - half) & (xyz[:, 0] <= center[0] + half) &
            (xyz[:, 1] >= center[1] - half) & (xyz[:, 1] <= center[1] + half)
        )
        return np.where(mask)[0]


# ─────────────────────────── Augmentations ───────────────────────────────────

class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, xyz, rgb, labels):
        for t in self.transforms:
            xyz, rgb, labels = t(xyz, rgb, labels)
        return xyz, rgb, labels


class RandomRotateZ:
    def __call__(self, xyz, rgb, labels):
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ R.T
        return xyz, rgb, labels


class RandomScale:
    def __init__(self, lo: float = 0.8, hi: float = 1.2):
        self.lo, self.hi = lo, hi

    def __call__(self, xyz, rgb, labels):
        scale = np.random.uniform(self.lo, self.hi)
        return xyz * scale, rgb, labels


class RandomJitter:
    def __init__(self, sigma: float = 0.005, clip: float = 0.02):
        self.sigma, self.clip = sigma, clip

    def __call__(self, xyz, rgb, labels):
        noise = np.clip(np.random.randn(*xyz.shape) * self.sigma, -self.clip, self.clip)
        return xyz + noise.astype(np.float32), rgb, labels


class RandomFlipXY:
    def __call__(self, xyz, rgb, labels):
        if np.random.rand() > 0.5:
            xyz[:, 0] = -xyz[:, 0]
        if np.random.rand() > 0.5:
            xyz[:, 1] = -xyz[:, 1]
        return xyz, rgb, labels


class ChromaticJitter:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, xyz, rgb, labels):
        noise = np.random.randn(*rgb.shape) * self.std
        rgb = np.clip(rgb + noise.astype(np.float32), 0.0, 1.0)
        return xyz, rgb, labels


class RandomHeightJitter:
    def __call__(self, xyz, rgb, labels):
        xyz[:, 2] += np.random.uniform(-0.2, 0.2)
        return xyz, rgb, labels


class RandomColorDrop:
    """Zero out RGB with probability p — forces model to learn geometry."""
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, xyz, rgb, labels):
        if np.random.rand() < self.p:
            rgb = np.zeros_like(rgb)
        return xyz, rgb, labels


class ChromaticAutoContrast:
    def __call__(self, xyz, rgb, labels):
        lo = rgb.min(axis=0)
        hi = rgb.max(axis=0)
        denom = (hi - lo).clip(min=1e-6)
        return xyz, (rgb - lo) / denom, labels


def build_train_augmentations() -> Compose:
    return Compose([
        RandomRotateZ(),
        RandomFlipXY(),
        RandomScale(0.8, 1.25),
        RandomJitter(sigma=0.005),
        ChromaticJitter(std=0.05),
        RandomColorDrop(p=0.3),       # increase from 0.2
        ChromaticAutoContrast(),
        RandomHeightJitter(),
    ])


# ─────────────────────────── Main Dataset ────────────────────────────────────

class S3DISDataset(Dataset):
    """
    S3DIS dataset with sphere-crop sampling.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        test_area: int = 5,
        num_points: int = 16384,
        crop_strategy: str = 'sphere',
        crop_radius: float = 2.0,
        block_size: float = 1.0,
        samples_per_room_per_epoch: int = 50,
        cache_mode: str = 'mmap',
        augmentations: Optional[Compose] = None,
        test_stride: float = 1.5,
        ignore_label: int = -1,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.augmentations = augmentations
        self.ignore_label = ignore_label
        self.test_stride = test_stride

        test_area_str = f'Area_{test_area}'
        all_files = sorted(self.data_root.glob('*.npy'))

        if split == 'train':
            self.room_files = [f for f in all_files if test_area_str not in f.stem]
        else:
            self.room_files = [f for f in all_files if test_area_str in f.stem]

        if not self.room_files:
            raise ValueError(f"No .npy files found in {data_root} for split={split}, "
                             f"test_area={test_area}")

        if crop_strategy == 'sphere':
            self.crop_fn = SphereSampler(radius=crop_radius)
        elif crop_strategy == 'block':
            self.crop_fn = BlockSampler(block_size=block_size)
        else:
            raise ValueError(f"Unknown crop_strategy: {crop_strategy}")

        self.cache = RoomCache(self.room_files, mode=cache_mode)

        print(f"[S3DISDataset] {split}: {len(self.room_files)} rooms, "
              f"building KD-trees...")
        self.kdtrees: List[cKDTree] = []
        self.room_xyz: List[np.ndarray] = []

        for room_file in self.room_files:
            data = np.load(room_file, mmap_mode='r')
            xyz = data[:, :3].copy()
            self.room_xyz.append(xyz)
            self.kdtrees.append(cKDTree(xyz))

        self.samples_per_room = samples_per_room_per_epoch
        self._build_sample_list()

    def _build_sample_list(self):
        if self.split == 'train':
            self.sample_list = []
            for room_id, xyz in enumerate(self.room_xyz):
                n = min(self.samples_per_room, len(xyz))
                chosen = np.random.choice(len(xyz), size=n, replace=False)
                for idx in chosen:
                    self.sample_list.append((room_id, xyz[idx].copy()))
            random.shuffle(self.sample_list)
        else:
            self.sample_list = []
            for room_id, xyz in enumerate(self.room_xyz):
                x_min, y_min = xyz[:, :2].min(axis=0)
                x_max, y_max = xyz[:, :2].max(axis=0)
                z_mean = float(xyz[:, 2].mean())
                x = x_min
                while x <= x_max + self.test_stride:
                    y = y_min
                    while y <= y_max + self.test_stride:
                        center = np.array([x, y, z_mean], dtype=np.float32)
                        self.sample_list.append((room_id, center))
                        y += self.test_stride
                    x += self.test_stride

    def on_epoch_end(self):
        if self.split == 'train':
            self._build_sample_list()

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        room_id, center = self.sample_list[idx]

        raw = self.cache.get(room_id)
        room_xyz = self.room_xyz[room_id]

        # ── Crop ─────────────────────────────────────────────────────────────
        indices = self.crop_fn(room_xyz, self.kdtrees[room_id], center)

        if len(indices) < 32:
            # Fallback to a random sample if crop is empty / tiny
            return self.__getitem__(np.random.randint(len(self)))

        xyz    = raw[indices, :3].copy().astype(np.float32)
        rgb    = raw[indices, 3:6].copy().astype(np.float32)
        labels = raw[indices, 6].astype(np.int64).copy()

        # ── Subsample / pad ──────────────────────────────────────────────────
        xyz, rgb, labels, local_choice = self._resample(xyz, rgb, labels)
        point_idx = indices[local_choice]   # original room indices, shape (N,)

        # ── Normalize xyz ────────────────────────────────────────────────────
        # Center on crop mean; scale by room diagonal (stable across rooms)
        xyz_centered   = xyz - xyz.mean(axis=0)
        room_scale     = np.linalg.norm(room_xyz.max(axis=0) - room_xyz.min(axis=0))
        scale_factor   = room_scale + 1e-8

        # ── Augmentation ─────────────────────────────────────────────────────
        if self.augmentations is not None:
            xyz_centered, rgb, labels = self.augmentations(xyz_centered, rgb, labels)

        # Normalise (applies after augmentation, so xyz is in the augmented frame)
        xyz_normalized = xyz_centered / scale_factor

        # ── Build feature vector ─────────────────────────────────────────────
        # features shape: (num_points, 6) = [R, G, B, norm_x, norm_y, norm_z]
        # features = np.concatenate([rgb, xyz_normalized], axis=1)      SUCHE HUGE MISTAKE ;o
        features = np.concatenate([xyz_normalized, rgb], axis=1)

        return {
            # 'pos':       torch.from_numpy(xyz_centered),              # (N, 3) # unused
            'x':         torch.from_numpy(features.astype(np.float32)),  # (N, 6)
            'y':         torch.from_numpy(labels.astype(np.int64)),   # (N,)
            'room_id':   torch.tensor(room_id, dtype=torch.int64),
            'point_idx': torch.from_numpy(point_idx.astype(np.int64)), # (N,)
        }

    def _resample(self, xyz, rgb, labels):
        """Subsample or repeat-pad to self.num_points. Returns (xyz, rgb, labels, choice)."""
        N = len(xyz)
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            repeat = self.num_points // N + 1
            choice = np.tile(np.arange(N), repeat)[:self.num_points]
            np.random.shuffle(choice)
        return xyz[choice], rgb[choice], labels[choice], choice
