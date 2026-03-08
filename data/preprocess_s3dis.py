# preprocess_s3dis.py

# python data/preprocess_s3dis.py --raw data/dataset/s3dis/Stanford3dDataset_v1.2_Aligned_Version --out data/dataset/s3dis/processed/Stanford3dDataset_v1.2_Aligned_Version
# Issue https://github.com/Pointcept/PointTransformerV2/issues/25

import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm

CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
    'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

def parse_room(room_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse raw S3DIS room folder → xyz (N,3), rgb (N,3), labels (N,)"""
    xyz_list, rgb_list, label_list = [], [], []

    for ann_file in sorted((room_path / 'Annotations').glob('*.txt')):
        class_name = ann_file.stem.rsplit('_', 1)[0].lower()
        label_idx = CLASS2IDX.get(class_name, CLASS2IDX['clutter'])

        try:
            pts = np.loadtxt(ann_file, dtype=np.float32)
        except Exception:
            pts = np.loadtxt(ann_file, dtype=np.float32, comments='//')

        xyz_list.append(pts[:, :3])
        rgb_list.append(pts[:, 3:6])
        label_list.append(np.full(len(pts), label_idx, dtype=np.int64))

    xyz = np.concatenate(xyz_list, axis=0)
    rgb = np.concatenate(rgb_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    # Normalize RGB to [0, 1]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    return xyz, rgb, labels


def grid_subsample(xyz: np.ndarray, rgb: np.ndarray, labels: np.ndarray,
                   voxel_size: float = 0.04) -> tuple:
    """Voxel grid subsampling — keeps one random point per voxel."""
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    # Use a dict to collect one point per voxel
    voxel_map: dict[tuple, int] = {}
    keep_indices = []
    for i, c in enumerate(map(tuple, coords)):
        if c not in voxel_map:
            voxel_map[c] = i
            keep_indices.append(i)
    idx = np.array(keep_indices)
    return xyz[idx], rgb[idx], labels[idx]


def preprocess_s3dis(raw_root: str, out_root: str, voxel_size: float = 0.04):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for area_dir in sorted(raw_root.glob('Area_*')):
        area_id = area_dir.name  # e.g. "Area_1"
        for room_dir in tqdm(sorted(area_dir.iterdir()), desc=area_id):
            if not room_dir.is_dir():
                continue
            out_path = out_root / f"{area_id}_{room_dir.name}.npy"
            if out_path.exists():
                continue

            xyz, rgb, labels = parse_room(room_dir)

            if voxel_size > 0:
                xyz, rgb, labels = grid_subsample(xyz, rgb, labels, voxel_size)

            # Store as structured array: xyz(3) + rgb(3) + label(1)
            data = np.concatenate([
                xyz.astype(np.float32),
                rgb.astype(np.float32),
                labels[:, None].astype(np.float32) # labels[:, None].astype(np.int64).view(np.float32),
            ], axis=1)

            np.save(out_path, data)
            print(f"  Saved {room_dir.name}: {len(xyz)} points → {out_path.name}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--voxel_size', type=float, default=0.04)
    args = parser.parse_args()
    preprocess_s3dis(args.raw, args.out, args.voxel_size)
