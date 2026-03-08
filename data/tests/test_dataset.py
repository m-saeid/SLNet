# tests/test_dataset.py

import pytest
import numpy as np
from s3dis_dataset import S3DISDataset

def test_label_range(dataset):
    for i in range(min(100, len(dataset))):
        batch = dataset[i]
        assert batch['y'].min() >= 0
        assert batch['y'].max() < 13, f"Label {batch['y'].max()} out of range"

def test_point_count(dataset):
    for i in range(min(50, len(dataset))):
        batch = dataset[i]
        assert batch['pos'].shape == (dataset.num_points, 3)
        assert batch['x'].shape[1] == 6  # rgb + xyz_norm

def test_rgb_range(dataset):
    for i in range(min(50, len(dataset))):
        batch = dataset[i]
        rgb = batch['x'][:, :3]
        assert rgb.min() >= -0.1, "RGB too low (normalization issue?)"
        assert rgb.max() <= 1.1, "RGB too high (not normalized?)"

def test_no_nan(dataset):
    for i in range(min(100, len(dataset))):
        batch = dataset[i]
        assert not batch['pos'].isnan().any(), "NaN in positions"
        assert not batch['x'].isnan().any(), "NaN in features"

def test_train_test_split(data_root):
    train_ds = S3DISDataset(data_root, split='train', test_area=5)
    test_ds = S3DISDataset(data_root, split='test', test_area=5)
    train_rooms = {f.stem for f in train_ds.room_files}
    test_rooms = {f.stem for f in test_ds.room_files}
    assert len(train_rooms & test_rooms) == 0, "Train/test room overlap!"
    assert all('Area_5' in r for r in test_rooms)
    assert all('Area_5' not in r for r in train_rooms)