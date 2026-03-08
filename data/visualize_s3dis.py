# visualize_s3dis.py

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


LABEL_COLORS = np.array([
    [0,   255, 0  ],  # ceiling  – green
    [0,   0,   255],  # floor    – blue
    [0,   255, 255],  # wall     – cyan
    [255, 255, 0  ],  # beam     – yellow
    [255, 0,   255],  # column   – magenta
    [100, 100, 255],  # window   – light blue
    [200, 200, 100],  # door     – tan
    [170, 120, 200],  # chair    – purple
    [255, 0,   0  ],  # table    – red
    [200, 100, 100],  # bookcase – salmon
    [10,  200, 100],  # sofa     – teal
    [200, 200, 200],  # board    – gray
    [50,  50,  50 ],  # clutter  – dark gray
], dtype=np.float64) / 255.0

CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
    'door', 'chair', 'table', 'bookcase', 'sofa', 'board', 'clutter'
]


def load_room(npy_path: str) -> tuple:
    data = np.load(npy_path)
    xyz = data[:, :3]
    rgb = data[:, 3:6]
    labels = data[:, 6].view(np.int64) if data.dtype == np.float32 else data[:, 6].astype(np.int64)
    return xyz, rgb, labels


def visualize_room_labels(npy_path: str, use_gt_colors: bool = True):
    """Open3D visualization of a room with semantic label colors."""
    xyz, rgb, labels = load_room(npy_path)
    colors = LABEL_COLORS[labels]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors if use_gt_colors else rgb)

    o3d.visualization.draw_geometries([pcd], window_name=Path(npy_path).stem)


def visualize_crop_region(npy_path: str, center: np.ndarray,
                           radius: float = 2.0):
    """Visualize a spherical crop region within the room."""
    xyz, rgb, labels = load_room(npy_path)
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    indices = np.array(tree.query_ball_point(center, radius))

    # Full room: gray
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(xyz)
    pcd_full.colors = o3d.utility.Vector3dVector(
        np.full((len(xyz), 3), 0.6))

    # Crop: colored by label
    pcd_crop = o3d.geometry.PointCloud()
    pcd_crop.points = o3d.utility.Vector3dVector(xyz[indices])
    pcd_crop.colors = o3d.utility.Vector3dVector(LABEL_COLORS[labels[indices]])

    # Sphere wireframe
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(center)
    sphere.paint_uniform_color([1, 0, 0])
    sphere_wire = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)

    o3d.visualization.draw_geometries(
        [pcd_full, pcd_crop, sphere_wire],
        window_name=f"Crop at {center}"
    )


def plot_class_distribution(data_root: str, split_files: list):
    """Bar chart of point count per class across split."""
    counts = np.zeros(13, dtype=np.int64)
    for f in split_files:
        data = np.load(f)
        labels = data[:, 6].view(np.int64) if data.dtype == np.float32 else data[:, 6].astype(np.int64)
        for i in range(13):
            counts[i] += (labels == i).sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CLASS_NAMES, counts, color=[LABEL_COLORS[i] for i in range(13)])
    ax.set_ylabel('Point Count')
    ax.set_title('Class Distribution')
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()


def visualize_batch(batch: dict, max_samples: int = 4):
    """Visualize multiple samples from a DataLoader batch."""
    pos = batch['pos'].numpy()    # (B, N, 3)
    labels = batch['y'].numpy()   # (B, N)
    B = min(pos.shape[0], max_samples)

    pcds = []
    for i in range(B):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos[i])
        pcd.colors = o3d.utility.Vector3dVector(LABEL_COLORS[labels[i]])
        pcd.translate([i * 5.0, 0, 0])  # offset each sample horizontally
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds, window_name="Batch Samples")


def plot_sampling_density(npy_path: str, n_crops: int = 100, radius: float = 2.0):
    """Heatmap of how often each point is sampled across random crops."""
    xyz, _, _ = load_room(npy_path)
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    counts = np.zeros(len(xyz), dtype=np.float32)

    for _ in range(n_crops):
        center = xyz[np.random.randint(len(xyz))]
        idx = tree.query_ball_point(center, radius)
        counts[np.array(idx)] += 1

    # Color by count: low=blue, high=red
    norm_counts = counts / counts.max()
    colors = plt.cm.hot(norm_counts)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Sampling Density")