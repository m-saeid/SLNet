# SLNet: A Super-Lightweight Geometry-Adaptive Network for 3D Point Cloud Recognition

Official implementation of the **ICRA 2026** paper:

### **SLNet: A Super-Lightweight Geometry-Adaptive Network for 3D Point Cloud Recognition**

SLNet is a super-lightweight PyTorch framework for 3D point cloud classification and segmentation.
The architecture integrates:

* **Nonparametric Adaptive Point Embedding (NAPE)**
* **Lightweight Geometric Modulation Units (GMU)**
* **Parameter-free normalization**
* **Compact residual MLP refinement**

The design objective is **maximum accuracy-per-parameter efficiency**, achieving strong performance with:

* extremely low parameter count
* minimal memory footprint
* low computational complexity
* no attention mechanisms
* no graph convolutions
* no heavy residual stacks

### This makes SLNet particularly suitable for **edge devices, embedded GPUs, and resource-constrained systems** such as NVIDIA Jetson platforms.

---

## Table of Contents

* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Dataset Preparation](#dataset-preparation)
* [Usage](#usage)

  * [Running All Tests](#running-all-tests)
  * [Classification on ModelNet40](#classification-on-modelnet40)
  * [Classification on ScanObjectNN](#classification-on-scanobjectnn)
  * [Part Segmentation on ShapeNet](#part-segmentation-on-shapenet)
  * [Few-Shot Classification](#few-shot-classification)
  * [One-Command Jetson Execution](#one-command-jetson-execution)
  * [Quick Start on Jetson Orin Nano](#quick-start-on-jetson-orin-nano)
  * [Evaluation Metrics](#evaluation-metrics)
  * [Attention Map Visualization](#attention-map-visualization)
* [Directory Details](#directory-details)
* [System Requirements](#system-requirements)
* [Jetson Orin Nano Setup](#jetson-orin-nano-setup)
* [Version Pins](#version-pins)

---

## Repository Structure

```
.
├── attention/                     
│   └── *.png          # Attention map images comparing NAPE, NAPE+GMU, and DGCNN
├── checkpoints/                     # Trained model checkpoints
├── data/                           
│   └── dataset/                    # All datasets
├── decoder/                         # Segmentation decoders
├── encoder/                         # NAPE, GMU, backbone modules
├── pointnet2_ops_lib/               # Custom CUDA/C++ ops (compiled)
├── pytorch3d/                       # PyTorch3D source (compiled from source)
├── scripts/                         # Training / evaluation scripts
├── tasks/                           # Python entry points
├── utils/                           # Utilities (logging, loaders, metrics)
├── requirements.txt                 
└── README.md                        
```

---

# Installation

SLNet supports:

* ✅ **x86_64 Linux + NVIDIA Desktop GPU**
* ✅ **ARM64 NVIDIA Jetson Orin (JetPack 6.x)**

---

# 🔹 Option A — x86_64 Desktop GPU (CUDA 12.x)

## 1) Clone Repository

```bash
git clone https://github.com/m-saeid/SLNet
cd SLNet
```

---

## 2) Create Virtual Environment

```bash
python3 -m venv ~/venvs/slnet
source ~/venvs/slnet/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 3) Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4) Install PyTorch (Desktop GPU)

Install the version matching your CUDA version.

Example (CUDA 12.1):

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python - << 'EOF'
import torch
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
EOF
```

---

### 5) Compiler setup

```bash
sudo apt update
sudo apt install gcc-10 g++-10

export CC=gcc-10
export CXX=g++-10
```
Optional: Add the export lines to ~/.bashrc for persistence.

---

## 6) Install pointnet2_ops_lib

```bash
cd pointnet2_ops_lib
pip install .
cd ..
```

---

## 7) Install PyTorch3D

```bash
pip install pytorch3d
```

Or build from source:

```bash
git clone https://github.com/facebookresearch/pytorch3d
cd pytorch3d
pip install -e .
cd ..
```

---

## 8) Final Verification

```bash
python - << 'EOF'
import torch
import pointnet2_ops
import pytorch3d

print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
EOF
```

---

# 🔹 Option B — Jetson Orin Nano / Orin Family (JetPack 6.x, CUDA 12.6)

> ⚠️ Jetson Orin GPUs are **Ampere (sm_87)**
> CUDA ≥ 12.0 removes legacy architectures.
> Always set `TORCH_CUDA_ARCH_LIST=8.7` when compiling CUDA extensions.

---

## 1) Enable Maximum Performance Mode (Recommended)

Before building CUDA extensions:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 2) Clone Repository

```bash
git clone https://github.com/m-saeid/SLNet
cd SLNet
```

---

## 3) Create Virtual Environment

```bash
python3 -m venv ~/venvs/slnet
source ~/venvs/slnet/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 4) Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 5) Install PyTorch (Jetson – CUDA 12.6)

For JetPack 6.2 (CUDA 12.6), the following Jetson-optimized index works:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

### Why this is required

Jetson uses:

* ARM64 (aarch64)
* CUDA integrated with JetPack
* Ampere GPU (sm_87)

Standard PyPI wheels **do not provide CUDA acceleration** on Jetson.

---

### Verify CUDA

```bash
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
EOF
```

Expected output:

```
CUDA: True
Device: Orin (nvgpu)
```

---

## 6) Compiler Setup (Required for CUDA Extensions)

```bash
sudo apt update
sudo apt install gcc-10 g++-10

export CC=gcc-10
export CXX=g++-10
```

Optional (persistent):

```bash
echo 'export CC=gcc-10' >> ~/.bashrc
echo 'export CXX=g++-10' >> ~/.bashrc
```

---

## 7) Install pointnet2_ops_lib (Jetson)

Set CUDA architecture:

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
export FORCE_CUDA=1
```

Build and install:

```bash
cd pointnet2_ops_lib
pip install -e . --no-build-isolation
cd ..
```

If building fails, ensure the following line is present inside:
* pointnet2_ops_lib/setup.py
* pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py
```python
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.7"
```

Expected result:
```nginx
Successfully built pointnet2_ops
Successfully installed pointnet2_ops-3.0.0
```

Verify:
```bash
python - << 'EOF'
import pointnet2_ops._ext
print("pointnet2_ops extension loaded successfully")
EOF
```

---

## 8) Install PyTorch3D (Build from Source on Jetson)

```bash
git clone https://github.com/facebookresearch/pytorch3d
cd pytorch3d

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.7"
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2

pip install -e . --no-build-isolation
cd ..
```

This compiles CUDA kernels for:

* `knn_points`
* `ball_query`
* `sample_farthest_points`


Verify PyTorch3D CUDA
```bash
python - << 'EOF'
import torch
from pytorch3d.ops import knn_points
print("CUDA available:", torch.cuda.is_available())
EOF
```

---

## 9) Final Verification (Jetson)

```bash
python - << 'EOF'
import torch
import pointnet2_ops
import pytorch3d

print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
EOF
```

#### Jetson Orin output should resemble:
```vbnet
CUDA: True
Device: Orin (nvgpu)
```

---

### Jetson Notes

* GPU Architecture: **sm_87**
* Always export:

```bash
export TORCH_CUDA_ARCH_LIST=8.7
export FORCE_CUDA=1
```

* Use performance mode when compiling:

  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```

---

After completing these steps, SLNet runs with full CUDA acceleration on both:

* ✅ Desktop NVIDIA GPUs
* ✅ Jetson Orin devices

---


## Dataset Preparation

Place datasets under:

```
data/dataset/
```

Required structure:

```
data/
└── dataset/
    ├── modelnet40_ply_hdf5_2048/
    ├── scanobject/h5_files/
    ├── shapenetcore_partanno_segmentation_benchmark_v0_normal/
    └── modelnet_fewshot/
```

No manual preprocessing is required.
All sampling and normalization are handled internally.

---

## Usage

All tasks are launched via `scripts/`:

```bash
cd SLNet
```

---

### Running All Tests

```bash
bash scripts/run_all_test.sh
```

---

### Classification on ModelNet40

```bash
bash scripts/run_modelnet.sh
```

---

### Classification on ScanObjectNN

```bash
bash scripts/run_scanobject.sh
```

---

### Part Segmentation on ShapeNet

```bash
bash scripts/run_shapenet.sh
```

---

### Few-Shot Classification

```bash
bash scripts/run_fewshot.sh
```

---

## One-Command Jetson Execution

Unified launcher for Jetson devices:

```bash
source ~/venvs/slnet/bin/activate
cd SLNet

python tasks/cls_modelnet.py --device cuda
```

This directly runs the main model using GPU acceleration without shell scripts.

---

## Quick Start on Jetson Orin Nano

```bash
source ~/venvs/slnet/bin/activate
cd SLNet

# ModelNet40
bash scripts/run_modelnet.sh

# Few-shot
bash scripts/run_fewshot.sh
```

---

### GPU + PyTorch3D CUDA Validation

```python
import torch
from pytorch3d.ops import knn_points, ball_query, sample_farthest_points

print("Torch CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

x = torch.randn(1, 1000, 3, device='cuda')
y = torch.randn(1, 1000, 3, device='cuda')

d, i, _ = knn_points(x, y, K=3)
print("knn_points CUDA:", d.is_cuda, i.is_cuda)

idx = ball_query(x, y, radius=0.2, K=5)
print("ball_query CUDA:", idx[0].is_cuda)

pts, idx2 = sample_farthest_points(x, K=100)
print("fps CUDA:", pts.is_cuda, idx2.is_cuda)
```

Expected output: all `True`.

---

## Evaluation Metrics

```bash
bash scripts/eval_model.sh
```

Computes:

* GFLOPs
* peak GPU memory
* inference latency
* parameter count

---

## Attention Map Visualization

```bash
bash scripts/attention_map.sh
```

Outputs PNG comparisons between:

* NAPE
* NAPE + GMU
* DGCNN (2nd layer)

---

## Directory Details

* `encoder/` – NAPE, GMU, backbone
* `decoder/` – segmentation heads
* `tasks/` – training/eval entry points
* `scripts/` – automation wrappers
* `utils/` – loaders, metrics, logging
* `attention/` – visualization outputs
* `checkpoints/` – trained weights

---

## Benchmark Results

All benchmark results below follow the exact evaluation protocol described in the paper.
Unless otherwise stated, profiling is performed on a single RTX 2080 (CUDA 11.8) with identical compile flags and cuDNN benchmarking disabled for determinism.

---

### ModelNet40 (Supervised Classification)

| Model            | OA (%)    | mAcc (%)  | Param (M) | FLOPs (G) | Mem (MB)  | Time (ms) | NetScore  | NetScore+ |
| ---------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| PointNet         | 90.04     | 86.52     | 3.47      | 0.44      | 30.20     | 0.53      | 76.34     | 70.29     |
| PointNet++ (SSG) | 92.31     | 89.65     | 1.47      | 0.85      | 34.15     | 5.07      | 77.61     | 66.41     |
| DGCNN            | 92.82     | 89.09     | 1.81      | 2.68      | 86.22     | 6.05      | 71.84     | 58.25     |
| CurveNet         | 93.38     | -         | 2.14      | 0.32      | 29.45     | 11.70     | 80.34     | 67.65     |
| PointMLP         | 93.66     | 90.99     | 13.23     | 15.67     | 94.80     | 5.96      | 55.69     | 41.93     |
| **SLNet-S**      | **93.64** | 89.46     | **0.14**  | **0.31**  | **19.61** | 0.95      | **92.45** | **86.10** |
| **SLNet-M**      | 93.92     | **91.10** | 0.54      | 1.21      | 32.09     | 1.75      | 80.72     | 72.00     |

---

### ModelNet-R (Supervised Classification)

| Model       | OA (%)    | mAcc (%)  | NetScore  | NetScore+ |
| ----------- | --------- | --------- | --------- | --------- |
| PointNet    | 91.39     | 88.79     | 76.60     | 70.58     |
| PointNet++  | 94.02     | 92.40     | 77.96     | 66.77     |
| DGCNN       | 94.03     | 92.64     | 72.07     | 58.49     |
| CurveNet    | 94.12     | 92.65     | 80.59     | 67.91     |
| PointMLP    | **95.33** | **94.30** | 56.00     | 42.24     |
| **SLNet-S** | 94.53     | 92.21     | **92.65** | **86.30** |
| **SLNet-M** | 94.81     | 93.76     | 88.53     | 79.78     |

---

### Few-Shot Classification (ModelNet40)

| Model       | 5w-10s   | 5w-20s   | 10w-10s  | 10w-20s  |
| ----------- | -------- | -------- | -------- | -------- |
| PointNet    | 52.0     | 57.8     | 46.6     | 35.2     |
| PointCNN    | 65.4     | 68.6     | 46.6     | 50.0     |
| Point-NN    | 88.8     | 90.9     | 79.9     | 84.9     |
| NPNet       | **92.0** | 93.2     | **82.5** | 87.6     |
| **SLNet-S** | 84.0     | 89.0     | 75.5     | 93.5     |
| **SLNet-M** | 89.0     | **95.0** | 80.0     | **94.0** |

---

### ScanObjectNN (Supervised Classification)

| Model       | OA (%)   | mAcc (%) | Param (M) | FLOPs (G) | Mem (MB)  | Time (ms) | NetScore  | NetScore+ |
| ----------- | -------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- |
| PointNet    | 68.2     | 63.4     | 3.46      | 0.44      | 30.17     | 0.52      | 71.52     | 65.64     |
| PointNet++  | 77.9     | 75.4     | 1.47      | 0.85      | 34.12     | 5.02      | 74.70     | 63.52     |
| DGCNN       | 78.1     | 73.6     | 1.80      | 2.68      | 86.19     | 6.23      | 68.87     | 55.22     |
| PointMLP    | **85.4** | **83.9** | 13.23     | 15.67     | 94.80     | 5.96      | 54.09     | 40.33     |
| **SLNet-S** | 83.45    | 81.45    | **0.12**  | **0.26**  | **16.87** | 0.94      | **91.76** | **85.75** |
| **SLNet-M** | 84.25    | 82.86    | 0.48      | 1.02      | 26.47     | 1.69      | 80.12     | 71.85     |

---

### ShapeNetPart (Part Segmentation)

| Model            | ins-IoU | cls-IoU  | Param (M) | FLOPs (G) | Mem (MB) | Time (ms) | NetScore  | NetScore+ |
| ---------------- | ------- | -------- | --------- | --------- | -------- | --------- | --------- | --------- |
| PointNet         | 83.7    | 80.4     | 8.34      | 5.79      | 124.39   | 1.90      | 59.37     | 47.50     |
| PointNet++ (SSG) | 85.1    | 81.9     | 1.40      | 1.12      | 42.20    | 5.74      | 74.57     | 62.65     |
| DGCNN            | 85.2    | 82.3     | 1.46      | 4.96      | 157.26   | 9.86      | 68.01     | 52.06     |
| PointMLP         | 86.1    | **84.6** | 16.75     | 6.26      | 121.04   | 4.42      | 56.88     | 43.24     |
| **SLNet-S**      | 85.27   | 83.99    | **1.24**  | **0.90**  | 46.86    | 3.30      | **76.49** | **65.54** |
| **SLNet-M**      | 85.53   | 84.45    | 1.90      | 2.33      | 49.45    | 4.37      | 70.60     | 58.93     |

---

### Embedded GPU Performance (Jetson Orin Nano)

Measured on Jetson Orin Nano (1024 points, batch size 4).

| Dataset      | Model   | GFLOPs   | Param (M) | Mem (MB)  | Time (ms) |
| ------------ | ------- | -------- | --------- | --------- | --------- |
| ModelNet40   | SLNet-S | **0.31** | **0.14**  | **21.30** | **17.83** |
| ModelNet40   | SLNet-M | 1.22     | 0.55      | 33.00     | 30.11     |
| ScanObjectNN | SLNet-S | **0.26** | **0.12**  | **17.87** | **16.36** |
| ScanObjectNN | SLNet-M | 1.02     | 0.48      | 27.37     | 27.43     |
| ShapeNetPart | SLNet-S | **0.83** | **1.24**  | **30.09** | **38.50** |
| ShapeNetPart | SLNet-M | 2.25     | 1.90      | 48.27     | 57.81     |

---

### Composite Indices

NetScore:

NetScore = 20 log10( a² / sqrt(p · m) )

NetScore+:

NetScore+ = 20 log10( a² / ( sqrt(p · m) · (t · r)^(1/4) ) )

Where:

* a = accuracy
* p = parameters
* m = FLOPs
* t = latency
* r = peak memory

Higher is better for both metrics.

---

---

## Architectural Overview

This section provides a visual breakdown of SLNet components and efficiency characteristics.

---

### 1. Overall Architecture

![SLNet Architecture](images/arch.pdf)

**Pipeline:**

1. **NAPE Front-End** – Nonparametric Adaptive Point Embedding
2. **GMU Front-End** – Geometric Modulation Unit
3. **Four Encoder Stages**

   * Farthest Point Sampling
   * Local grouping (ball query / kNN)
   * Parameter-free normalization
   * Lightweight residual MLP refinement
4. **Task Heads**

   * Classification head with global pooling
   * U-Net–style decoder for segmentation with feature propagation and global context

The architecture eliminates attention layers and heavy graph convolutions while preserving geometric awareness.

---

### 2. NAPE Block

![NAPE Block](images/nape.pdf)

The Adaptive Point Embedding (APE) block maps raw 3D coordinates using:

* Fused Gaussian RBF bases
* Cosine harmonic bases
* Adaptive bandwidth estimation
* Parameter-free geometric encoding

This produces geometry-aware embeddings without learned convolutional kernels.

---

### 3. Geometric Modulation Unit (GMU)

![GMU Block](images/gmu.pdf)

GMU performs lightweight geometric feature modulation via:

* Local context aggregation
* Channel-wise modulation
* Minimal parameter overhead
* Residual refinement coupling

It acts as a geometry-sensitive gating mechanism between embedding and encoder refinement.

---

### 4. Efficiency Radar Plot

![Radar Plot](images/radar_plot.png)

Radar plot comparing **NetScore** and **NetScore+** for:

* SLNet-S
* SLNet-M
* PointNet
* PointNet++
* DGCNN
* PointMLP

SLNet-S dominates in composite efficiency metrics due to its ultra-low parameter count and reduced FLOPs.

---

### 5. Qualitative Saliency Comparison

![Qualitative Comparison](qualitative_comparison.png)

Comparison on ModelNet40 samples:

* Top row: DGCNN stage-2 EdgeConv gradient saliency
* Middle row: NAPE gradient saliency (channel width 16)
* Bottom row: NAPE gradient saliency (channel width 32)

NAPE produces sharper and more semantically focused responses despite operating at the input embedding stage.

---

## System Requirements

### Standard Linux GPU

* NVIDIA GPU (RTX / A-series)
* CUDA ≥ 11.8
* Python ≥ 3.9

---

## Jetson Orin Nano Setup

| Component | Value                        |
| --------- | ---------------------------- |
| Device    | Jetson Orin Nano Engineering |
| JetPack   | 6.2.2                        |
| CUDA      | 12.6                         |
| Driver    | 540.5.0                      |
| Python    | 3.10.12                      |
| RAM       | 8GB                          |
| Swap      | 3.7GB                        |
| Storage   | 56GB (35GB free)             |

Performance mode:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Version Pins

```text
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
pytorch3d==0.7.8
numpy==1.26.4
```

---

## Design Goal

SLNet is explicitly designed for:

* embedded inference
* robotics perception
* autonomous systems
* edge AI
* low-power GPUs
* mobile robotics
* real-time 3D perception

with full CUDA acceleration and minimal memory pressure.

---

## 📝 Citation

```bibtex
@article{saeid2026slnet,
  title={SLNet: A Super-Lightweight Geometry-Adaptive Network for 3D Point Cloud Recognition},
  author={Saeid, Mohammad and Salarpour, Amir and MohajerAnsari, Pedram and Pes{\'e}, Mert D},
  journal={...},
  year={2026}
}
```

---

## 📬 Contact

📧 Questions? Reach out to: **[imm.saeid@gmail.com](imm.saeid@gmail.com)**
