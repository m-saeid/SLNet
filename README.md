# SLNet: A Superlight Network for Point Cloud Learning with Hybrid Nonparametric Embedding

A lightweight PyTorch implementation of SLNet, a superlight network for point cloud classification and segmentation based on the paper [“SLNet: A Superlight Network for Point Cloud Learning with Hybrid Nonparametric Embedding”](https://arxiv.org/abs/XXXX.XXXX). SLNet uses a novel Nonparametric Adaptive Point Embedding (NAPE) module combined with a minimal Geometric Modulation Unit (GMU) to achieve competitive accuracy with only 0.14M parameters and without relying on attention, graph convolutions, or deep residual stacks.

---

## Table of Contents

* [Paper Abstract](#paper-abstract)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Dataset Preparation](#dataset-preparation)
* [Usage](#usage)

  * [Running All Tests](#running-all-tests)
  * [Classification on ModelNet40](#classification-on-modelnet40)
  * [Classification on ScanObjectNN](#classification-on-scanobjectnn)
  * [Part Segmentation on ShapeNet](#part-segmentation-on-shapenet)
  * [Few-Shot Classification](#few-shot-classification)
  * [Evaluation Metrics](#evaluation-metrics)
  * [Attention Map Visualization](#attention-map-visualization)
* [Directory Details](#directory-details)
* [System Requirements](#system-requirements)
* [Citation](#citation)

---

## Paper Abstract

> Point cloud understanding has advanced rapidly with deep neural architectures that exploit local geometry and permutation invariance. However, state-of-the-art models often rely on expensive modules—such as attention layers, graph convolutions, or deep residual stacks—resulting in high memory and compute costs. In this paper, we propose SLNet, a superlight network for point cloud classification and segmentation that combines nonparametric encoding with minimal parameterization. At its core is a novel Nonparametric Adaptive Point Embedding (NAPE) module that maps raw 3D coordinates to a high-dimensional space using a hybrid of Gaussian radial basis functions and cosine encodings. The kernel bandwidth is adapted to the spatial dispersion of the input, enabling flexible geometric modeling without learnable weights. A lightweight Geometric Modulation Unit (GMU) introduces per-channel affine adaptation at negligible cost. These components feed into a residual shared-MLP backbone that omits graph or attention-based operations entirely. Despite its extreme compactness (only 0.14M parameters), SLNet achieves competitive accuracy on ModelNet40 and strong generalization on part and semantic segmentation tasks. We also demonstrate that NAPE alone can power a zero-parameter, similarity-based classifier with surprising effectiveness, highlighting its strong geometric inductive bias.

---

## Features

* **Nonparametric Adaptive Point Embedding (NAPE):**

  * Hybrid of Gaussian radial basis functions and cosine encodings.
  * Adapted kernel bandwidth based on spatial dispersion—no learnable weights.

* **Geometric Modulation Unit (GMU):**

  * Per-channel affine adaptation at negligible computational cost.

* **Residual Shared-MLP Backbone:**

  * No attention layers, graph convolutions, or deep residual stacks—extremely lightweight.

* **Zero-Parameter Classifier:**

  * NAPE alone can act as a similarity-based classifier.

* **Competitive Performance:**

  * \~0.14M parameters, achieves state-of-the-art‐adjacent accuracy on ModelNet40, ScanObjectNN, and ShapeNet part segmentation.

---

## Repository Structure

```
.
├── attention/                     
│   └── *.png          # Attention map images comparing:
│                        # - NAPE block
│                        # - NAPE+GMU block
│                        # - 2nd layer of DGCNN
├── checkpoints/                     
├── data/                           
│   └── dataset/                    
│       ├── modelnet40_ply_hdf5_2048/
│       ├── scanobject/h5_files/
│       ├── shapenetcore_partanno_segmentation_benchmark_v0_normal/
│       └── modelnet_fewshot/
├── decoder/                         
├── encoder/                         
├── pointnet2_ops_lib/               # Custom ops; install separately
├── pytorch3d/                       # PyTorch3D submodule; install separately
├── scripts/                         
│   ├── attention_map.sh             # Generate attention maps
│   ├── eval_model.sh                # Compute GFLOPs, peak memory, params, inference time
│   ├── run_all_test.sh              # Quick smoke-test: run all tasks for 1 epoch
│   ├── run_fewshot.sh               # Few-shot classification
│   ├── run_modelnet.sh              # Classification on ModelNet40
│   ├── run_scanobject.sh            # Classification on ScanObjectNN
│   └── run_shapenet.sh              # Part segmentation on ShapeNet
├── tasks/                           
│   ├── attention_map.py             
│   ├── cls_fewshot.py               
│   ├── cls_modelnet.py              
│   ├── cls_scanobject.py            
│   ├── eval_model.py                
│   └── partseg_shapenet.py          
├── utils/                           
├── requirements.txt                 
└── README.md                        
```

---

## Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/SLNet.git
   cd SLNet
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install `pointnet2_ops_lib`**

   ```bash
   cd pointnet2_ops_lib
   pip install .
   cd ..
   ```

4. **Install `pytorch3d`**

   ```bash
   cd pytorch3d
   pip install .
   cd ..
   ```

5. **Verify CUDA & GPU setup**

   * Ensure CUDA 11.x or 12.x is installed and matches your GPU drivers.
   * Confirm with:

     ```bash
     nvidia-smi
     nvcc --version
     ```

---

## Dataset Preparation

1. **Download Datasets** and place them under `data/dataset/` with the following folder structure:

   * `data/dataset/modelnet40_ply_hdf5_2048/`
   * `data/dataset/scanobject/h5_files/`
   * `data/dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/`
   * `data/dataset/modelnet_fewshot/`

2. **Directory Tree Example**

   ```bash
   data/
   └── dataset/
       ├── modelnet40_ply_hdf5_2048/
       │   ├── train_files.txt
       │   ├── test_files.txt
       │   └── *.h5
       ├── scanobject/
       │   └── h5_files/
       │       └── *.h5
       ├── shapenetcore_partanno_segmentation_benchmark_v0_normal/
       │   ├── train_test_split/
       │   └── *.h5
       └── modelnet_fewshot/
           └── ...
   ```

3. **Preprocessing**
   No additional preprocessing is required—data loaders handle point sampling internally.

---

## Usage

All training and evaluation tasks can be launched using the shell scripts in the `scripts/` directory. Edit these scripts to modify hyperparameters, paths, or other arguments as needed.

Before running any script, ensure you are in the repository root:

```bash
cd /path/to/SLNet
```

### Running All Tests

A quick smoke test that runs each task for 1 epoch to verify setup:

```bash
bash scripts/run_all_test.sh
```

### Classification on ModelNet40

Train and evaluate SLNet on ModelNet40:

```bash
bash scripts/run_modelnet.sh
```

* **Default Hyperparameters:**

  * Batch size: 32
  * Learning rate: 0.001
  * Epochs: 200 (modifiable in the script)
  * Output checkpoints saved to `checkpoints/modelnet40/`

### Classification on ScanObjectNN

Train and evaluate SLNet on ScanObjectNN:

```bash
bash scripts/run_scanobject.sh
```

* **Default Hyperparameters:**

  * Batch size: 32
  * Learning rate: 0.001
  * Epochs: 200
  * Output checkpoints saved to `checkpoints/scanobject/`

### Part Segmentation on ShapeNet

Train and evaluate SLNet for part segmentation on ShapeNet:

```bash
bash scripts/run_shapenet.sh
```

* **Default Hyperparameters:**

  * Batch size: 16
  * Learning rate: 0.001
  * Epochs: 300
  * Output segmentation results to `checkpoints/shapenet/`

### Few-Shot Classification

Few-shot experiments on ModelNet40:

```bash
bash scripts/run_fewshot.sh
```

* **Setup:**

  * Predefine “N-way K-shot” splits in `data/dataset/modelnet_fewshot/`.
  * Script loads the splits and runs episodes accordingly.

### Evaluation Metrics

Compute GFLOPs, peak memory, number of parameters, and inference time for a trained checkpoint:

```bash
bash scripts/eval_model.sh
```

* **Usage:**

  * Edit `eval_model.sh` to point to the desired checkpoint path.
  * Outputs a text summary under the `eval/` directory.

### Attention Map Visualization

Generate and save attention map images comparing NAPE, NAPE+GMU, and the 2nd layer of DGCNN:

```bash
bash scripts/attention_map.sh
```

* **Output:**

  * Saved PNGs in the `attention/` folder.

---

## Directory Details

* **`attention/`**
  Contains comparison visualizations (PNG) of:

  1. SLNet’s NAPE block
  2. NAPE + GMU block
  3. 2nd layer of DGCNN

* **`checkpoints/`**
  Stores trained model weights and associated logs. Each task creates its own subfolder (e.g., `modelnet40/`, `scanobject/`, `shapenet/`, `fewshot/`).

* **`data/`**
  Root directory for all datasets. Subfolders need to be populated manually as described above.

* **`decoder/`**
  SLNet’s decoder modules for segmentation tasks.

* **`encoder/`**
  SLNet’s encoder modules (NAPE, GMU, and shared-MLP backbone).

* **`pointnet2_ops_lib/`**
  Custom CUDA/C++ operations required by pointnet2. Install with `pip install .`.

* **`pytorch3d/`**
  Submodule for PyTorch3D. Install with `pip install .`.

* **`scripts/`**
  Shell wrappers to launch training, evaluation, and visualization pipelines.

* **`tasks/`**
  Python entry points for each task:

  * `attention_map.py`
  * `cls_fewshot.py`
  * `cls_modelnet.py`
  * `cls_scanobject.py`
  * `eval_model.py`
  * `partseg_shapenet.py`

* **`utils/`**
  Helper functions (data loaders, logging, metrics).

* **`requirements.txt`**
  Python dependencies (PyTorch ≥1.10, numpy, h5py, etc.).

---

## System Requirements

The following setup was used to reproduce results and benchmarks:

* **GPU:** NVIDIA GeForce RTX 3090 (CUDA 12.0 driver)

  ```bash
  $ nvidia-smi
  Fri Jun  6 17:16:19 2025
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 525.147.05   Driver Version: 525.147.05   CUDA Version: 12.0     |
  |-------------------------------+----------------------+----------------------|
  ```
  <!--
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
  |  0%   44C    P8     2W / 300W |    332MiB / 11264MiB |      6%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+
  -->

* **CUDA Compiler Version:**

  ```
  $ nvcc --version
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Wed_Sep_21_10:33:58_PDT_2022
  Cuda compilation tools, release 11.8, V11.8.89
  Build cuda_11.8.r11.8/compiler.31833905_0
  ```

* **Python 3.8+**

* **PyTorch ≥1.10**, **PyTorch3D**, **h5py**, **NumPy**, **SciPy**, etc. (see `requirements.txt`)

---

## Citation

If you find SLNet useful, please cite:

```bibtex
@inproceedings{name2025slnet,
  title     = {SLNet: A Superlight Network for Point Cloud Learning with Hybrid Nonparametric Embedding},
  author    = {Authors},
  booktitle = {Conference/Journal Name},
  year      = {2025},
  pages     = {XXX--XXX},
  url       = {https://arxiv.org/abs/XXXX.XXXX}
}
```

---

For any questions or bug reports, please open an issue on this repository or contact the authors directly.
imm.saeid@gmail.com
