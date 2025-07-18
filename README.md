# SLNet: A Superlight Network for Point Cloud Learning with Nonparametric Adaptive Point Embedding

A PyTorch implementation of SLNet, a superlight network for point cloud classification and segmentation based on the paper **“SLNet: A Super‑Lightweight and Geometry‑Adaptive Network\\for 3D Point Cloud Classification and Segmentation”**. SLNet uses a novel **Nonparametric Adaptive Point Embedding (NAPE)** module which embeds raw 3D coordinates using a parameter-free fusion of Gaussian and cosine responses with adaptive bandwidth and  blending, combined with a minimal **Geometric Modulation Unit (GMU)** which applies lightweight per-channel affine transforms for scale-aware feature modulation, a **parameter-free normalization block** enables expansion of feature dimensionality without added parameters, and **shared lightweight residual MLPs** refine features with minimal overhead to achieve competitive accuracy with only 0.14M/0.54M parameters and without relying on attention, graph convolutions, or deep residual stacks.

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
  * [Evaluation Metrics](#evaluation-metrics)
  * [Attention Map Visualization](#attention-map-visualization)
* [Directory Details](#directory-details)
* [System Requirements](#system-requirements)

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

3. **Install `gcc-10 g++-10`**
   ```bash
   sudo apt update
   sudo apt install gcc-10 g++-10
   
   export CC=gcc-10
   export CXX=g++-10
   ```

4. **Install `pointnet2_ops_lib`**

   ```bash
   cd pointnet2_ops_lib
   pip install .
   cd ..
   ```

5. **Install `pytorch3d`**

   ```bash
   cd pytorch3d
   pip install .
   cd ..
   ```

6. **Verify CUDA & GPU setup**

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
       │       └── main_split
       |           └── *.h5
       ├── shapenetcore_partanno_segmentation_benchmark_v0_normal/
       │   ├── synsetoffset2category.txt
       │   └── 02691156
       │   └── 02773838
       │   └── ...
       └── modelnet_fewshot/
           └── 5way_10shot
           └── 5way_20shot
           └── 10way_10shot
           └── 10way_20shot
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

* **GPU:** NVIDIA GeForce RTX 3090/2080 (CUDA 12.0 driver)

  ```bash
  $ nvidia-smi
  +---------------------------------------------------------------------------------------+
  | NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
  |-----------------------------------------+----------------------+----------------------+

  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 525.147.05   Driver Version: 525.147.05   CUDA Version: 12.0     |
  |-------------------------------+----------------------+----------------------|
  ```

* **CUDA Compiler Version:**

  ```
  $ nvcc --version

  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Wed_Sep_21_10:33:58_PDT_2022
  Cuda compilation tools, release 11.8, V11.8.89
  Build cuda_11.8.r11.8/compiler.31833905_0

  
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2022 NVIDIA Corporation
  Built on Wed_Sep_21_10:33:58_PDT_2022
  Cuda compilation tools, release 11.8, V11.8.89
  Build cuda_11.8.r11.8/compiler.31833905_0
  ```
