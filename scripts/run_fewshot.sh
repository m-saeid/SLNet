#!/bin/bash


# Set which GPU(s) to use
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1s


# export TORCH_USE_CUDA_DSA=1

N=1024
EMBED_Modelnet='no'
INITIAL_DIM=3                   # 3
EMBED_DIM=(32) # 32)
ALPHA_BETA='yes'
SIGMA=0.4            # 0.4

RES_DIM_RATIO=0.25
BIAS=false
USE_XYZ=true
NORM_MODE='anchor'
STD_MODE='BN11'

DIM_RATIO='2-2-2-1'

NUM_BLOCKS1='1-1-2-1'
TRANSFER_MODE='mlp-mlp-mlp-mlp'
BLOCK1_MODE='mlp-mlp-mlp-mlp'

NUM_BLOCKS2='1-1-2-1'
BLOCK2_MODE='mlp-mlp-mlp-mlp'

K_ModelNet='32-32-32-32'
SAMPLING_MODE='fps-fps-fps-fps'
SAMPLING_RATIO='2-2-2-2'

CLASSIFIER_MODE='mlp_very_large'  # 'mlp_very_very_large' 'mlp_very_large' 'mlp_large' 'mlp_medium' 'mlp_small' 'mlp_very_small'

BATCH_SIZE_ModelNet=50
EPOCH_ModelNet=300
LEARNING_RATE_ModelNet=0.1
MIN_LR_ModelNet=0.005
WEIGHT_DECAY_ModelNet=2e-4

#N_WAY=(5 10)
#K_SHOT=(10 20)

#SEED=42
WORKERS_ModelNet=6
EMA='yes'

FPS_METHOD='pointops2'  # [pytorch3d, pointops2, pytorch]
KNN_METHOD='pytorch3d'  # [pytorch3d, pytorch]

for i in {1..2}; do
    for EMBED_DIM in "${EMBED_DIM[@]}"; do
        python tasks/cls_fewshot.py --embed_dim "$EMBED_DIM" --n_way 5 --k_shot 10 --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"
        python tasks/cls_fewshot.py --embed_dim "$EMBED_DIM" --n_way 5 --k_shot 20 --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"
        python tasks/cls_fewshot.py --embed_dim "$EMBED_DIM" --n_way 10 --k_shot 10 --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"
        python tasks/cls_fewshot.py --embed_dim "$EMBED_DIM" --n_way 10 --k_shot 20 --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"
        echo "====================================================================="
    done
done

