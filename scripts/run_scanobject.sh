#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

N=1024
EMBED_Scanobject='no'                 # ('no' 'mlp' 'nape') > no: nape is performed in dataloader
INITIAL_DIM=3
EMBED_DIM=(16 32)                     # (16 32)
ALPHA_BETA='no'                       # ('yes' 'no')
SIGMA=0.4

RES_DIM_RATIO=0.25
# BIAS=false
# USE_XYZ=true
NORM_MODE='anchor'                  # ('nearest_to_mean' 'anchor' 'center')
STD_MODE='BN11'                     # ('BN1D' 'BN11' '1111' 'B111')

DIM_RATIO='2-2-2-1'

NUM_BLOCKS1='1-1-3-1'
TRANSFER_MODE='mlp-mlp-mlp-mlp'
BLOCK1_MODE='mlp-mlp-mlp-mlp'

NUM_BLOCKS2='1-1-2-1'
BLOCK2_MODE='mlp-mlp-mlp-mlp'

K_ScanObject='24-24-24-24'
SAMPLING_MODE='fps-fps-fps-fps'
SAMPLING_RATIO='2-2-2-2'

CLASSIFIER_MODE='mlp_very_small'    # ('mlp_very_small' 'mlp_very_large' 'mlp_large' 'mlp_medium' 'mlp_small')

BATCH_SIZE_ScanObject=50
EPOCH_ScanObject=300
LEARNING_RATE_ScanObject=0.01
MIN_LR_ScanObject=0.005
WEIGHT_DECAY_ScanObject=1e-4

#SEED=42
WORKERS_ScanObject=6
EMA='yes'

FPS_METHOD='pointops2'  # [pytorch3d, pointops2, pytorch]
KNN_METHOD='pytorch3d'  # [pytorch3d, pytorch]

for i in {1..2}; do
    for EMBED_DIM in "${EMBED_DIM[@]}"; do
        python tasks/cls_scanobject.py --n "$N" --embed "$EMBED_Scanobject" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
        --res_dim_ratio "$RES_DIM_RATIO" --norm_mode "$NORM_MODE" --std_mode "$STD_MODE" --sigma "$SIGMA" \
        --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS1" --transfer_mode "$TRANSFER_MODE" \
        --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$K_ScanObject" \
        --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
        --batch_size "$BATCH_SIZE_ScanObject" --epoch "$EPOCH_ScanObject" --learning_rate "$LEARNING_RATE_ScanObject" --min_lr "$MIN_LR_ScanObject" \
        --weight_decay "$WEIGHT_DECAY_ScanObject" --workers "$WORKERS_ScanObject" \
        --alpha_beta "$ALPHA_BETA" --ema "$EMA" --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"
        echo "====================================================================="
        # --use_xyz True --bias False --seed "$SEED"
    done
done




