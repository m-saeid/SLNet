#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

N=1024
EMBED_Modelnet='no'                 # ('no' 'mlp' 'nape') > no: nape is performed in dataloader
INITIAL_DIM=3
EMBED_DIM=(16 32)                   # (16 32)
ALPHA_BETA='yes'                    # ('yes' 'no')
SIGMA=0.4

RES_DIM_RATIO=0.25
# BIAS=false
# USE_XYZ=true
NORM_MODE='anchor'                  # ('nearest_to_mean' 'anchor' 'center')
STD_MODE='BN11'                     # ('BN1D' 'BN11' '1111' 'B111')

DIM_RATIO='2-2-2-1'

NUM_BLOCKS1='1-1-2-1'
TRANSFER_MODE='mlp-mlp-mlp-mlp'
BLOCK1_MODE='mlp-mlp-mlp-mlp'

NUM_BLOCKS2='1-1-2-1'               # '1-1-2-1'
BLOCK2_MODE='mlp-mlp-mlp-mlp'

K_ModelNet='32-32-32-32'
K_ScanObject='24-24-24-24'
SAMPLING_MODE='fps-fps-fps-fps'
SAMPLING_RATIO='2-2-2-2'

CLASSIFIER_MODE='mlp_very_large'    # ('mlp_very_very_large' 'mlp_very_large' 'mlp_large' 'mlp_medium' 'mlp_small' 'mlp_very_small')

BATCH_SIZE_ModelNet=50
EPOCH_ModelNet=300
LEARNING_RATE_ModelNet=0.1
MIN_LR_ModelNet=0.005
WEIGHT_DECAY_ModelNet=2e-4

#SEED=42
WORKERS_ModelNet=6
EMA='yes'

FPS_METHOD='pointops2'  # [pytorch3d, pointops2, pytorch]
KNN_METHOD='pytorch3d'  # [pytorch3d, pytorch]

for i in {1..2}; do
   for EMBED_DIM in "${EMBED_DIM[@]}"; do
        python tasks/cls_modelnet.py --n "$N" --embed "$EMBED_Modelnet" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
        --res_dim_ratio "$RES_DIM_RATIO" --norm_mode "$NORM_MODE" --std_mode "$STD_MODE" --sigma "$SIGMA" \
        --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS2" --transfer_mode "$TRANSFER_MODE" \
        --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$K_ModelNet" \
        --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
        --batch_size "$BATCH_SIZE_ModelNet" --epoch "$EPOCH_ModelNet" --learning_rate "$LEARNING_RATE_ModelNet" \
        --weight_decay "$WEIGHT_DECAY_ModelNet" --workers "$WORKERS_ModelNet" \
         --min_lr "$MIN_LR_ModelNet" --alpha_beta "$ALPHA_BETA" --ema "$EMA" --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"
         # --use_xyz True --bias False --seed "$SEED"
        echo "====================================================================="
   done
done

