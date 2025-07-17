#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

EMBED_DIM=(16 32)
STD_MODE='BN1D'
EPOCH=350
BS=128
TBS=64

FPS_METHOD='pointops2'  # [pytorch3d, pointops2, pytorch]
KNN_METHOD='pytorch3d'  # [pytorch3d, pytorch]

for i in {1..2}; do
    for EMBED_DIM in "${EMBED_DIM[@]}"; do
        python tasks/partseg_shapenet.py --embed_dim "$EMBED_DIM" --std_mode "$STD_MODE" --epochs "$EPOCH" --batch_size "$BS" --test_batch_size "$TBS" \
        --fps_method "$FPS_METHOD" --knn_method "$KNN_METHOD"
    done
done

