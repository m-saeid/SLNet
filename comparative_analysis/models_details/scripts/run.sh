#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

MODELS=('slnet_embed_dim_16' 'slnet_embed_dim_32' 'pointnet' 'pointnet2_ssg' 'pointnet2_msg' 'dgcnn' 'curvenet' 'pointmlp_elite' 'pointmlp' 'apes_global' 'apes_local')

for MODEL in "${MODELS[@]}"; do
    python tasks/eval_modelnet.py --model "$MODEL"
done

for MODEL in "${MODELS[@]}"; do
    python tasks/eval_scanobject.py --model "$MODEL"
done

for MODEL in "${MODELS[@]}"; do
    python tasks/eval_shapenet.py --model "$MODEL"
done