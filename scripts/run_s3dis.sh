WANDB_ENTITY="WANDB_ENTITY"     # CHANGE WANDB_ENTITY
WANDB_PROJECT="SLNet-T"
LOG_DIR="checkpoints"
DATA_DIR="dataset/s3dis/processed/Stanford3dDataset_v1.2_Aligned_Version"
SCRIPT="tasks/semseg_s3dis.py"

# ─── BEST KNOWN CONFIG (from Phase 1: B2 = weighted_ce + attn + EMA + LS=0.1)
# All Phase 2 runs use this as the base, changing only the listed parameter.
COMMON="--test_area 5
        --num_points 16384
        --batch_size 48
        --epochs 120
        --warmup_epochs 15
        --validate_every 5
        --early_stop_patience 25
        --seed 42
        --num_workers 8
        --data_dir ${DATA_DIR}
        --log_dir ${LOG_DIR}
        --use_wandb True
        --wandb_project ${WANDB_PROJECT}
        --wandb_entity ${WANDB_ENTITY}
        --use_amp True
        --encoder_type attn
        --encoder_dims 64 128 256 512
        --decoder_dims 256 128 64 64
        --k_neighbors 16 16 32 64
        --dropout 0.5
        --weight_decay 2e-3
        --grad_clip 1.0
        --test_stride 0.75
        --batch_size_test 16
        --crop_radius 2.5
        --samples_per_room 75
        --use_ema True
        --ema_decay 0.999
        --optimizer adamw
        --scheduler cosine
        --min_lr 1e-6
        --focal_gamma 2.0
        --class_weights_mode inv_sqrt
        --accum_steps 1
        --rare_room_weight 8.0"

run_exp "B2_loss_weighted_ce" \
    --encoder_type attn \
    --loss_type weighted_ce \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512