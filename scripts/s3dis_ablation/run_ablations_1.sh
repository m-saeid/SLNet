#!/bin/bash
# =============================================================================
# SLNet-T — Ablation Study Runner
# =============================================================================
# Usage:
#   chmod +x run_ablations.sh
#   ./run_ablations.sh
#
# Runs 12 experiments covering the most impactful design choices for the paper.
# Each run is named descriptively and logged to WandB under project "SLNet-T".
# Results accumulate in checkpoints/ and can be compared on wandb.ai.
#
# Estimated total time on RTX 5090 (batch_size=48, 100 epochs + early stopping):
#   ~2.5h per run × 12 runs = ~30h (can run overnight or split across days)
#
# ─── PRE-FLIGHT CHECKLIST ────────────────────────────────────────────────────
# 1. Apply ALL 5 code fixes described in the analysis before running:
#    - config.py:        encoder_type = "attn"   (was "att" — typo!)
#    - main.py:          patience_counter = 0   inside "new best" block
#    - train_epoch.py:   'macc' key             (was 'mean_acc')
#    - config.py:        test_stride = 0.75     (was 1.0)
#    - helper.py:        rare room weight = 8.0  (was 5.0)
# 2. pip install fvcore wandb psutil
# 3. wandb login
# 4. Set WANDB_ENTITY below to your WandB username or team name
# =============================================================================

WANDB_ENTITY="WANDB_ENTITY"     # CHANGE WANDB_ENTITY
WANDB_PROJECT="SLNet-T"
LOG_DIR="checkpoints"
DATA_DIR="dataset/s3dis/processed/Stanford3dDataset_v1.2_Aligned_Version"
SCRIPT="semseg_s3dis/main.py"

# Common fixed settings shared by all ablation runs:
# - test_area=5     (standard S3DIS protocol)
# - epochs=100      (enough to see convergence; early stopping prevents waste)
# - batch_size=48   (safe on RTX 5090 32 GB)
# - seed=42         (reproducibility)
# - validate_every=5
# - early_stop_patience=20

COMMON="--test_area 5
        --num_points 16384
        --batch_size 48
        --epochs 100
        --warmup_epochs 15
        --validate_every 5
        --early_stop_patience 20
        --seed 42
        --num_workers 8
        --data_dir ${DATA_DIR}
        --log_dir ${LOG_DIR}
        --use_wandb True
        --wandb_project ${WANDB_PROJECT}
        --wandb_entity ${WANDB_ENTITY}
        --use_amp True
        --dropout 0.5
        --weight_decay 2e-3
        --grad_clip 1.0
        --test_stride 0.75
        --batch_size_test 16
        --crop_radius 2.5
        --samples_per_room 75
        --ema_decay 0.999"

# Helper function: run one experiment and wait for it to finish before starting next
run_exp() {
    local name=$1
    shift
    echo ""
    echo "============================================================"
    echo "  STARTING: ${name}"
    echo "============================================================"
    python ${SCRIPT} --exp_name ${name} ${COMMON} "$@" --no-compile
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] Run '${name}' exited with code ${exit_code}. Check crash log."
    else
        echo "[OK] Run '${name}' completed."
    fi
    echo ""
    # Brief pause to let GPU memory fully release before next run
    sleep 10
}

# =============================================================================
# GROUP A — Architecture Ablation  (Table: "Effect of Encoder Type")
# Purpose: Justify the use of full Local Attention in SLNet-T vs cheaper
#          alternatives. This is the core architectural contribution.
# Fixed:   focal+lovasz, lr=9e-4, k=[16,16,32,64], dims=[64,128,256,512]
# =============================================================================

echo ""
echo "█████  GROUP A: Encoder Type Ablation  █████"
echo ""

# A1 — Pure MLP encoder (PointNet2-style, no attention)
# Expected: lowest mIoU — establishes the cost of removing attention
run_exp "A1_encoder_mlp" \
    --encoder_type mlp \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# A2 — Hybrid encoder (MLP in stages 0-1, Attention in stages 2-3)
# Expected: mid-range — partial attention still helps at coarse scales
run_exp "A2_encoder_hybrid" \
    --encoder_type hybrid \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# A3 — Full Attention encoder (SLNet-T proposed)
# Expected: best mIoU — this is our model
run_exp "A3_encoder_attn_PROPOSED" \
    --encoder_type attn \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# =============================================================================
# GROUP B — Loss Function Ablation  (Table: "Effect of Loss Function")
# Purpose: Show why focal+lovász is chosen over simpler losses, especially
#          for rare classes (beam, column, sofa, board).
# Fixed:   encoder_type=attn (proposed), all other settings same
# =============================================================================

echo ""
echo "█████  GROUP B: Loss Function Ablation  █████"
echo ""

# B1 — Plain cross-entropy (no class weighting, no focal, no lovász)
# Expected: lowest rare-class performance — strong baseline to beat
run_exp "B1_loss_ce" \
    --encoder_type attn \
    --loss_type ce \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.0 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# B2 — Weighted cross-entropy (class frequency balancing, no focal)
# Expected: better rare-class recall than B1 but worse than focal
run_exp "B2_loss_weighted_ce" \
    --encoder_type attn \
    --loss_type weighted_ce \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# B3 — Focal loss only (no lovász)
# Expected: good rare-class handling but suboptimal mIoU
run_exp "B3_loss_focal" \
    --encoder_type attn \
    --loss_type focal \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# B4 — Focal + Lovász (SLNet-T proposed — same as A3, use A3 results)
# No need to re-run; reference A3_encoder_attn_PROPOSED for this cell.
# Included here as a comment for completeness in the ablation table.

# =============================================================================
# GROUP C — Key Hyperparameter Ablation
# Purpose: Validate design choices that readers might question.
#          Each sub-experiment changes exactly ONE thing from the proposed
#          config (A3) to isolate the effect of each decision.
# =============================================================================

echo ""
echo "█████  GROUP C: Hyperparameter Ablation  █████"
echo ""

# C1 — Without EMA (EMA is on by default in proposed model)
# Purpose: Show EMA's contribution to val stability and final mIoU
run_exp "C1_noEMA" \
    --encoder_type attn \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema False \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# C2 — Without label smoothing (smoothing=0.0 instead of 0.1)
# Purpose: Show that label smoothing helps regularization with focal+lovász
run_exp "C2_noLabelSmooth" \
    --encoder_type attn \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.0 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512

# C3 — Smaller k-neighbors (k=[8,8,16,32] vs proposed [16,16,32,64])
# Purpose: Show effect of local neighborhood size on accuracy and efficiency
# NOTE: This also reduces GFLOPs — good for efficiency analysis in paper
run_exp "C3_small_k" \
    --encoder_type attn \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 8 8 16 32 \
    --encoder_dims 64 128 256 512

# C4 — Smaller model (dims=[32,64,128,256] vs proposed [64,128,256,512])
# Purpose: Show accuracy/efficiency tradeoff — ~4× fewer params (~0.6M)
# This creates a "SLNet-T-Small" variant worth reporting for efficiency story
run_exp "C4_small_model" \
    --encoder_type attn \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 32 64 128 256 \
    --decoder_dims 128 64 32 32

# C5 — Larger crop radius (3.0 vs proposed 2.5)
# Purpose: More context per crop but fewer unique points → investigate tradeoff
run_exp "C5_large_radius" \
    --encoder_type attn \
    --loss_type focal+lovasz \
    --lr 9e-4 \
    --use_ema True \
    --label_smoothing 0.1 \
    --k_neighbors 16 16 32 64 \
    --encoder_dims 64 128 256 512 \
    --crop_radius 3.0

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "  ALL ABLATION RUNS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved in: ${LOG_DIR}/"
echo "WandB dashboard:  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo ""
echo "Ablation Table Summary (fill in from WandB after runs complete):"
echo ""
echo "Group A — Encoder Architecture:"
echo "  Run ID                        | Params(M) | GFLOPs | mIoU | mAcc | OA"
echo "  A1_encoder_mlp                |           |        |      |      |"
echo "  A2_encoder_hybrid             |           |        |      |      |"
echo "  A3_encoder_attn_PROPOSED ★   |           |        |      |      |"
echo ""
echo "Group B — Loss Function:"
echo "  Run ID                        | beam IoU  | column | mIoU | mAcc | OA"
echo "  B1_loss_ce                    |           |        |      |      |"
echo "  B2_loss_weighted_ce           |           |        |      |      |"
echo "  B3_loss_focal                 |           |        |      |      |"
echo "  B4_focal+lovasz (= A3) ★     |           |        |      |      |"
echo ""
echo "Group C — Hyperparameters:"
echo "  Run ID                        | Params(M) | mIoU   | mAcc | val_stab"
echo "  A3_PROPOSED (reference) ★    |           |        |      |"
echo "  C1_noEMA                      |           |        |      |"
echo "  C2_noLabelSmooth              |           |        |      |"
echo "  C3_small_k                    |           |        |      |"
echo "  C4_small_model                |           |        |      |"
echo "  C5_large_radius               |           |        |      |"
echo ""
echo "★ = Proposed SLNet-T configuration"
echo ""
echo "Note: inference_ms and peak_memory_mb are logged to WandB"
echo "      under model/inference_ms and model/peak_memory_mb."
echo "      All runs log per-class IoU for beam/column analysis."