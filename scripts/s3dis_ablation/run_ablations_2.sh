#!/bin/bash
# =============================================================================
# SLNet-T — Phase 2 Ablation Study Runner
# =============================================================================
# MOTIVATION: Based on Phase 1 results (11 runs), we identified 4 key findings
# that require further investigation before finalizing the paper:
#
#  FINDING 1: weighted_ce BEATS focal+lovasz by +0.049 mIoU (B2 vs A3)
#             → Need to understand WHY and find the optimal loss recipe
#
#  FINDING 2: beam IoU = 0.000 in ALL 11 runs
#             → Need targeted architectural/sampling fixes
#
#  FINDING 3: Best result so far is B2 (0.5816 mIoU) — can we push higher?
#             → Need a final optimized run with the right configuration
#
#  FINDING 4: Small model (C4, 0.618M) is 4× cheaper but only −0.055 mIoU
#             → C4 + weighted_ce may close the gap significantly
#
# DESIGN PRINCIPLE: Each run changes EXACTLY ONE thing from the best config
# (attn + weighted_ce + EMA=true + label_smoothing=0.1 + k=[16,16,32,64])
# to isolate each effect cleanly for the ablation table.
#
# Estimated total time: ~2.5h × 8 runs = ~20h on RTX 5090
# =============================================================================

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

run_exp() {
    local name=$1
    shift
    echo ""
    echo "============================================================"
    echo "  STARTING: ${name}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    python ${SCRIPT} --exp_name ${name} ${COMMON} "$@" --no-compile
    local code=$?
    if [ $code -ne 0 ]; then
        echo "[ERROR] ${name} failed (exit ${code}). Check crash log."
    else
        echo "[OK] ${name} done at $(date '+%H:%M:%S')"
    fi
    sleep 15   # let GPU memory fully release
}

# =============================================================================
# GROUP D — Loss Function Deep-Dive
# Phase 1 showed weighted_ce unexpectedly beats focal+lovász.
# This group isolates exactly WHY and finds the optimal loss recipe.
# Base for all D runs: attn + EMA + LS=0.1 + lr=9e-4
# =============================================================================

echo "█████  GROUP D: Loss Function Deep-Dive  █████"

# ─────────────────────────────────────────────────────────────────────────────
# D1 — weighted_ce + Lovász (no Focal)
# QUESTION: Does adding Lovász to weighted_ce improve or hurt?
#           Phase 1 showed Lovász HURT when combined with Focal (+lovász made
#           focal+lovász 0.0492 worse than plain weighted_ce).
#           Now we test if Lovász helps when paired with weighted_ce instead.
# HYPOTHESIS: weighted_ce provides stable class balancing; Lovász adds mIoU
#             optimization on top → should beat pure weighted_ce (B2=0.5816)
# KEY METRIC TO WATCH: sofa IoU and column IoU vs B2
# ─────────────────────────────────────────────────────────────────────────────
run_exp "D1_weightedCE_plus_lovasz" \
    --loss_type weighted_ce+lovasz \
    --label_smoothing 0.1 \
    --lr 9e-4

# NOTE: If your codebase doesn't have 'weighted_ce+lovasz' as a loss_type,
# add it to build_criterion() in tasks/helper.py:
#
#   elif cfg.loss_type == 'weighted_ce+lovasz':
#       if w is None:
#           cw = _compute_class_weights_inv_sqrt_freq(cfg.data_dir, cfg.test_area)
#           w = torch.tensor(cw, dtype=torch.float32, device=device)
#       wce = nn.CrossEntropyLoss(weight=w, ignore_index=-100,
#                                 label_smoothing=cfg.label_smoothing)
#       lovasz = LovaszSoftmaxLoss(per_class='all', ignore_index=-100)
#       # Simple wrapper:
#       class WCELovasz(nn.Module):
#           def forward(self, logits, targets):
#               return wce(logits, targets) + 0.5 * lovasz(logits, targets)
#       return WCELovasz()


# ─────────────────────────────────────────────────────────────────────────────
# D2 — weighted_ce with median_freq class weights (more aggressive balancing)
# QUESTION: Does more aggressive rare-class weighting (median/freq instead of
#           1/sqrt(freq)) push beam/column IoU above zero?
# Phase 1: beam=0 in ALL runs including B2 which uses inv_sqrt weights.
#           median_freq gives ~5-8x higher weight to beam vs inv_sqrt.
# EXPECTED: May hurt common classes (ceiling/floor) but should help beam.
# ─────────────────────────────────────────────────────────────────────────────
run_exp "D2_weightedCE_medianFreq" \
    --loss_type weighted_ce \
    --class_weights_mode median_freq \
    --label_smoothing 0.1 \
    --lr 9e-4

# ─────────────────────────────────────────────────────────────────────────────
# D3 — Focal loss with higher gamma (γ=3.0 vs γ=2.0 in B3)
# QUESTION: Does harder focusing on misclassified rare points help beam?
#           γ=2 is standard; γ=3 makes the loss ignore well-classified points
#           even more aggressively, forcing the model to focus on beam/column.
# Phase 1 reference: B3 (focal, γ=2.0) → mIoU=0.5584, beam=0.000
# ─────────────────────────────────────────────────────────────────────────────
run_exp "D3_focal_gamma3" \
    --loss_type focal \
    --focal_gamma 3.0 \
    --label_smoothing 0.1 \
    --lr 9e-4

# =============================================================================
# GROUP E — Beam/Rare-Class Targeted Fixes
# The beam problem (IoU=0 across all 11 Phase 1 runs) requires structural fixes
# beyond loss weighting. This group tests targeted solutions.
# Base: weighted_ce (best loss from Phase 1) + attn + EMA
# =============================================================================

echo "█████  GROUP E: Beam/Rare-Class Targeted Fixes  █████"

# ─────────────────────────────────────────────────────────────────────────────
# E1 — Much more aggressive rare-room sampling (weight=20× vs current 8×)
# QUESTION: Is beam still unlearnable with extreme oversampling of rare rooms?
# Phase 1 used rare_room_weight=8. Here we use 20 to force beam exposure.
# IMPLEMENTATION: Change in tasks/helper.py:
#   make_sampler_from_rare_rooms: weights = [20.0 if rid in rare_rooms else 1.0 ...]
# EXPECTED RISK: May hurt common class performance slightly.
# ─────────────────────────────────────────────────────────────────────────────
run_exp "E1_aggressiveRareSampling_w20" \
    --loss_type weighted_ce \
    --class_weights_mode median_freq \
    --label_smoothing 0.1 \
    --lr 9e-4 \
    --rare_room_weight 20.0

# NOTE: Before running E1, temporarily change the rare room weight to 20.0
# in make_sampler_from_rare_rooms() in tasks/helper.py.
# REMEMBER to change it back to 8.0 after this run.

# ─────────────────────────────────────────────────────────────────────────────
# E2 — Larger crop radius specifically for better beam context
# QUESTION: Beam is a thin structural element. A larger sphere (r=4.0) captures
#           more surrounding structure that provides context clues for beam ID.
#           C5 tested r=3.0 and showed no gain. r=4.0 is much more aggressive.
# EXPECTED: May hurt overall mIoU (more ceiling/floor dominates crop) but
#           could specifically help beam through richer geometric context.
# ─────────────────────────────────────────────────────────────────────────────
run_exp "E2_largeRadius_4m" \
    --loss_type weighted_ce \
    --class_weights_mode inv_sqrt \
    --label_smoothing 0.1 \
    --lr 9e-4 \
    --crop_radius 4.0

# =============================================================================
# GROUP F — Final Best Config + Efficiency Variants
# Now that we know weighted_ce is best, these runs finalize the optimal model
# and provide the efficiency story for the paper.
# =============================================================================

echo "█████  GROUP F: Final Best Config & Efficiency Variants  █████"

# ─────────────────────────────────────────────────────────────────────────────
# F1 — FINAL BEST MODEL: weighted_ce + attn + EMA + LS=0.1 (longer training)
# This is the definitive SLNet-T run for reporting in the paper.
# Changes vs B2: epochs=150, early_stop_patience=30, lr slightly tuned.
# B2 peaked at epoch 70 and stopped at 90. We give it 150 epochs + patience=30
# to fully converge. This is THE model whose final mIoU goes in Table 1.
# ─────────────────────────────────────────────────────────────────────────────
run_exp "F1_FINAL_SLNetT" \
    --loss_type weighted_ce \
    --class_weights_mode inv_sqrt \
    --label_smoothing 0.1 \
    --lr 9e-4 \
    --epochs 150 \
    --early_stop_patience 30 \
    --warmup_epochs 20

# ─────────────────────────────────────────────────────────────────────────────
# F2 — SLNet-T-Small with best loss (weighted_ce)
# Phase 1: C4 (small model) got 0.4773 mIoU with focal+lovász.
#           With weighted_ce (which gave +0.049 for full model), small model
#           could reach ~0.52+, making it competitive with A3 (0.5324)
#           at 4× fewer parameters — a remarkable efficiency story.
# PAPER IMPACT: Creates "SLNet-T-Small" variant with params/GFLOPs table.
# ─────────────────────────────────────────────────────────────────────────────
run_exp "F2_SLNetT_Small_bestLoss" \
    --loss_type weighted_ce \
    --class_weights_mode inv_sqrt \
    --label_smoothing 0.1 \
    --lr 9e-4 \
    --encoder_dims 32 64 128 256 \
    --decoder_dims 128 64 32 32

# ─────────────────────────────────────────────────────────────────────────────
# F3 — k=[16,24,32,48] — Intermediate k (between C3 and proposed)
# Phase 1: C3 (k=[8,8,16,32])  → 3.658 GFLOPs, mIoU=0.5017 (−0.031 vs A3)
#          Proposed k=[16,16,32,64] → 6.492 GFLOPs, mIoU=0.5324
# This k=[16,24,32,48] is a middle ground:
#   - More expressive than C3 at fine/mid scales (k=24,32 vs 8,16)
#   - Less memory-hungry than full k=64 at deepest stage
# If F3 ≈ A3 mIoU at lower GFLOPs → better efficiency story for paper table.
# ─────────────────────────────────────────────────────────────────────────────
run_exp "F3_kNeighbors_medium" \
    --loss_type weighted_ce \
    --class_weights_mode inv_sqrt \
    --label_smoothing 0.1 \
    --lr 9e-4 \
    --k_neighbors 16 24 32 48

# =============================================================================
# SUMMARY TABLE (fill in after runs complete via WandB)
# =============================================================================
echo ""
echo "============================================================"
echo "  ALL PHASE 2 RUNS COMPLETE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "WandB: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo ""
echo "Phase 1 Best Reference:"
echo "  B2_weighted_ce  mIoU=0.5816  mAcc=0.6466  OA=0.8511  beam=0.000"
echo ""
echo "Phase 2 Expected Results (fill in from WandB):"
echo ""
echo "Group D — Loss Deep-Dive (base: attn + EMA + LS=0.1):"
echo "  Run                        mIoU   mAcc   OA    beam  sofa  col"
echo "  D1_weightedCE_plus_lovasz  ____   ____   ____  ____  ____  ____"
echo "  D2_weightedCE_medianFreq   ____   ____   ____  ____  ____  ____"
echo "  D3_focal_gamma3            ____   ____   ____  ____  ____  ____"
echo "  [Phase1-B2: ref]           0.5816 0.6466 0.8511 0.000 0.549 0.344"
echo ""
echo "Group E — Beam Fix Attempts:"
echo "  Run                        mIoU   beam  column  Note"
echo "  E1_aggressiveRareSampling  ____   ____  ______  rare_w=20"
echo "  E2_largeRadius_4m          ____   ____  ______  r=4.0m"
echo ""
echo "Group F — Final Models:"
echo "  Run                  mIoU   Params  GFLOPs  Inference  Description"
echo "  F1_FINAL_SLNetT      ____   2.462M  6.492   ~40ms      Paper main result"
echo "  F2_SLNetT_Small      ____   0.618M  1.644   ~40ms      Efficiency variant"
echo "  F3_kNeighbors_medium ____   2.462M  ~4.5    ~40ms      k=[16,24,32,48]"
echo ""
echo "Decision tree after Phase 2:"
echo "  IF F1 mIoU > 0.60 → Strong result, submit to top venue"
echo "  IF beam > 0 in any E run → Use that config as final model"
echo "  IF F2 mIoU > 0.52 → Include SLNet-T-Small in efficiency table"
echo "  IF D1 beats B2 → Change proposed loss to weighted_ce+lovász"
echo ""
echo "Paper Table Structure (draft):"
echo "  Table 1 - Comparison with SOTA:"
echo "    SLNet-T-Small:  0.618M params | ~1.6 GFLOPs | F2 mIoU"
echo "    SLNet-T:        2.462M params | 6.49 GFLOPs | F1 mIoU"
echo "    PointNet:       3.6M          | -            | 41.1"
echo "    PointNet++:     6.0M          | -            | 53.5"
echo "    PointTransV3:   46.1M         | -            | 74.7"
echo ""
echo "  Table 2 - Encoder Ablation (A1/A2/A3)"
echo "  Table 3 - Loss Ablation (B1/B2/B3/A3 + D1/D2)"
echo "  Table 4 - Efficiency Analysis (C3/C4/F2/F3 vs full)"