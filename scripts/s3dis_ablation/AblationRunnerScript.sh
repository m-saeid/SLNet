#!/usr/bin/env bash
# ablation/run_ablations.sh
# Usage: bash run_ablations.sh [--dry-run] [--resume]
set -euo pipefail

DRY_RUN=0
RESUME=0
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
    [[ "$arg" == "--resume" ]] && RESUME=1
done

PYTHON="python"
SCRIPT="tasks/semseg_s3dis.py"
LOG_ROOT="ablation_results"
RESULTS_CSV="${LOG_ROOT}/results.csv"
mkdir -p "$LOG_ROOT"

# Write CSV header if not exists
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "exp_name,encoder_type,k_neighbors,num_points,focal_gamma,gmp,dropout,params_M,best_miou,best_oa,train_time_h,val_time_min,gpu_mem_gb,status" > "$RESULTS_CSV"
fi

run_exp() {
    local EXP_NAME="$1"; shift
    local EXTRA_ARGS="$@"
    local CKPT_DIR="${LOG_ROOT}/${EXP_NAME}"
    local DONE_FLAG="${CKPT_DIR}/.done"
    local LOG_FILE="${CKPT_DIR}/run.log"

    if [[ "$RESUME" == "1" && -f "$DONE_FLAG" ]]; then
        echo "[SKIP] ${EXP_NAME} — already completed"
        return 0
    fi

    mkdir -p "$CKPT_DIR"
    echo "[RUN] ${EXP_NAME}"
    echo "  Args: $EXTRA_ARGS"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] would run: $PYTHON $SCRIPT --exp_name $EXP_NAME --log_dir $LOG_ROOT $EXTRA_ARGS"
        return 0
    fi

    set +e
    $PYTHON $SCRIPT \
        --exp_name "$EXP_NAME" \
        --log_dir "$LOG_ROOT" \
        --seed 42 \
        --epochs 100 \
        --validate_every 5 \
        $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
        touch "$DONE_FLAG"
        echo "[OK] ${EXP_NAME}"
        # Extract best mIoU from log and append to CSV
        BEST_MIOU=$(grep "New best mIoU" "$LOG_FILE" | tail -1 | grep -oP 'mIoU=\K[0-9.]+' || echo "N/A")
        echo "${EXP_NAME},,,,,,,,$BEST_MIOU,,,,, OK" >> "$RESULTS_CSV"
    else
        echo "[FAIL] ${EXP_NAME} — exit code $EXIT_CODE"
        echo "${EXP_NAME},,,,,,,, N/A,,,,, FAIL" >> "$RESULTS_CSV"
    fi
}

# --k_neighbors 16 16 32 64 \  #16 16 24 24 \

# ─── Baseline ────────────────────────────────────────────────────────────────
run_exp "baseline_hybrid_k16" \
    --encoder_type hybrid \
    --k_neighbors 16 16 32 64 \
    --num_points 16384 \
    --focal_gamma 2.0 \
    --dropout 0.3

# ─── Ablation 1: Encoder type ─────────────────────────────────────────────────
run_exp "abl_all_mlp" \
    --encoder_type mlp \
    --k_neighbors 16 16 32 64 \
    --num_points 16384 \
    --focal_gamma 2.0 \
    --dropout 0.3

run_exp "abl_all_attn" \
    --encoder_type attn \
    --k_neighbors 16 16 32 64 \
    --num_points 16384 \
    --focal_gamma 2.0 \
    --dropout 0.3

# ─── Ablation 2: k neighbors ──────────────────────────────────────────────────
for K in 8 16 32; do
    run_exp "abl_k${K}" \
        --encoder_type hybrid \
        --k_neighbors $K $K $K $K \
        --num_points 16384 \
        --focal_gamma 2.0 \
        --dropout 0.3
done

# ─── Ablation 3: num_points ───────────────────────────────────────────────────
for NP in 8192 16384 32768; do
    run_exp "abl_np${NP}" \
        --encoder_type hybrid \
        --k_neighbors 16 16 32 64 \
        --num_points $NP \
        --focal_gamma 2.0 \
        --dropout 0.3
done

# ─── Ablation 4: Focal gamma ──────────────────────────────────────────────────
for G in 1.0 2.0 3.0; do
    run_exp "abl_gamma${G}" \
        --encoder_type hybrid \
        --k_neighbors 16 16 32 64 \
        --num_points 16384 \
        --focal_gamma $G \
        --dropout 0.3
done

# ─── Ablation 5: Dropout ──────────────────────────────────────────────────────
for D in 0.2 0.3 0.5; do
    run_exp "abl_drop${D}" \
        --encoder_type hybrid \
        --k_neighbors 16 16 32 64 \
        --num_points 16384 \
        --focal_gamma 2.0 \
        --dropout $D
done

# ─── Ablation 6: GMP fusion ───────────────────────────────────────────────────
run_exp "abl_with_gmp" \
    --encoder_type hybrid \
    --k_neighbors 16 16 32 64 \
    --num_points 16384 \
    --focal_gamma 2.0 \
    --dropout 0.3 \
    --use_gmp

echo "All experiments completed. Results: $RESULTS_CSV"