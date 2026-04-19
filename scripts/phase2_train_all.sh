#!/usr/bin/env bash
# =============================================================================
# scripts/phase2_train_all.sh — LAUNCH ALL 17 TRAINING RUNS IN PARALLEL
# DAY 5 — Run this once. All 17 runs fire simultaneously.
# Requires 17 GPU slots. Each run gets one GPU via CUDA_VISIBLE_DEVICES.
#
# EDIT: GPU_IDS — list of GPU device IDs available on your machine.
#       If using a cluster, replace the local launch block with your
#       SLURM/PBS submission commands.
# =============================================================================
set -euo pipefail

PYTHONPATH_ROOT="$(pwd)"
CONDA_ENV=""
CONFIG="config/config.yaml"

# ── EDIT: available GPU IDs ──────────────────────────────────
# Example: 8 GPUs, IDs 0-7. Runs share GPUs if fewer than 17.
GPU_IDS=(0 1 2 3 4 5 6 7)
# ────────────────────────────────────────────────────────────

if [ -n "$CONDA_ENV" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
mkdir -p outputs/phase2/main
mkdir -p outputs/phase2/ablation_no_lperp
mkdir -p outputs/phase2/ablation_no_lcurv
mkdir -p outputs/phase2/ablation_uniform_cost
mkdir -p outputs/phase2/ablation_datadriven_cost

n_gpus=${#GPU_IDS[@]}
run_idx=0

launch() {
    local mode=$1
    local seed=$2
    local gpu=${GPU_IDS[$((run_idx % n_gpus))]}
    local log="outputs/phase2/${mode}/train_${mode}_seed${seed}.log"
    echo "[$(date)] Launching: mode=$mode seed=$seed GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup python phase2/train.py \
        --config "$CONFIG" --seed "$seed" --mode "$mode" \
        > "$log" 2>&1 &
    echo "          PID=$! log=$log"
    run_idx=$((run_idx + 1))
}

# ── Main runs: 5 seeds ──────────────────────────────────────
for seed in 0 1 2 3 4; do
    launch "main" $seed
done

# ── Ablation A: no L_perp, 3 seeds ─────────────────────────
for seed in 0 1 2; do
    launch "ablation_no_lperp" $seed
done

# ── Ablation B: no L_curv, 3 seeds ─────────────────────────
for seed in 0 1 2; do
    launch "ablation_no_lcurv" $seed
done

# ── Ablation C: uniform cost matrix, 3 seeds ───────────────
for seed in 0 1 2; do
    launch "ablation_uniform_cost" $seed
done

# ── Ablation D: data-driven cost matrix, 3 seeds ───────────
for seed in 0 1 2; do
    launch "ablation_datadriven_cost" $seed
done

echo ""
echo "[$(date)] All 17 training runs launched."
echo "          Monitor logs in outputs/phase2/*/"
echo "          Total runs: 5 main + 3 + 3 + 3 + 3 ablation = 17"
echo ""
echo "IMPORTANT: Pre-committed robustness criterion for ablations C/D:"
echo "  Ablation mean within 0.5 * main_std = ROBUST"
echo "  This criterion was written before training ran. Do not change it."

wait
echo "[$(date)] All training runs completed."
