#!/usr/bin/env bash
# scripts/phase1_step4.sh — STRATIFIED SAMPLING AND COST MATRICES
# DAY 2 — Run after Step 1 and Step 3 (or in parallel with Step 3).
set -euo pipefail
PYTHONPATH_ROOT="$(pwd)"; CONDA_ENV=""
if [ -n "$CONDA_ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
echo "[$(date)] Phase 1 Step 4: Stratified Sampling and Cost Matrices"
python phase1/step4_stratify_and_cost_matrices.py
echo "[$(date)] Done. subgraph_paintings.parquet and cost matrices saved."
