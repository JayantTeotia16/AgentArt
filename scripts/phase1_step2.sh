#!/usr/bin/env bash
# scripts/phase1_step2.sh — RICCI RUNTIME CALIBRATION
# DAY 1 — Run after Step 1 completes.
set -euo pipefail
CONFIG="config/config.yaml"
PYTHONPATH_ROOT="$(pwd)"
CONDA_ENV=""
if [ -n "$CONDA_ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
echo "[$(date)] Phase 1 Step 2: Ricci Runtime Calibration"
python phase1/step2_ricci_calibration.py
echo "[$(date)] Done. Check step2_ricci_calibration.json for recommended_k."
echo "          If recommended_k differs from config, update config.yaml before Step 5."
