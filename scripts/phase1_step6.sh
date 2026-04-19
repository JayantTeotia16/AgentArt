#!/usr/bin/env bash
# scripts/phase1_step6.sh — VALIDATION FIGURES AND TRIPLET MINING
# DAY 4 — Run after Step 5 completes.
set -euo pipefail
PYTHONPATH_ROOT="$(pwd)"; CONDA_ENV=""
if [ -n "$CONDA_ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
echo "[$(date)] Phase 1 Step 6: Validation Figures and Triplet Mining"
python phase1/step6_validation_and_triplets.py
echo "[$(date)] Done. Figures saved to outputs/phase1/fig1a and fig1b."
echo "          IMPORTANT: Fig 1a is appendix only (circular). Fig 1b is the main finding."
