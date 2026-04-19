#!/usr/bin/env bash
# =============================================================================
# scripts/phase1_step1.sh — DATA ALIGNMENT AND COVERAGE CHECK
# DAY 1 — Run this FIRST before anything else.
# Edit CONFIG and PYTHONPATH below to match your environment.
# =============================================================================
set -euo pipefail

# ── EDIT THESE ────────────────────────────────────────────────
CONFIG="config/config.yaml"          # path to your config file
PYTHONPATH_ROOT="$(pwd)"             # root of the affective_manifold repo
CONDA_ENV=""                         # optional: conda env name (leave blank if not using)
# ─────────────────────────────────────────────────────────────

if [ -n "$CONDA_ENV" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
echo "[$(date)] Starting Phase 1 Step 1: Data Alignment and Coverage Check"
python phase1/step1_align_and_coverage.py
echo "[$(date)] Step 1 complete. Check outputs/phase1/step1_summary.json"
echo "          Verify PATH (A or B) before proceeding to Step 2."
