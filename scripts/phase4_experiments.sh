#!/usr/bin/env bash
# =============================================================================
# scripts/phase4_experiments.sh — ALL 11 EXPERIMENTS
# DAY 12 — Launch all experiments simultaneously.
# Each experiment saves its own results JSON and figures.
# The Days 20-23 buffer check runs automatically at the end.
# =============================================================================
set -euo pipefail

PYTHONPATH_ROOT="$(pwd)"
CONDA_ENV=""
CONFIG="config/config.yaml"

if [ -n "$CONDA_ENV" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
mkdir -p outputs/phase4

echo "[$(date)] Phase 4: Launching all 11 experiments..."
echo "          Results will be saved to outputs/phase4/"

# Run all experiments (each is a function in experiments.py)
# They can run in parallel if you have multiple machines.
# For single machine: run sequentially (still fast individually).

# Option 1: Sequential (safe, recommended for single machine)
python phase4/experiments.py

# Option 2: Parallel (uncomment if multiple machines available)
# python -c "
# import sys; sys.path.insert(0,'.')
# from phase4.experiments import *
# from utils.common import load_config, get_output_dir
# import logging, numpy as np
# from pathlib import Path
# cfg = load_config()
# out = get_output_dir(cfg, 'phase4')
# log = logging.getLogger('parallel')
# C = np.load(Path(cfg['paths']['output_dir']) / 'phase1' / 'cost_matrix_A_russell.npy')
# p1 = Path(cfg['paths']['output_dir']) / 'phase1'
# p2 = Path(cfg['paths']['output_dir']) / 'phase2'
# exp1_retrieval(cfg, out, log, p1, p2, C)
# " &
# python -c "..." &
# wait

echo ""
echo "[$(date)] All experiments complete."
echo ""
echo "Results files:"
ls -la outputs/phase4/*.json 2>/dev/null || echo "  (no JSON results found yet)"
echo ""
echo "DAYS 20-23 BUFFER: Review outputs/phase4/days2023_buffer_check.json"
echo "  ANOMALY = re-run or limitation note required."
echo "  No anomalies = proceed to writing sprint on Day 24."
