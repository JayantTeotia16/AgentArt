#!/usr/bin/env bash
# scripts/phase1_step3.sh — SENTENCE-BERT EMBEDDINGS (overnight)
# DAY 1-2 — Launch in background, it runs overnight.
# Uses all-mpnet-base-v2 as specified in the paper.
set -euo pipefail
CONFIG="config/config.yaml"
PYTHONPATH_ROOT="$(pwd)"
CONDA_ENV=""
LOG_FILE="outputs/phase1/step3_nohup.log"
if [ -n "$CONDA_ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
mkdir -p outputs/phase1
echo "[$(date)] Phase 1 Step 3: Sentence-BERT Embeddings (running in background)"
echo "          Log: $LOG_FILE"
nohup python phase1/step3_sbert_embeddings.py > "$LOG_FILE" 2>&1 &
echo "          PID: $!"
echo "          Monitor with: tail -f $LOG_FILE"
