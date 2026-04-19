#!/usr/bin/env bash
# scripts/phase1_step5.sh — W2 DISTANCES, AFFECTIVE GRAPH, RICCI CURVATURE
# DAYS 2-4 — This is the longest step (~8-20hrs depending on calibration).
# Launch in background.
set -euo pipefail
PYTHONPATH_ROOT="$(pwd)"; CONDA_ENV=""
LOG_FILE="outputs/phase1/step5_nohup.log"
if [ -n "$CONDA_ENV" ]; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
mkdir -p outputs/phase1
echo "[$(date)] Phase 1 Step 5: W2, Affective Graph, Ricci Curvature"
echo "          Log: $LOG_FILE"
nohup python phase1/step5_w2_and_ricci.py > "$LOG_FILE" 2>&1 &
echo "          PID: $!"
echo "          Monitor with: tail -f $LOG_FILE"
echo "          Expected outputs: w2_distances_subgraph.npy, affective_graph.gpickle,"
echo "          node_ricci_targets.npy, clip_embeddings_subgraph.npy, step5_summary.json"
