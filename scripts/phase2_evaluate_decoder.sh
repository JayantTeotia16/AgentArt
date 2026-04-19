#!/usr/bin/env bash
# =============================================================================
# scripts/phase2_evaluate_decoder.sh — DECODER EVALUATION ON HELD-OUT APOLO
# DAY 11 — Run after all training converges.
# Evaluates decoder, generates checkpoint interpolation figures.
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

echo "[$(date)] Phase 2: Decoder Evaluation on Held-out APOLO"
python phase2/evaluate_decoder.py
echo "[$(date)] Done."
echo "          Check: outputs/phase2/decoder_evaluation_results.json"
echo "          Figures: outputs/phase2/decoder_interpolation_epoch{10,30,50}.pdf"
echo ""
echo "DECISION POINT: Check decoder W2 error against expected range."
echo "  Expected: mean W2 < $(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['phase4']['expected_ranges']['decoder_w2_max'])")"
echo "  If above threshold, see Days 20-23 buffer decision rules."
