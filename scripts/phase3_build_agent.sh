#!/usr/bin/env bash
# =============================================================================
# scripts/phase3_build_agent.sh — BUILD AND SMOKE-TEST THE NAVIGATION AGENT
# DAYS 5-11 (parallel with Phase 2 training)
# Builds the agent, generates 50 test queries, caches shortest paths.
# =============================================================================
set -euo pipefail

PYTHONPATH_ROOT="$(pwd)"
CONDA_ENV=""
CONFIG="config/config.yaml"

# ── ANTHROPIC API KEY (OPTIONAL) ─────────────────────────────
# Only needed if query_parser = "anthropic" in config.yaml.
# Default is "local" (sentence-BERT, no API key needed).
# export ANTHROPIC_API_KEY="sk-ant-..."
# ────────────────────────────────────────────────────────────

if [ -n "$CONDA_ENV" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$PYTHONPATH_ROOT:$PYTHONPATH"
mkdir -p outputs/phase3

echo "[$(date)] Phase 3: Generating 50 test queries via LLM parser..."
python - <<'PYEOF'
import sys, os, json
sys.path.insert(0, os.getcwd())
from utils.common import load_config, get_output_dir, ARTEMIS_EMOTIONS
from phase3.agent import build_agent

cfg = load_config()
out = get_output_dir(cfg, "phase3")

# 50 hand-crafted emotional queries spanning the full space
TEST_QUERIES = [
    "a vast melancholy landscape with a faint glimmer of hope on the horizon",
    "the sublime terror of standing at the edge of something infinite",
    "quiet contentment of an afternoon that asks nothing of you",
    "the uncanny unease of a familiar place made strange",
    "raw exhilaration of speed and motion and becoming",
    "grief that has aged into something almost beautiful",
    "the disgust and fascination of something grotesque yet compelling",
    "joy so pure it trembles on the edge of grief",
    "the cold rage of injustice held with rigid composure",
    "awe before a force that dwarfs human understanding",
    "tender sadness of something ending too quietly to notice",
    "the electric tension before something momentous",
    "dread wrapped in extraordinary beauty",
    "the satisfaction of order and geometry and rightness",
    "nostalgia so sharp it is almost pain",
    "wonder mixed with a low hum of existential fear",
    "the lightness of pure play and delight",
    "sorrow for something that was lost before it could be named",
    "the serenity at the absolute limit of endurance",
    "violent beauty — destruction that overwhelms with its scale",
    "the wry amusement of recognising your own absurdity",
    "deep unease beneath a placid surface",
    "the specific ache of things almost remembered",
    "exaltation that makes the body feel too small to contain it",
    "the hollow aftermath of strong emotion",
    "fear edged with curiosity, moving toward the unknown anyway",
    "the domesticity of small comforts against vast indifference",
    "anger transmuted into cold precision",
    "the vertigo of beauty that is also threatening",
    "quiet wonder at the ordinary made suddenly visible",
    "ambivalence: love and loss so intertwined they cannot be separated",
    "the peculiar melancholy of crowds: alone together",
    "a heaviness that is also a kind of peace",
    "the violence of colour and light demanding your attention",
    "reverence bordering on terror",
    "the low-grade dread of something inevitable approaching",
    "amusement at the gap between human aspiration and human reality",
    "the clarity that comes only after catastrophe",
    "tenderness so fragile it cannot be touched directly",
    "defiance compressed into stillness",
    "the texture of time passing without incident",
    "wonder that does not resolve, that prefers to remain open",
    "sadness for a future that will not arrive",
    "the specific exhaustion of sustained hope",
    "disgust at one's own complicity",
    "the coolness of distance — not indifference, but perspective",
    "excitement that contains within it the seed of its own ending",
    "grief that is also gratitude",
    "the sensation of being watched by something old and indifferent",
    "absolute stillness that is not emptiness but fullness",
]

agent, graph = build_agent()
queries_with_mu = []

for i, q in enumerate(TEST_QUERIES):
    try:
        mu_star = agent.parse_query(q)
        entry = {"id": i, "query": q,
                 "mu_star": dict(zip(ARTEMIS_EMOTIONS, mu_star.tolist()))}
        queries_with_mu.append(entry)
        print(f"  Query {i:2d}: {q[:50]}... -> dominant={ARTEMIS_EMOTIONS[mu_star.argmax()]}")
    except Exception as e:
        print(f"  Query {i}: ERROR: {e}")
        queries_with_mu.append({"id": i, "query": q, "mu_star": None, "error": str(e)})

out_path = out / "test_queries.json"
json.dump(queries_with_mu, open(out_path, "w"), indent=2)
print(f"\nSaved {len(queries_with_mu)} test queries to {out_path}")
PYEOF

echo "[$(date)] Test queries generated."

echo "[$(date)] Pre-computing all-pairs shortest paths on affective graph (for O(1) inference)..."
python - <<'PYEOF'
import sys, os, json, pickle
sys.path.insert(0, os.getcwd())
from phase3.agent import build_agent
from utils.common import get_output_dir, load_config

cfg = load_config()
out = get_output_dir(cfg, "phase3")
agent, graph = build_agent()

print("  Computing all-pairs shortest paths...")
sp = graph.precompute_shortest_paths()
sp_path = out / "shortest_paths_cache.pkl"
with open(sp_path, "wb") as f:
    pickle.dump(sp, f)
print(f"  Saved shortest paths cache to {sp_path}")
PYEOF

echo "[$(date)] Phase 3 agent build complete."
echo "          Outputs: outputs/phase3/test_queries.json"
echo "                   outputs/phase3/shortest_paths_cache.pkl"
