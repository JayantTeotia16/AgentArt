"""
phase1/step2_ricci_calibration.py

DAY 1 PRIORITY 2
- Runs Ollivier-Ricci on a 1K node subsample
- Measures wall-clock time and extrapolates to 10K
- Outputs recommended k and core budget

Run: bash scripts/phase1_step2.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from utils.common import load_config, get_output_dir, setup_logger, save_json

def build_knn_graph(embeddings, k):
    """Build k-NN graph from embedding matrix using cosine distance."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", n_jobs=-1)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    G = nx.Graph()
    n = len(embeddings)
    for i in range(n):
        G.add_node(i)
        for j_idx, d in zip(indices[i][1:], distances[i][1:]):
            G.add_edge(i, int(j_idx), weight=float(d))
    return G

def run(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase1")
    log = setup_logger("step2_ricci_cal", log_file=str(out / "step2_ricci_calibration.log"))

    log.info("=" * 60)
    log.info("STEP 2: RICCI RUNTIME CALIBRATION")
    log.info("=" * 60)

    # Load unified dataset (from step 1)
    ds_path = out / "unified_dataset.parquet"
    if not ds_path.exists():
        log.error("unified_dataset.parquet not found. Run step1 first.")
        sys.exit(1)
    df = pd.read_parquet(ds_path)
    log.info(f"Loaded {len(df):,} paintings from unified dataset.")

    # Sample 1K for calibration
    sample_n = min(1000, len(df))
    sample = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    mu_matrix = np.array(sample["mu"].tolist())
    log.info(f"Calibration sample: {sample_n} paintings")

    k = cfg["phase1"]["knn_k"]
    log.info(f"Building k-NN graph with k={k}...")
    G_sample = build_knn_graph(mu_matrix, k)
    log.info(f"  Graph: {G_sample.number_of_nodes()} nodes, {G_sample.number_of_edges()} edges")

    # Run Ricci curvature
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError:
        log.error("GraphRicciCurvature not installed. Run: pip install GraphRicciCurvature")
        sys.exit(1)

    log.info("Running Ollivier-Ricci on 1K subsample...")
    max_cores = cfg["phase1"]["ricci_max_cores"]
    alpha = cfg["phase1"]["ricci_alpha"]

    t0 = time.time()
    orc = OllivierRicci(G_sample, alpha=alpha, verbose="ERROR")
    orc.compute_ricci_curvature()
    elapsed_1k = time.time() - t0
    log.info(f"  1K Ricci wall-clock: {elapsed_1k:.1f}s")

    # Extrapolate: Ricci scales roughly as O(n * k) OT subproblems
    target_n = cfg["phase1"]["subgraph_n"]
    # Empirical scaling: roughly quadratic in n due to neighbourhood dependencies
    scale_factor = (target_n / sample_n) ** 1.5
    projected_s = elapsed_1k * scale_factor
    projected_h = projected_s / 3600
    log.info(f"  Projected time for {target_n:,} nodes: {projected_h:.1f} hrs")

    # Recommendation
    MAX_HOURS = 20
    recommended_k = k
    if projected_h > MAX_HOURS:
        recommended_k = max(3, k // 2)
        log.warning(f"  Projected time exceeds {MAX_HOURS}h ceiling.")
        log.warning(f"  Recommendation: reduce k from {k} to {recommended_k}")
        log.warning(f"  Re-run calibration with k={recommended_k} if concerned.")
    else:
        log.info(f"  Projected time within budget. Proceed with k={k}.")

    # Save curvature values from calibration sample
    kappas = nx.get_edge_attributes(orc.G, "ricciCurvature")
    node_kappas = {}
    for (u, v), kappa in kappas.items():
        node_kappas.setdefault(u, []).append(kappa)
        node_kappas.setdefault(v, []).append(kappa)
    mean_kappas = {n: float(np.mean(vals)) for n, vals in node_kappas.items()}
    log.info(f"  Curvature range: [{min(mean_kappas.values()):.3f}, {max(mean_kappas.values()):.3f}]")

    result = {
        "sample_n": sample_n,
        "elapsed_1k_seconds": elapsed_1k,
        "projected_seconds_10k": projected_s,
        "projected_hours_10k": projected_h,
        "recommended_k": recommended_k,
        "within_budget": projected_h <= MAX_HOURS,
        "max_hours_threshold": MAX_HOURS,
    }
    save_json(result, out / "step2_ricci_calibration.json")
    log.info(f"Calibration results saved to step2_ricci_calibration.json")
    log.info(f"Use recommended_k={recommended_k} in config.yaml if needed.")
    log.info("STEP 2 COMPLETE.")
    return result

if __name__ == "__main__":
    run()
