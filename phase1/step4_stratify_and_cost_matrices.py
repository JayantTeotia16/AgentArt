"""
phase1/step4_stratify_and_cost_matrices.py

DAY 2
- Stratified sampling for 10K subgraph (Path A or B)
- JSD k-medoids (NOT Euclidean k-means on simplex)
- Constructs all 3 cost matrices (Russell, Uniform, Data-driven)

Run: bash scripts/phase1_step4.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from utils.common import load_config, get_output_dir, setup_logger, save_json, ARTEMIS_EMOTIONS

def compute_jsd_matrix(mu_matrix):
    """Compute full pairwise JSD distance matrix."""
    n = len(mu_matrix)
    jsd = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = jensenshannon(mu_matrix[i], mu_matrix[j])
            jsd[i, j] = jsd[j, i] = float(d)
    return jsd

def jsd_kmedoids(mu_matrix, k, max_iter=100, random_state=42):
    """JSD k-medoids clustering on the probability simplex."""
    rng = np.random.RandomState(random_state)
    n = len(mu_matrix)
    # Compute pairwise JSD
    D = compute_jsd_matrix(mu_matrix)
    # Initialise medoids randomly
    medoid_idxs = rng.choice(n, size=k, replace=False)
    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iter):
        # Assign each point to nearest medoid
        for i in range(n):
            dists = [D[i, m] for m in medoid_idxs]
            labels[i] = int(np.argmin(dists))

        # Update medoids
        new_medoids = []
        for c in range(k):
            cluster_idxs = np.where(labels == c)[0]
            if len(cluster_idxs) == 0:
                new_medoids.append(medoid_idxs[c])
                continue
            # New medoid: point with min sum of distances to cluster members
            sub_D = D[np.ix_(cluster_idxs, cluster_idxs)]
            best_local = int(np.argmin(sub_D.sum(axis=1)))
            new_medoids.append(cluster_idxs[best_local])

        if new_medoids == list(medoid_idxs):
            break
        medoid_idxs = np.array(new_medoids)

    return labels, medoid_idxs

def build_russell_cost_matrix(cfg):
    """Build 8x8 cost matrix from Russell circumplex coordinates."""
    coords = cfg["phase1"]["emotion_coordinates"]
    C = np.zeros((8, 8))
    for i, e1 in enumerate(ARTEMIS_EMOTIONS):
        for j, e2 in enumerate(ARTEMIS_EMOTIONS):
            v1, a1 = coords[e1]
            v2, a2 = coords[e2]
            C[i, j] = np.sqrt((v1-v2)**2 + (a1-a2)**2)
    # Normalise to [0, 1]
    C = C / C.max()
    return C

def build_uniform_cost_matrix():
    """Uniform cost: all off-diagonal entries equal."""
    C = np.ones((8, 8)) - np.eye(8)
    return C

def run(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase1")
    log = setup_logger("step4_stratify", log_file=str(out / "step4_stratify.log"))

    log.info("=" * 60)
    log.info("STEP 4: STRATIFIED SAMPLING AND COST MATRICES")
    log.info("=" * 60)

    # Load unified dataset and step 1 summary
    df = pd.read_parquet(out / "unified_dataset.parquet")
    summary1 = __import__("json").load(open(out / "step1_summary.json"))
    path = summary1["path"]
    log.info(f"Path determined in Step 1: Path {path}")

    mu_matrix = np.array(df["mu"].tolist())
    target_n = cfg["phase1"]["subgraph_n"]
    k_clusters = cfg["phase1"]["emotion_clusters"]

    # ── Stratified sampling ───────────────────────────────────────
    log.info(f"Clustering {len(df):,} paintings into {k_clusters} emotion clusters (JSD k-medoids)...")
    log.info("This may take 10–30 minutes for large datasets.")

    labels, medoids = jsd_kmedoids(mu_matrix, k=k_clusters)
    df["emotion_cluster"] = labels
    log.info(f"  Emotion clusters: {np.unique(labels, return_counts=True)[1].tolist()}")

    if path == "A":
        log.info("Path A: Stratifying by movement x emotion cluster.")
        movements = df["movement"].unique()
        log.info(f"  Movements found: {len(movements)}")
        strata_col = df["movement"].astype(str) + "_" + df["emotion_cluster"].astype(str)
    else:
        log.info("Path B: Stratifying by emotion cluster only.")
        strata_col = df["emotion_cluster"].astype(str)

    df["stratum"] = strata_col
    strata_counts = df["stratum"].value_counts()
    log.info(f"  Total strata: {len(strata_counts)}")

    # Proportional sampling from each stratum
    sampled_idxs = []
    for stratum, count in strata_counts.items():
        n_sample = max(1, int(np.round(target_n * count / len(df))))
        stratum_df = df[df["stratum"] == stratum]
        n_sample = min(n_sample, len(stratum_df))
        sampled = stratum_df.sample(n=n_sample, random_state=42)
        sampled_idxs.extend(sampled.index.tolist())

    subgraph_df = df.loc[sampled_idxs].copy().reset_index(drop=True)
    log.info(f"  Subgraph size: {len(subgraph_df):,} paintings (target: {target_n:,})")

    subgraph_path = out / "subgraph_paintings.parquet"
    subgraph_df.to_parquet(subgraph_path, index=False)
    log.info(f"  Saved subgraph to {subgraph_path}")

    # Save strata counts for appendix
    strata_counts.to_csv(out / "strata_counts.csv")
    log.info("  Saved strata counts to strata_counts.csv (for paper appendix)")

    # ── Cost matrices ─────────────────────────────────────────────
    log.info("Building cost matrices...")

    # A: Russell circumplex
    C_russell = build_russell_cost_matrix(cfg)
    np.save(out / "cost_matrix_A_russell.npy", C_russell)
    log.info(f"  Matrix A (Russell): saved. Range [{C_russell.min():.3f}, {C_russell.max():.3f}]")

    # B: Uniform
    C_uniform = build_uniform_cost_matrix()
    np.save(out / "cost_matrix_B_uniform.npy", C_uniform)
    log.info(f"  Matrix B (Uniform): saved.")

    # C: Data-driven from ArtEmis co-occurrence
    log.info("  Building data-driven cost matrix from ArtEmis co-occurrence...")
    artemis = pd.read_csv(cfg["paths"]["artemis_csv"])
    artemis["painting_id"] = (
        artemis["artist_name"].str.strip().str.lower().str.replace(" ", "_") + "/" +
        artemis["painting_name"].str.strip().str.lower().str.replace(" ", "_")
    )
    from utils.common import EMOTION_IDX
    cooccur = np.zeros((8, 8))
    for pid, grp in artemis.groupby("painting_id"):
        emotions = [e for e in grp["emotion"].tolist() if e in EMOTION_IDX]
        for e1 in emotions:
            for e2 in emotions:
                if e1 != e2:
                    cooccur[EMOTION_IDX[e1], EMOTION_IDX[e2]] += 1
    # Convert co-occurrence to cost: high co-occurrence = low cost
    total = cooccur.sum()
    if total > 0:
        cooccur_norm = cooccur / total
        # Cost = 1 - P(co-occurrence) (normalised)
        max_cooccur = cooccur_norm.max()
        C_data = 1.0 - (cooccur_norm / max(max_cooccur, 1e-8))
        np.fill_diagonal(C_data, 0.0)
    else:
        C_data = build_uniform_cost_matrix()
        log.warning("  Co-occurrence matrix is empty. Using uniform fallback.")
    np.save(out / "cost_matrix_C_datadriven.npy", C_data)
    log.info(f"  Matrix C (Data-driven): saved.")

    save_json({
        "path": path,
        "subgraph_n": len(subgraph_df),
        "n_strata": len(strata_counts),
        "k_clusters": k_clusters,
        "cost_matrices": ["A_russell", "B_uniform", "C_datadriven"],
    }, out / "step4_summary.json")

    log.info("STEP 4 COMPLETE.")

if __name__ == "__main__":
    run()
