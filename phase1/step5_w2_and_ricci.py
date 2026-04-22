"""
phase1/step5_w2_and_ricci.py

DAYS 2–4
- Pairwise W2 distances on 10K stratified subgraph
- Builds k-NN affective graph
- Computes Ollivier-Ricci curvature (fixed targets for L_curv)
- Computes CLIP edge distance distribution for delta
- Extends approximate W2 to full 80K for triplet mining

Run: bash scripts/phase1_step5.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from pathlib import Path
from tqdm import tqdm
from utils.common import load_config, get_output_dir, setup_logger, save_json, ARTEMIS_EMOTIONS

def sinkhorn_w2(mu1, mu2, C, eps=0.1, max_iter=200):
    """Sinkhorn regularised W2 between two discrete distributions."""
    try:
        import ot
        return float(ot.sinkhorn2(mu1, mu2, C, eps, numItermax=max_iter)[0])
    except ImportError:
        raise ImportError("POT not installed. Run: pip install POT")

def compute_clip_embedding(image_path, clip_model, clip_preprocess, device):
    from PIL import Image
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()
    except Exception:
        return None

def run(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase1")
    log = setup_logger("step5_w2_ricci", log_file=str(out / "step5_w2_ricci.log"))

    log.info("=" * 60)
    log.info("STEP 5: W2 DISTANCES, AFFECTIVE GRAPH, RICCI CURVATURE")
    log.info("=" * 60)

    subgraph_df = pd.read_parquet(out / "subgraph_paintings.parquet")
    C = np.load(out / "cost_matrix_A_russell.npy")
    eps = cfg["phase1"]["sinkhorn_eps"]
    k = cfg["phase1"]["knn_k"]
    n = len(subgraph_df)
    mu_matrix = np.array(subgraph_df["mu"].tolist(), dtype=np.float64)
    # Add small epsilon to avoid zero-probability issues
    mu_matrix = mu_matrix + 1e-8
    mu_matrix = mu_matrix / mu_matrix.sum(axis=1, keepdims=True)

    log.info(f"Computing pairwise W2 on {n:,} paintings (Sinkhorn eps={eps})...")

    w2_path = out / "w2_distances_subgraph.npy"
    if w2_path.exists():
        log.info("  W2 matrix already exists, loading...")
        W2 = np.load(w2_path)
    else:
        import ot
        W2 = np.zeros((n, n), dtype=np.float32)
        # POT's sinkhorn2 accepts a 1-D source vs a (dim, n_hists) stack of targets
        # and returns all n_hists distances in one call — avoids n² Python overhead
        # by vectorising the inner loop (~10–100× speedup on GPU-backed POT, and
        # a big win on CPU too).
        for i in tqdm(range(n - 1), desc="W2 rows"):
            a = mu_matrix[i]                         # (dim,)
            b = mu_matrix[i + 1:].T                  # (dim, n - i - 1)
            row = ot.sinkhorn2(a, b, C, eps, numItermax=200)  # (n - i - 1,)
            row = np.asarray(row, dtype=np.float32)
            W2[i, i + 1:] = row
            W2[i + 1:, i] = row
        np.save(w2_path, W2)
        log.info(f"  W2 matrix saved. Range: [{W2[W2>0].min():.4f}, {W2.max():.4f}]")

    # ── Build k-NN affective graph ──────────────────────────────
    log.info(f"Building k-NN affective graph (k={k})...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric="precomputed", n_jobs=-1)
    nn.fit(W2)
    distances, indices = nn.kneighbors(W2)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, painting_id=subgraph_df.iloc[i]["painting_id"])
        for j_idx, d in zip(indices[i][1:], distances[i][1:]):
            G.add_edge(i, int(j_idx), weight=float(d))

    log.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    with open(out / "affective_graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    # ── Ollivier-Ricci curvature ────────────────────────────────
    log.info("Computing Ollivier-Ricci curvature...")
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError:
        log.error("GraphRicciCurvature not installed.")
        sys.exit(1)

    alpha = cfg["phase1"]["ricci_alpha"]
    orc = OllivierRicci(G, alpha=alpha, verbose="INFO")
    orc.compute_ricci_curvature()

    # Extract per-node mean curvature — these are fixed targets for L_curv
    kappas = nx.get_edge_attributes(orc.G, "ricciCurvature")
    node_kappas = {}
    for (u, v), kappa in kappas.items():
        node_kappas.setdefault(u, []).append(kappa)
        node_kappas.setdefault(v, []).append(kappa)
    mean_kappa = {n_: float(np.mean(vals)) for n_, vals in node_kappas.items()}

    subgraph_df["ricci_curvature"] = subgraph_df.index.map(mean_kappa).fillna(0.0)
    subgraph_df.to_parquet(out / "subgraph_with_ricci.parquet", index=False)
    np.save(out / "node_ricci_targets.npy", subgraph_df["ricci_curvature"].values)
    log.info(f"  Ricci range: [{subgraph_df['ricci_curvature'].min():.4f}, {subgraph_df['ricci_curvature'].max():.4f}]")
    log.info("  node_ricci_targets.npy saved — these are fixed L_curv training targets.")

    # ── CLIP embeddings and delta ────────────────────────────────
    log.info("Computing CLIP embeddings for subgraph paintings (for delta computation)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import clip
        clip_model, clip_preprocess = clip.load(cfg["phase2"]["clip_model"], device=device)
        clip_model.eval()
        clip_embs = []
        img_root = Path(cfg["paths"]["wikiart_images"])
        for _, row in tqdm(subgraph_df.iterrows(), total=n, desc="CLIP encode"):
            pid = row["painting_id"]
            img_path = img_root / (pid + ".jpg")
            if not img_path.exists():
                img_path = img_root / (pid + ".png")
            emb = compute_clip_embedding(str(img_path), clip_model, clip_preprocess, device)
            if emb is None:
                emb = np.zeros(768)
            clip_embs.append(emb)
        clip_embs = np.array(clip_embs, dtype=np.float32)
        np.save(out / "clip_embeddings_subgraph.npy", clip_embs)

        # Compute CLIP cosine distances on all graph edges
        clip_edge_dists = []
        for (u, v) in G.edges():
            cos_sim = float(np.dot(clip_embs[u], clip_embs[v]))
            clip_edge_dists.append(1.0 - cos_sim)
        clip_edge_dists = np.array(clip_edge_dists)

        delta_percentile = cfg["phase1"]["delta_percentile"]
        delta = float(np.percentile(clip_edge_dists, delta_percentile))
        log.info(f"  CLIP edge distance distribution: mean={clip_edge_dists.mean():.4f}, std={clip_edge_dists.std():.4f}")
        log.info(f"  Delta (={delta_percentile}th percentile): {delta:.6f}")
        log.info(f"  Delta written to step5_summary.json — used in Phase 3 agent.")

        save_json({
            "n_subgraph": n, "k": k,
            "ricci_mean": float(subgraph_df["ricci_curvature"].mean()),
            "ricci_std": float(subgraph_df["ricci_curvature"].std()),
            "clip_edge_dist_mean": float(clip_edge_dists.mean()),
            "delta": delta,
            "delta_percentile": delta_percentile,
        }, out / "step5_summary.json")

    except ImportError:
        log.warning("CLIP not available. Skipping CLIP embeddings and delta computation.")
        log.warning("Install with: pip install git+https://github.com/openai/CLIP.git")
        save_json({"n_subgraph": n, "k": k, "delta": None}, out / "step5_summary.json")

    log.info("STEP 5 COMPLETE.")

if __name__ == "__main__":
    run()
