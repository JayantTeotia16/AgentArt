"""
phase1/step6_validation_and_triplets.py

DAY 4
- Validation figures: Ricci vs H(mu) [appendix], Ricci vs ling divergence [main]
- Mediation analysis (H(mu) as mediator)
- Hard negative triplet mining
- 2x2 factorial pairs (Path A) or emotion-cluster pairs (Path B)

Run: bash scripts/phase1_step6.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from utils.common import load_config, get_output_dir, setup_logger, save_json

def partial_correlation(x, y, controls):
    """Partial correlation of x and y controlling for columns in controls df."""
    import statsmodels.api as sm
    Xc = sm.add_constant(controls)
    res_x = sm.OLS(x, Xc).fit().resid
    res_y = sm.OLS(y, Xc).fit().resid
    rho, pval = spearmanr(res_x, res_y)
    return float(rho), float(pval)

def mediation_analysis(x, m, y):
    """
    Baron-Kenny mediation: does M mediate X -> Y?
    Returns dict with direct, indirect, total effects.
    Bootstrap CI requires pingouin.
    """
    try:
        import pingouin as pg
        results = pg.mediation_analysis(data=pd.DataFrame({"x":x,"m":m,"y":y}),
                                         x="x", m="m", y="y", n_boot=1000, seed=42)
        return results.to_dict("records")
    except ImportError:
        from scipy.stats import pearsonr
        a_coef, _ = pearsonr(x, m)
        b_coef, _ = pearsonr(m, y)
        c_coef, _ = pearsonr(x, y)
        indirect = a_coef * b_coef
        direct = c_coef - indirect
        return [{"path": "indirect", "coef": indirect},
                {"path": "direct", "coef": direct},
                {"path": "total", "coef": c_coef}]

def run(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase1")
    log = setup_logger("step6_validation", log_file=str(out / "step6_validation.log"))

    log.info("=" * 60)
    log.info("STEP 6: VALIDATION FIGURES AND TRIPLET MINING")
    log.info("=" * 60)

    # Load data
    subgraph = pd.read_parquet(out / "subgraph_with_ricci.parquet")
    ling_df = pd.read_parquet(out / "linguistic_divergence.parquet")
    merged = subgraph.merge(ling_df, on="painting_id", how="left")
    merged = merged.dropna(subset=["ricci_curvature", "linguistic_divergence", "entropy"])
    log.info(f"Merged dataset for validation: {len(merged):,} paintings")

    ricci = merged["ricci_curvature"].values
    entropy = merged["entropy"].values
    ling_div = merged["linguistic_divergence"].values
    utt_len = merged["mean_utterance_length"].values
    ttr = merged["mean_ttr"].values

    # ── Fig 1a: SANITY CHECK (appendix) — Ricci vs H(mu) ─────────
    log.info("Fig 1a [APPENDIX]: Ricci vs H(mu_i) — circular sanity check")
    rho_circular, p_circular = spearmanr(ricci, entropy)
    log.info(f"  Spearman rho={rho_circular:.4f}, p={p_circular:.4e}")
    log.warning("  NOTE: This is STRUCTURALLY CIRCULAR by construction. Appendix only.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(entropy, ricci, alpha=0.3, s=8, color="gray")
    ax.set_xlabel("H(μᵢ) — Affective entropy")
    ax.set_ylabel("Ollivier-Ricci curvature κᵢ")
    ax.set_title(f"[APPENDIX — SANITY CHECK ONLY]\nρ={rho_circular:.3f}, p={p_circular:.2e}")
    ax.text(0.05, 0.95, "Structurally circular by construction\nNot an independent finding",
            transform=ax.transAxes, fontsize=8, color="red", va="top")
    plt.tight_layout()
    plt.savefig(out / "fig1a_ricci_vs_entropy_APPENDIX.pdf", dpi=150)
    plt.close()
    log.info("  Saved fig1a_ricci_vs_entropy_APPENDIX.pdf")

    # ── Fig 1b: PRIMARY — Partial correlation + mediation ─────────
    log.info("Fig 1b [MAIN TEXT]: Partial correlation Ricci vs Linguistic Divergence")
    controls = pd.DataFrame({"utt_len": utt_len, "ttr": ttr})
    partial_rho, partial_p = partial_correlation(ricci, ling_div, controls)
    log.info(f"  Partial rho (controlling for length+TTR): {partial_rho:.4f}, p={partial_p:.4e}")

    log.info("  Mediation analysis: does H(mu_i) mediate Ricci -> LingDiv?")
    med_results = mediation_analysis(ricci, entropy, ling_div)
    log.info(f"  Mediation results: {med_results}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(ling_div, ricci, alpha=0.3, s=8, c=entropy, cmap="viridis")
    axes[0].set_xlabel("Linguistic divergence")
    axes[0].set_ylabel("Ollivier-Ricci curvature κᵢ")
    axes[0].set_title(f"Partial ρ={partial_rho:.3f}, p={partial_p:.2e}\n(controlling for utterance length, TTR)")
    axes[1].scatter(entropy, ling_div, alpha=0.3, s=8, color="royalblue")
    axes[1].set_xlabel("H(μᵢ) — Affective entropy (mediator)")
    axes[1].set_ylabel("Linguistic divergence")
    axes[1].set_title("Mediation path: Ricci → H(μ) → Ling. Divergence")
    plt.tight_layout()
    plt.savefig(out / "fig1b_ricci_vs_ling_div_MAIN.pdf", dpi=150)
    plt.close()
    log.info("  Saved fig1b_ricci_vs_ling_div_MAIN.pdf")

    save_json({
        "fig1a_rho_circular": rho_circular, "fig1a_p_circular": p_circular,
        "fig1b_partial_rho": partial_rho, "fig1b_partial_p": partial_p,
        "mediation_results": med_results,
    }, out / "step6_validation_stats.json")

    # ── Triplet mining ────────────────────────────────────────────
    log.info("Mining hard negative triplets...")
    W2 = np.load(out / "w2_distances_subgraph.npy")
    clip_path = out / "clip_embeddings_subgraph.npy"

    if not clip_path.exists():
        log.warning("CLIP embeddings not found. Skipping hard negative mining.")
        log.warning("Run step5 with CLIP available, then re-run step6.")
    else:
        clip_embs = np.load(clip_path)
        clip_sims = clip_embs @ clip_embs.T  # cosine similarity (normalised embs)

        num_triplets = cfg["phase2"]["num_triplets"]
        triplets = []
        rng = np.random.RandomState(42)
        n = len(subgraph)

        log.info(f"  Mining {num_triplets:,} triplets...")
        attempts = 0
        while len(triplets) < num_triplets and attempts < num_triplets * 10:
            attempts += 1
            anchor = rng.randint(0, n)
            # Positive: low W2 AND high CLIP sim
            w2_row = W2[anchor]
            clip_row = clip_sims[anchor]
            pos_score = (1 - w2_row / (w2_row.max() + 1e-8)) * clip_row
            pos_score[anchor] = -1
            positive = int(np.argmax(pos_score))
            # Hard negative: high W2 AND high CLIP sim (visually similar, emotionally distant)
            neg_score = w2_row * clip_row
            neg_score[anchor] = -1
            neg_score[positive] = -1
            negative = int(np.argmax(neg_score))
            if W2[anchor, positive] < W2[anchor, negative]:
                triplets.append((anchor, positive, negative))

        triplets_arr = np.array(triplets, dtype=np.int32)
        np.save(out / "triplets.npy", triplets_arr)
        log.info(f"  Saved {len(triplets_arr):,} triplets to triplets.npy")

    # ── Factorial pairs (Path A) or emotion pairs (Path B) ────────
    step1_summary = __import__("json").load(open(out / "step1_summary.json"))
    path = step1_summary["path"]

    if path == "A":
        log.info("Path A: Mining 2x2 factorial pairs (movement x emotion)...")
        pairs = {"same_move_same_emotion": [], "same_move_diff_emotion": [],
                 "diff_move_same_emotion": [], "diff_move_diff_emotion": []}
        emotion_clusters = subgraph["emotion_cluster"].values
        movements = subgraph["movement"].values

        rng = np.random.RandomState(42)
        n = len(subgraph)
        max_pairs = 50000
        for _ in range(max_pairs * 20):
            if all(len(v) >= max_pairs for v in pairs.values()):
                break
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            same_move = movements[i] == movements[j]
            same_emot = emotion_clusters[i] == emotion_clusters[j]
            key = f"{'same' if same_move else 'diff'}_move_{'same' if same_emot else 'diff'}_emotion"
            if len(pairs[key]) < max_pairs:
                pairs[key].append((i, j))

        for key, pair_list in pairs.items():
            arr = np.array(pair_list, dtype=np.int32)
            np.save(out / f"factorial_pairs_{key}.npy", arr)
            log.info(f"  {key}: {len(arr):,} pairs")
    else:
        log.info("Path B: Mining emotion-cluster distance pairs...")
        # High-emotion-distance vs low-emotion-distance pairs
        emotion_clusters = subgraph["emotion_cluster"].values
        rng = np.random.RandomState(42)
        n = len(subgraph)
        close_pairs, far_pairs = [], []
        for _ in range(200000):
            if len(close_pairs) >= 50000 and len(far_pairs) >= 50000:
                break
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            w2_dist = W2[i, j]
            if w2_dist < np.percentile(W2[W2 > 0], 25) and len(close_pairs) < 50000:
                close_pairs.append((i, j))
            elif w2_dist > np.percentile(W2[W2 > 0], 75) and len(far_pairs) < 50000:
                far_pairs.append((i, j))
        np.save(out / "emotion_pairs_close.npy", np.array(close_pairs, dtype=np.int32))
        np.save(out / "emotion_pairs_far.npy", np.array(far_pairs, dtype=np.int32))
        log.info(f"  Close pairs: {len(close_pairs):,}, Far pairs: {len(far_pairs):,}")

    log.info("STEP 6 COMPLETE.")

if __name__ == "__main__":
    run()
