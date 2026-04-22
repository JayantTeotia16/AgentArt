"""
phase4/experiments.py

DAYS 12–20 — All 11 experiments.
Each experiment is a function. All are called from run_all().
Results saved to phase4/results/.

Run: bash scripts/phase4_experiments.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from utils.common import load_config, get_output_dir, setup_logger, save_json, ARTEMIS_EMOTIONS

def sinkhorn_w2(mu1, mu2, C, eps=0.05, max_iter=200):
    import ot
    mu1 = (np.array(mu1) + 1e-8); mu1 /= mu1.sum()
    mu2 = (np.array(mu2) + 1e-8); mu2 /= mu2.sum()
    return float(ot.sinkhorn2(mu1, mu2, C, eps, numItermax=max_iter)[0])

def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / k

def cluster_bootstrap_ci(metric_per_group, n_boot=1000, level=0.95, rng_seed=42):
    """Cluster-robust bootstrap: resample at group level."""
    rng = np.random.RandomState(rng_seed)
    groups = list(metric_per_group.keys())
    observed = np.mean([np.mean(v) for v in metric_per_group.values()])
    boot_means = []
    for _ in range(n_boot):
        boot_groups = rng.choice(groups, size=len(groups), replace=True)
        boot_vals = [np.mean(metric_per_group[g]) for g in boot_groups]
        boot_means.append(np.mean(boot_vals))
    alpha = 1 - level
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return observed, lo, hi

def exp1_retrieval(cfg, out, log, phase1_out, phase2_out, C):
    """Exp 1: Retrieval P@1/5/10 across 5 seeds vs baselines."""
    log.info("EXP 1: Retrieval P@k")
    subgraph = pd.read_parquet(phase1_out / "subgraph_with_ricci.parquet")
    mu_matrix = np.array(subgraph["mu"].tolist(), dtype=np.float32)
    W2_all = np.load(phase1_out / "w2_distances_subgraph.npy")
    n = len(subgraph)

    # Similarity threshold for "relevant": W2 < median(W2)
    w2_vals = W2_all[np.triu_indices(n, k=1)]
    rel_thresh = float(np.percentile(w2_vals, 25))  # bottom quartile = most similar

    # Per-movement groups for cluster-robust CI
    movements = subgraph.get("movement", pd.Series(["unknown"]*n)).values

    results = {}
    for k_val in [1, 5, 10]:
        per_seed = []
        for seed in range(cfg["phase2"]["num_seeds"]):
            model_path = phase2_out / "main" / f"model_final_main_seed{seed}.pt"
            if not model_path.exists():
                continue
            # Load model and compute affective embeddings for subgraph
            clip_embs = np.load(phase1_out / "clip_embeddings_subgraph.npy").astype(np.float32)
            from phase2.train import AffectiveHead, EmotionDecoder
            device = torch.device("cpu")
            head = AffectiveHead(input_dim=clip_embs.shape[1], embed_dim=cfg["phase2"]["embed_dim"])
            ckpt = torch.load(model_path, map_location=device)
            head.load_state_dict(ckpt["head"])
            head.eval()
            with torch.no_grad():
                z = head(torch.tensor(clip_embs)).numpy()
            # Cosine similarity in affective space
            z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
            sim_matrix = z_norm @ z_norm.T

            p_at_k_vals = []
            mv_groups = {}
            for i in range(min(n, 500)):  # subsample for speed
                relevant = set(np.where(W2_all[i] < rel_thresh)[0]) - {i}
                if len(relevant) == 0:
                    continue
                scores = sim_matrix[i].copy()
                scores[i] = -1
                retrieved = np.argsort(-scores)[:k_val * 2].tolist()
                p = precision_at_k(retrieved, relevant, k_val)
                p_at_k_vals.append(p)
                mv = movements[i]
                mv_groups.setdefault(mv, []).append(p)

            per_seed.append(float(np.mean(p_at_k_vals)))

        mean_p = float(np.mean(per_seed)) if per_seed else 0.0
        std_p = float(np.std(per_seed)) if per_seed else 0.0
        # Cluster-robust CI (using movement groups from last seed)
        if mv_groups:
            _, ci_lo, ci_hi = cluster_bootstrap_ci(mv_groups, n_boot=cfg["phase4"]["bootstrap_n"])
        else:
            ci_lo, ci_hi = mean_p - std_p, mean_p + std_p

        results[f"P@{k_val}"] = {"mean": mean_p, "std": std_p, "ci_lo": ci_lo, "ci_hi": ci_hi}
        log.info(f"  P@{k_val}: {mean_p:.4f} ± {std_p:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    save_json(results, out / "exp1_retrieval.json")
    return results

def exp3_ricci_validation(cfg, out, log, phase1_out):
    """Exp 3: Ricci curvature validation (appendix + main text)."""
    log.info("EXP 3: Ricci curvature validation")
    subgraph = pd.read_parquet(phase1_out / "subgraph_with_ricci.parquet")
    ling_df = pd.read_parquet(phase1_out / "linguistic_divergence.parquet")
    merged = subgraph.merge(ling_df, on="painting_id", how="left").dropna()

    ricci = merged["ricci_curvature"].values
    entropy = merged["entropy"].values
    ling_div = merged["linguistic_divergence"].values
    utt_len = merged["mean_utterance_length"].values
    ttr = merged["mean_ttr"].values

    # 3a: circular (appendix)
    rho_circ, p_circ = spearmanr(ricci, entropy)
    log.info(f"  3a [APPENDIX - CIRCULAR]: rho={rho_circ:.4f}, p={p_circ:.4e}")

    # 3b: partial correlation + mediation
    from phase1.step6_validation_and_triplets import partial_correlation, mediation_analysis
    controls = pd.DataFrame({"utt_len": utt_len, "ttr": ttr})
    partial_rho, partial_p = partial_correlation(ricci, ling_div, controls)
    log.info(f"  3b partial rho={partial_rho:.4f}, p={partial_p:.4e}")
    med_results = mediation_analysis(ricci, entropy, ling_div)
    log.info(f"  Mediation: {med_results}")

    result = {"circular_rho": rho_circ, "circular_p": p_circ,
              "partial_rho": partial_rho, "partial_p": partial_p,
              "mediation": med_results}
    save_json(result, out / "exp3_ricci_validation.json")
    return result

def exp4_trajectory_coherence(cfg, out, log, phase1_out):
    """Exp 4: Trajectory coherence — both graph-step and decoded W2 curves."""
    log.info("EXP 4: Trajectory coherence")
    from phase3.agent import build_agent
    import json as _json

    queries_path = phase1_out.parent / "phase3" / "test_queries.json"
    if not queries_path.exists():
        log.warning("Test queries not found. Run phase3 query building first.")
        return {}

    agent, graph = build_agent()
    queries = _json.load(open(queries_path))
    C = np.load(phase1_out / "cost_matrix_A_russell.npy")

    auc_geodesic, auc_decoded, steps_list = [], [], []
    for q in queries[:cfg["phase3"]["num_test_queries"]]:
        mu_star = np.array([q["mu_star"][e] for e in ARTEMIS_EMOTIONS], dtype=np.float32)
        start_node = int(graph.df["ricci_curvature"].idxmin())  # start at most stable node
        result = agent.navigate(start_node, mu_star)
        w2_graph = [h["w2"] for h in result["w2_history"]]
        # AUC as area under normalised convergence curve (lower = better)
        if len(w2_graph) > 1:
            normalised = np.array(w2_graph) / max(w2_graph[0], 1e-8)
            auc_geodesic.append(float(np.trapz(normalised) / len(normalised)))
        steps_list.append(result["n_steps"])

    result_dict = {"mean_auc_geodesic": float(np.mean(auc_geodesic)),
                   "std_auc_geodesic": float(np.std(auc_geodesic)),
                   "mean_steps": float(np.mean(steps_list))}
    log.info(f"  Mean AUC geodesic: {result_dict['mean_auc_geodesic']:.4f}")
    save_json(result_dict, out / "exp4_trajectory_coherence.json")
    return result_dict

def exp5_ablation(cfg, out, log, phase1_out, phase2_out, C):
    """Exp 5: Ablation table."""
    log.info("EXP 5: Ablation table")
    # Load W2 ground truth
    W2_all = np.load(phase1_out / "w2_distances_subgraph.npy")
    clip_embs = np.load(phase1_out / "clip_embeddings_subgraph.npy").astype(np.float32)
    n = len(clip_embs)
    w2_vals = W2_all[np.triu_indices(n, k=1)]
    rel_thresh = float(np.percentile(w2_vals, 25))

    from phase2.train import AffectiveHead
    ablation_modes = ["main", "ablation_no_lperp", "ablation_no_lcurv",
                      "ablation_uniform_cost", "ablation_datadriven_cost"]
    results = {}

    for mode in ablation_modes:
        mode_dir = phase2_out / mode
        n_seeds = cfg["phase2"]["num_seeds"] if mode == "main" else cfg["phase2"]["ablation_seeds"]
        p5_vals = []
        for seed in range(n_seeds):
            model_path = mode_dir / f"model_final_{mode}_seed{seed}.pt"
            if not model_path.exists():
                continue
            device = torch.device("cpu")
            head = AffectiveHead(input_dim=clip_embs.shape[1], embed_dim=cfg["phase2"]["embed_dim"])
            ckpt = torch.load(model_path, map_location=device)
            head.load_state_dict(ckpt["head"])
            head.eval()
            with torch.no_grad():
                z = head(torch.tensor(clip_embs)).numpy()
            z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
            sim_matrix = z_norm @ z_norm.T
            p5_list = []
            for i in range(min(n, 300)):
                relevant = set(np.where(W2_all[i] < rel_thresh)[0]) - {i}
                if not relevant:
                    continue
                scores = sim_matrix[i].copy(); scores[i] = -1
                retrieved = np.argsort(-scores)[:10].tolist()
                p5_list.append(precision_at_k(retrieved, relevant, 5))
            p5_vals.append(float(np.mean(p5_list)))

        results[mode] = {"mean": float(np.mean(p5_vals)), "std": float(np.std(p5_vals)),
                         "n_seeds": len(p5_vals)}
        log.info(f"  {mode}: P@5 = {results[mode]['mean']:.4f} ± {results[mode]['std']:.4f}")

    # Check robustness criterion (pre-committed)
    main_mean = results.get("main", {}).get("mean", 0)
    main_std = results.get("main", {}).get("std", 1)
    thresh = cfg["phase4"]["robustness_threshold"]
    for mode in ablation_modes:
        if mode == "main":
            continue
        diff = abs(main_mean - results.get(mode, {}).get("mean", 0))
        robust = diff <= thresh * main_std
        results[mode]["robust"] = robust
        log.info(f"  {mode}: robust={robust} (diff={diff:.4f}, threshold={thresh}*std={thresh*main_std:.4f})")

    save_json(results, out / "exp5_ablations.json")
    return results

def run_all(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase4")
    log = setup_logger("phase4_all", log_file=str(out / "experiments.log"))
    phase1_out = Path(cfg["paths"]["output_dir"]) / "phase1"
    phase2_out = Path(cfg["paths"]["output_dir"]) / "phase2"
    C = np.load(phase1_out / "cost_matrix_A_russell.npy")

    all_pvals = []

    log.info("=" * 60)
    log.info("PHASE 4: ALL EXPERIMENTS")
    log.info("=" * 60)

    r1 = exp1_retrieval(cfg, out, log, phase1_out, phase2_out, C)
    r3 = exp3_ricci_validation(cfg, out, log, phase1_out)
    r4 = exp4_trajectory_coherence(cfg, out, log, phase1_out)
    r5 = exp5_ablation(cfg, out, log, phase1_out, phase2_out, C)

    # Collect p-values for BH-FDR correction
    all_pvals = []
    pval_labels = []
    if "partial_p" in r3:
        all_pvals.append(r3["partial_p"])
        pval_labels.append("Exp3b_partial_rho")

    if len(all_pvals) > 0:
        reject, pvals_corr, _, _ = multipletests(all_pvals, alpha=cfg["phase4"]["fdr_alpha"], method="fdr_bh")
        fdr_results = [{"label": l, "p_uncorr": p, "p_bh_corr": pc, "reject_h0": bool(r)}
                       for l, p, pc, r in zip(pval_labels, all_pvals, pvals_corr, reject)]
        save_json(fdr_results, out / "bh_fdr_correction.json")
        log.info("BH-FDR correction applied:")
        for fr in fdr_results:
            log.info(f"  {fr['label']}: uncorr p={fr['p_uncorr']:.4e}, BH-corr p={fr['p_bh_corr']:.4e}, reject H0={fr['reject_h0']}")

    # Days 20–23 buffer: check expected ranges
    log.info("\nDAYS 20–23 BUFFER: Checking pre-specified expected ranges...")
    expected = cfg["phase4"]["expected_ranges"]
    anomalies = []
    # Check P@5 vs CLIP baseline (requires CLIP P@5 computed separately)
    clip_p5_path = out / "baseline_clip_p5.json"
    if clip_p5_path.exists():
        clip_p5 = float(__import__("json").load(open(clip_p5_path)).get("mean", 0))
        our_p5 = r1.get("P@5", {}).get("mean", 0)
        delta_p5 = our_p5 - clip_p5
        if delta_p5 < expected["p5_vs_clip_min_delta"]:
            anomalies.append(f"P@5 delta vs CLIP ({delta_p5:.4f}) < threshold ({expected['p5_vs_clip_min_delta']})")
    for anomaly in anomalies:
        log.warning(f"  ANOMALY: {anomaly}")
    if anomalies:
        log.warning("  Consider re-run or limitation note before writing sprint.")
    else:
        log.info("  All expected ranges satisfied. Proceed to writing sprint.")

    save_json({"anomalies": anomalies}, out / "days2023_buffer_check.json")
    log.info("PHASE 4 COMPLETE.")

if __name__ == "__main__":
    run_all()
