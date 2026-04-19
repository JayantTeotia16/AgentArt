"""
phase2/evaluate_decoder.py

After training converges (Day 11):
- Evaluates decoder on held-out APOLO 4718 paintings
- Generates interpolation figures at checkpoint epochs
- Reports per-movement sample sizes, flags n<30 strata

Run: bash scripts/phase2_evaluate_decoder.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from utils.common import load_config, get_output_dir, setup_logger, save_json, ARTEMIS_EMOTIONS
from phase2.train import AffectiveHead, EmotionDecoder

def sinkhorn_w2_np(mu1, mu2, C, eps=0.05, max_iter=200):
    import ot
    return float(ot.sinkhorn2(mu1 + 1e-8, mu2 + 1e-8, C, eps, numItermax=max_iter)[0])

def run(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase2")
    log = setup_logger("eval_decoder", log_file=str(out / "decoder_evaluation.log"))

    log.info("=" * 60)
    log.info("PHASE 2: DECODER EVALUATION ON HELD-OUT APOLO")
    log.info("=" * 60)

    phase1_out = Path(cfg["paths"]["output_dir"]) / "phase1"
    C = np.load(phase1_out / "cost_matrix_A_russell.npy")
    device = torch.device(cfg["phase2"]["device"] if torch.cuda.is_available() else "cpu")
    embed_dim = cfg["phase2"]["embed_dim"]

    # Load APOLO mu values (should be precomputed from step1)
    apolo_path = phase1_out / "apolo_mu.parquet"
    if not apolo_path.exists():
        log.info("Building APOLO mu from unified dataset...")
        unified = pd.read_parquet(phase1_out / "unified_dataset.parquet")
        apolo_dir = Path(cfg["paths"]["apolo_dir"])
        apolo_files = list(apolo_dir.glob("**/*.csv"))
        apolo = pd.concat([pd.read_csv(f) for f in apolo_files], ignore_index=True)
        id_col = next((c for c in ["painting_id","image_id","artwork_id"] if c in apolo.columns), None)
        apolo["painting_id"] = apolo[id_col].astype(str).str.strip().str.lower()
        apolo_paintings = set(apolo["painting_id"].unique())
        apolo_mu = unified[unified["painting_id"].isin(apolo_paintings)].copy()
        apolo_mu.to_parquet(apolo_path, index=False)
    else:
        apolo_mu = pd.read_parquet(apolo_path)

    log.info(f"APOLO paintings for evaluation: {len(apolo_mu):,}")

    # Load CLIP embeddings for APOLO (if available)
    apolo_clip_path = phase1_out / "clip_embeddings_apolo.npy"
    if not apolo_clip_path.exists():
        log.warning("CLIP embeddings for APOLO not found. Run step5 with full dataset.")
        log.warning("Skipping quantitative decoder evaluation.")
        return

    apolo_clip = np.load(apolo_clip_path).astype(np.float32)
    apolo_mu_matrix = np.array(apolo_mu["mu"].tolist(), dtype=np.float32)

    # Per-movement strata sizes
    min_n = cfg["phase4"]["min_stratum_n"]
    if "movement" in apolo_mu.columns:
        movement_counts = apolo_mu["movement"].value_counts()
        underpowered = movement_counts[movement_counts < min_n]
        if len(underpowered) > 0:
            log.warning(f"  Underpowered strata (n<{min_n}):")
            for mv, cnt in underpowered.items():
                log.warning(f"    {mv}: n={cnt}")

    # Evaluate all 5 seeds
    clip_dim = apolo_clip.shape[1]
    all_w2_errors = []

    for seed in range(cfg["phase2"]["num_seeds"]):
        model_path = out / "main" / f"model_final_main_seed{seed}.pt"
        if not model_path.exists():
            log.warning(f"  Model for seed {seed} not found at {model_path}")
            continue

        head = AffectiveHead(input_dim=clip_dim, embed_dim=embed_dim).to(device)
        decoder_model = EmotionDecoder(embed_dim=embed_dim).to(device)
        ckpt = torch.load(model_path, map_location=device)
        head.load_state_dict(ckpt["head"])
        decoder_model.load_state_dict(ckpt["decoder"])
        head.eval(); decoder_model.eval()

        w2_errors = []
        batch_size = 128
        with torch.no_grad():
            for i in range(0, len(apolo_clip), batch_size):
                batch = torch.tensor(apolo_clip[i:i+batch_size], dtype=torch.float32, device=device)
                z = head(batch)
                mu_hat = decoder_model(z).cpu().numpy()
                for j, (mh, mt) in enumerate(zip(mu_hat, apolo_mu_matrix[i:i+batch_size])):
                    w2_e = sinkhorn_w2_np(mh, mt, C)
                    w2_errors.append(w2_e)

        mean_w2 = float(np.mean(w2_errors))
        log.info(f"  Seed {seed}: mean W2 decoder error = {mean_w2:.4f}")
        all_w2_errors.append(mean_w2)

    if all_w2_errors:
        log.info(f"  Across {len(all_w2_errors)} seeds: {np.mean(all_w2_errors):.4f} ± {np.std(all_w2_errors):.4f}")
        save_json({
            "n_apolo": len(apolo_mu),
            "decoder_w2_mean": float(np.mean(all_w2_errors)),
            "decoder_w2_std": float(np.std(all_w2_errors)),
            "per_seed_w2": all_w2_errors,
            "underpowered_strata": underpowered.to_dict() if "underpowered" in dir() else {},
        }, out / "decoder_evaluation_results.json")

    # ── Interpolation figures at checkpoints ──────────────────────
    log.info("Generating decoder interpolation figures at checkpoints...")
    # Pick 10 painting pairs from subgraph for interpolation
    subgraph = pd.read_parquet(phase1_out / "subgraph_with_ricci.parquet")
    subgraph_clip = np.load(phase1_out / "clip_embeddings_subgraph.npy").astype(np.float32)
    subgraph_mu = np.array(subgraph["mu"].tolist(), dtype=np.float32)

    rng = np.random.RandomState(99)
    pair_idxs = [(rng.randint(0, len(subgraph)), rng.randint(0, len(subgraph))) for _ in range(10)]

    for ep in cfg["phase2"]["checkpoint_epochs"]:
        ckpt_path = out / "main" / f"checkpoint_ep{ep}_main_seed0.pt"
        if not ckpt_path.exists():
            continue
        head = AffectiveHead(input_dim=clip_dim, embed_dim=embed_dim).to(device)
        decoder_model = EmotionDecoder(embed_dim=embed_dim).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        head.load_state_dict(ckpt["head"])
        decoder_model.load_state_dict(ckpt["decoder"])
        head.eval(); decoder_model.eval()

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        ts = [0.0, 0.25, 0.5, 0.75, 1.0]

        for pair_i, (idx_a, idx_b) in enumerate(pair_idxs[:10]):
            with torch.no_grad():
                za = head(torch.tensor(subgraph_clip[idx_a], device=device).unsqueeze(0))
                zb = head(torch.tensor(subgraph_clip[idx_b], device=device).unsqueeze(0))
            decoded_mus = []
            for t in ts:
                z_interp = (1 - t) * za + t * zb
                with torch.no_grad():
                    mu_t = decoder_model(z_interp).cpu().numpy()[0]
                decoded_mus.append(mu_t)

            ax = axes[pair_i]
            decoded_arr = np.array(decoded_mus)
            for em_i, em_name in enumerate(ARTEMIS_EMOTIONS):
                ax.plot(ts, decoded_arr[:, em_i], label=em_name, alpha=0.7)
            ax.set_title(f"Pair {pair_i}", fontsize=9)
            ax.set_xlabel("t")
            ax.set_ylabel("p(emotion)")
            ax.set_ylim(0, 1)

        plt.suptitle(f"Decoder interpolation at epoch {ep}", fontsize=12)
        axes[-1].legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig(out / f"decoder_interpolation_epoch{ep}.pdf", dpi=120)
        plt.close()
        log.info(f"  Saved decoder_interpolation_epoch{ep}.pdf")

    log.info("DECODER EVALUATION COMPLETE.")

if __name__ == "__main__":
    run()
