"""
phase1/step3_sbert_embeddings.py

DAY 1-2 (run overnight)
- Embeds all ArtEmis verbal explanation utterances with sentence-BERT
- Computes per-painting linguistic divergence, utterance length, TTR
- These are the L1/S1 fix: independent variable for Ricci validation

Run: bash scripts/phase1_step3.sh
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils.common import load_config, get_output_dir, setup_logger, save_json

def type_token_ratio(text):
    tokens = str(text).lower().split()
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)

def run(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase1")
    log = setup_logger("step3_sbert", log_file=str(out / "step3_sbert.log"))

    log.info("=" * 60)
    log.info("STEP 3: SENTENCE-BERT EMBEDDINGS AND LINGUISTIC DIVERGENCE")
    log.info("=" * 60)

    model_name = cfg["phase1"]["sbert_model"]  # all-mpnet-base-v2
    log.info(f"Loading sentence-BERT model: {model_name}")
    log.info("NOTE: Model version is specified in paper reproducibility section.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(model_name)
    log.info("Model loaded.")

    # Load ArtEmis
    log.info("Loading ArtEmis utterances...")
    artemis = pd.read_csv(cfg["paths"]["artemis_csv"])

    # Normalise painting ID
    artemis["painting_id"] = (
        artemis["artist_name"].str.strip().str.lower().str.replace(" ", "_") + "/" +
        artemis["painting_name"].str.strip().str.lower().str.replace(" ", "_")
    )

    utterance_col = next((c for c in ["utterance", "caption", "explanation", "text"] if c in artemis.columns), None)
    if utterance_col is None:
        log.error(f"Cannot find utterance column. Columns: {list(artemis.columns)}")
        sys.exit(1)

    artemis = artemis[artemis[utterance_col].notna()].copy()
    log.info(f"  {len(artemis):,} rows with utterances across {artemis['painting_id'].nunique():,} paintings")

    # Per-utterance features
    artemis["utterance_length"] = artemis[utterance_col].apply(lambda x: len(str(x).split()))
    artemis["ttr"] = artemis[utterance_col].apply(type_token_ratio)

    # Embed all utterances in batches
    log.info("Embedding utterances (this will take several hours)...")
    utterances = artemis[utterance_col].tolist()
    batch_size = 256
    embeddings = []

    for i in tqdm(range(0, len(utterances), batch_size), desc="Embedding"):
        batch = utterances[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype(np.float32)
    log.info(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings with painting IDs
    artemis = artemis.reset_index(drop=True)
    artemis["emb_idx"] = np.arange(len(artemis))
    np.save(out / "utterance_embeddings.npy", embeddings)
    artemis[["painting_id", "emb_idx", "utterance_length", "ttr", utterance_col]].to_parquet(
        out / "utterance_metadata.parquet", index=False
    )
    log.info(f"  Saved embeddings and metadata.")

    # Compute per-painting linguistic divergence, mean length, mean TTR
    log.info("Computing per-painting linguistic divergence...")
    results = []
    painting_groups = artemis.groupby("painting_id")

    for painting_id, group in tqdm(painting_groups, desc="Per-painting divergence"):
        idxs = group["emb_idx"].values
        if len(idxs) < 2:
            ling_div = 0.0
        else:
            embs = embeddings[idxs]
            # Mean pairwise cosine distance (embeddings are normalised → cosine sim = dot product)
            sims = embs @ embs.T
            np.fill_diagonal(sims, np.nan)
            mean_sim = np.nanmean(sims)
            ling_div = float(1.0 - mean_sim)  # convert similarity to distance

        results.append({
            "painting_id": painting_id,
            "linguistic_divergence": ling_div,
            "mean_utterance_length": float(group["utterance_length"].mean()),
            "mean_ttr": float(group["ttr"].mean()),
            "n_utterances": len(idxs),
        })

    ling_df = pd.DataFrame(results)
    ling_path = out / "linguistic_divergence.parquet"
    ling_df.to_parquet(ling_path, index=False)
    log.info(f"  Saved linguistic divergence to {ling_path}")
    log.info(f"  Divergence stats: mean={ling_df['linguistic_divergence'].mean():.4f}, "
             f"std={ling_df['linguistic_divergence'].std():.4f}")

    save_json({
        "n_paintings": len(ling_df),
        "n_utterances": len(artemis),
        "mean_linguistic_divergence": float(ling_df["linguistic_divergence"].mean()),
        "std_linguistic_divergence": float(ling_df["linguistic_divergence"].std()),
        "sbert_model": model_name,
    }, out / "step3_summary.json")

    log.info("STEP 3 COMPLETE.")

if __name__ == "__main__":
    run()
