"""
phase1/step1_align_and_coverage.py

DAY 1 PRIORITY 0 + 1
- Aligns ArtEmis, APOLO, and your multivalence scores by painting ID
- Checks WikiArt movement metadata coverage
- Determines Path A or Path B
- Validates every join

Run: bash scripts/phase1_step1.sh
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from utils.common import load_config, get_output_dir, setup_logger, save_json, ARTEMIS_EMOTIONS

def run(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    out = get_output_dir(cfg, "phase1")
    log = setup_logger("step1_align", log_file=str(out / "step1_align.log"))

    log.info("=" * 60)
    log.info("STEP 1: DATA ALIGNMENT AND COVERAGE CHECK")
    log.info("=" * 60)

    # ── Load ArtEmis ──────────────────────────────────────────────
    log.info("Loading ArtEmis...")
    artemis = pd.read_csv(cfg["paths"]["artemis_csv"])
    log.info(f"  ArtEmis raw rows: {len(artemis):,}")

    # Normalise painting ID from "painting" column (format: artist/painting_name)
    artemis["painting_id"] = artemis["painting"].str.strip().str.lower().str.replace(" ", "_")
    # Keep only recognised emotions
    artemis = artemis[artemis["emotion"].isin(ARTEMIS_EMOTIONS)].copy()
    log.info(f"  ArtEmis after emotion filter: {len(artemis):,} rows")
    artemis_ids = set(artemis["painting_id"].unique())
    log.info(f"  Unique ArtEmis paintings: {len(artemis_ids):,}")

    # ── Load APOLO ────────────────────────────────────────────────
    log.info("Loading APOLO...")
    apolo_dir = Path(cfg["paths"]["apolo_dir"])
    id_col_candidates = ["painting_id", "image_id", "artwork_id", "id"]
    apolo_files = list(apolo_dir.glob("**/*.csv")) + list(apolo_dir.glob("**/*.json"))
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    if apolo_files:
        apolo_frames = []
        for f in apolo_files:
            try:
                if f.suffix == ".csv":
                    apolo_frames.append(pd.read_csv(f))
                else:
                    apolo_frames.append(pd.read_json(f))
            except Exception as e:
                log.warning(f"  Could not load {f}: {e}")
        apolo = pd.concat(apolo_frames, ignore_index=True)
        log.info(f"  APOLO raw rows: {len(apolo):,}")
        apolo_id_col = next((c for c in id_col_candidates if c in apolo.columns), None)
        if apolo_id_col is None:
            log.error(f"Cannot find ID column in APOLO. Columns: {list(apolo.columns)}")
            sys.exit(1)
        apolo["painting_id"] = apolo[apolo_id_col].astype(str).str.strip().str.lower()
        apolo_ids = set(apolo["painting_id"].unique())
    else:
        # No metadata files — derive IDs from image filenames (style/painting.ext layout)
        log.warning("  No APOLO CSV/JSON found. Deriving painting IDs from image filenames.")
        image_files = [f for f in apolo_dir.glob("**/*") if f.suffix.lower() in IMAGE_EXTS]
        apolo_ids = set()
        for f in image_files:
            # Use relative path without extension as painting_id
            rel = f.relative_to(apolo_dir)
            pid = str(rel.with_suffix("")).strip().lower().replace(" ", "_")
            apolo_ids.add(pid)
        log.info(f"  APOLO images found: {len(apolo_ids):,}")

    log.info(f"  Unique APOLO paintings: {len(apolo_ids):,}")

    # ── Load multivalence scores (optional) ───────────────────────
    mv_csv = cfg["paths"].get("multivalence_csv", "")
    mv = None
    mv_ids = set()
    if mv_csv and Path(mv_csv).exists():
        log.info("Loading multivalence scores...")
        mv = pd.read_csv(mv_csv)
        mv_id_col = next((c for c in id_col_candidates if c in mv.columns), None)
        if mv_id_col is None:
            log.warning(f"Cannot find ID column in multivalence CSV. Skipping.")
            mv = None
        else:
            mv["painting_id"] = mv[mv_id_col].astype(str).str.strip().str.lower()
            mv_ids = set(mv["painting_id"].unique())
            log.info(f"  Unique multivalence paintings: {len(mv_ids):,}")
    else:
        log.warning("  multivalence_csv not found — skipping multivalence scores.")

    # ── Validate joins ────────────────────────────────────────────
    log.info("Validating joins...")
    artemis_apolo = artemis_ids & apolo_ids
    log.info(f"  ArtEmis ∩ APOLO: {len(artemis_apolo):,} paintings")
    if mv_ids:
        all_three = artemis_ids & apolo_ids & mv_ids
        log.info(f"  ArtEmis ∩ APOLO ∩ Multivalence: {len(all_three):,} paintings")

    apolo_miss = apolo_ids - artemis_ids
    miss_rate_apolo = len(apolo_miss) / max(len(apolo_ids), 1)
    log.info(f"  APOLO paintings not in ArtEmis: {len(apolo_miss):,} ({miss_rate_apolo:.2%})")

    MISMATCH_THRESHOLD = 0.001  # 0.1%
    if miss_rate_apolo > MISMATCH_THRESHOLD:
        log.warning(f"  APOLO mismatch rate {miss_rate_apolo:.2%} exceeds {MISMATCH_THRESHOLD:.2%} threshold.")
        log.warning("  Investigate before proceeding. Check painting_id normalisation.")
    miss_rate_mv = 0.0
    if mv_ids:
        mv_miss = mv_ids - artemis_ids
        miss_rate_mv = len(mv_miss) / max(len(mv_ids), 1)
        log.info(f"  Multivalence paintings not in ArtEmis: {len(mv_miss):,} ({miss_rate_mv:.2%})")
        if miss_rate_mv > MISMATCH_THRESHOLD:
            log.warning(f"  Multivalence mismatch rate {miss_rate_mv:.2%} exceeds threshold.")

    # ── WikiArt movement coverage ──────────────────────────────────
    # ArtEmis v0 already contains art_style — use it directly.
    log.info("Checking WikiArt movement coverage from ArtEmis art_style column...")
    if "art_style" in artemis.columns:
        has_style = artemis[artemis["art_style"].notna() & (artemis["art_style"].astype(str).str.strip() != "")]
        wikiart_ids_with_movement = set(has_style["painting_id"].unique())
        artemis_with_movement = artemis_ids & wikiart_ids_with_movement
        coverage_rate = len(artemis_with_movement) / max(len(artemis_ids), 1)
        log.info(f"  WikiArt movement coverage: {len(artemis_with_movement):,} / {len(artemis_ids):,} = {coverage_rate:.2%}")
    else:
        wikiart_ids_with_movement = set()
        coverage_rate = 0.0
        log.warning("  art_style column not found in ArtEmis — defaulting to Path B.")

    threshold = cfg["phase1"]["coverage_threshold"]
    path = "A" if coverage_rate >= threshold else "B"
    log.info(f"  Coverage threshold: {threshold:.0%}")
    log.info(f"  *** PATH DETERMINED: PATH {path} ***")
    if path == "A":
        log.info("  Path A: Full 2x2 factorial design with movement labels.")
    else:
        log.warning("  Path B: Coverage below threshold. Switching to emotion-only stratification.")
        log.warning("  Exp 2 will be redesigned. Disentanglement claim will be weakened.")
        log.warning("  Document coverage rate in paper appendix.")

    # ── Build unified dataframe ───────────────────────────────────
    log.info("Building unified dataframe...")
    # Per-painting emotion distribution from ArtEmis
    grouped = artemis.groupby("painting_id")["emotion"].apply(list).reset_index()
    grouped.columns = ["painting_id", "emotion_list"]

    # Compute mu_i: empirical distribution over 8 emotions
    def compute_mu(emotion_list):
        counts = np.zeros(len(ARTEMIS_EMOTIONS))
        for e in emotion_list:
            if e in ARTEMIS_EMOTIONS:
                counts[ARTEMIS_EMOTIONS.index(e)] += 1
        total = counts.sum()
        return (counts / total).tolist() if total > 0 else counts.tolist()

    grouped["mu"] = grouped["emotion_list"].apply(compute_mu)
    grouped["n_annotations"] = grouped["emotion_list"].apply(len)

    # Compute H(mu_i) — entropy
    def entropy(mu):
        mu_arr = np.array(mu)
        mu_arr = mu_arr[mu_arr > 0]
        return float(-np.sum(mu_arr * np.log(mu_arr)))

    grouped["entropy"] = grouped["mu"].apply(entropy)

    # Attach movement if Path A (use art_style from ArtEmis directly)
    if path == "A":
        style_map = artemis.drop_duplicates("painting_id").set_index("painting_id")["art_style"].to_dict()
        grouped["movement"] = grouped["painting_id"].map(style_map).fillna("unknown")
    else:
        grouped["movement"] = "unknown"

    # Attach multivalence scores (optional)
    if mv is not None:
        mv_map = mv.set_index("painting_id").iloc[:, 0].to_dict()
        grouped["multivalence_score"] = grouped["painting_id"].map(mv_map)
    else:
        grouped["multivalence_score"] = float("nan")

    # Save
    out_path = out / "unified_dataset.parquet"
    grouped.to_parquet(out_path, index=False)
    log.info(f"  Saved unified dataset: {out_path}")

    # Save summary
    summary = {
        "path": path,
        "coverage_rate": coverage_rate,
        "n_artemis_paintings": len(artemis_ids),
        "n_apolo_paintings": len(apolo_ids),
        "n_mv_paintings": len(mv_ids),
        "n_overlap_artemis_apolo": len(artemis_apolo),
        "apolo_mismatch_rate": miss_rate_apolo,
        "mv_mismatch_rate": miss_rate_mv,
        "warnings": []
    }
    if miss_rate_apolo > MISMATCH_THRESHOLD:
        summary["warnings"].append(f"APOLO mismatch {miss_rate_apolo:.4f} > threshold")
    if miss_rate_mv > MISMATCH_THRESHOLD:
        summary["warnings"].append(f"Multivalence mismatch {miss_rate_mv:.4f} > threshold")

    save_json(summary, out / "step1_summary.json")
    log.info(f"Summary saved to {out / 'step1_summary.json'}")
    log.info("STEP 1 COMPLETE. Check step1_summary.json before proceeding.")
    return summary

if __name__ == "__main__":
    run()
