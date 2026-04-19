# Affective Manifold Navigation for Art Emotion
### CVPR Submission — Battle Plan v4 Implementation

This repository contains the complete implementation for the paper:
**"Affective Manifold Navigation: Geodesic Retrieval on the Geometry of Art Emotion"**

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Setup](#setup)
4. [Configuration — The Only File You Need to Edit](#configuration)
5. [Phase 1 — Data Pipeline and Affective Geometry](#phase-1)
6. [Phase 2 — Dual Encoder Training](#phase-2)
7. [Phase 3 — Agentic Navigation System](#phase-3)
8. [Phase 4 — Experiments](#phase-4)
9. [Days 20–23 — Buffer and Re-run Protocol](#buffer)
10. [Phase 5 — Paper Writing](#phase-5)
11. [Key Design Decisions and Loopholes Addressed](#loopholes)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The project has a strict dependency ordering. Steps within some phases can
be parallelised, but phases must execute in order. The critical path is:

```
Phase 1 (Days 1-4) → Phase 2 (Days 5-11) → Phase 4 (Days 12-20)
                   → Phase 3 (Days 5-11, parallel) ↗
                                          → Buffer (Days 20-23)
                                                    → Phase 5 (Days 24-26)
```

**Before running anything:** Edit `config/config.yaml` with your data paths.
That is the only file you need to edit. Everything else is self-contained.

---

## Repository Structure

```
affective_manifold/
├── config/
│   └── config.yaml              ← EDIT THIS FILE WITH YOUR PATHS
├── phase1/
│   ├── step1_align_and_coverage.py   ← Day 1: data alignment + Path A/B
│   ├── step2_ricci_calibration.py    ← Day 1: Ricci runtime calibration
│   ├── step3_sbert_embeddings.py     ← Day 1-2: sentence-BERT (overnight)
│   ├── step4_stratify_and_cost_matrices.py  ← Day 2: sampling + cost matrices
│   ├── step5_w2_and_ricci.py         ← Days 2-4: W2 distances + Ricci
│   └── step6_validation_and_triplets.py     ← Day 4: figures + triplets
├── phase2/
│   ├── train.py                 ← Days 5-11: all training runs
│   └── evaluate_decoder.py     ← Day 11: decoder evaluation
├── phase3/
│   └── agent.py                ← Days 5-11: full navigation agent
├── phase4/
│   └── experiments.py          ← Days 12-20: all 11 experiments
├── utils/
│   └── common.py               ← shared utilities
├── scripts/
│   ├── phase1_step1.sh         ← runner for Step 1
│   ├── phase1_step2.sh         ← runner for Step 2
│   ├── phase1_step3.sh         ← runner for Step 3 (background)
│   ├── phase1_step4.sh         ← runner for Step 4
│   ├── phase1_step5.sh         ← runner for Step 5 (background)
│   ├── phase1_step6.sh         ← runner for Step 6
│   ├── phase2_train_all.sh     ← launches all 17 training runs
│   ├── phase2_evaluate_decoder.sh   ← decoder evaluation
│   ├── phase3_build_agent.sh   ← builds agent + generates test queries
│   └── phase4_experiments.sh   ← runs all 11 experiments
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Install CLIP separately (not on PyPI)
pip install git+https://github.com/openai/CLIP.git

# Install GraphRicciCurvature
pip install GraphRicciCurvature
```

### 2. Set your Anthropic API key (needed for Phase 3 query parser)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Verify data files exist

Before running anything, confirm:
- ArtEmis CSV is at the path specified in `config.yaml`
- APOLO directory is accessible
- WikiArt images directory is accessible
- WikiArt metadata CSV is accessible
- Your multivalence scores CSV is accessible

### 4. Make scripts executable

```bash
chmod +x scripts/*.sh
```

---

## Configuration

**Edit `config/config.yaml` only.** All scripts import from this file.
You should never need to edit a Python file directly.

Key sections to configure:

```yaml
paths:
  artemis_csv:      "/path/to/artemis_dataset_release_v0.csv"
  apolo_dir:        "/path/to/apolo/"
  wikiart_images:   "/path/to/wikiart/images/"
  wikiart_metadata: "/path/to/wikiart_metadata.csv"
  multivalence_csv: "/path/to/your_multivalence_scores.csv"
  output_dir:       "/path/to/outputs/"
```

**Column name assumptions:**
- ArtEmis CSV must have columns: `artist_name`, `painting_name`, `emotion`, `utterance`
  (or `caption` / `explanation` — the code tries all variants)
- WikiArt metadata must have columns: `artist_name`, `painting_name`, and one of
  `style` / `movement` / `genre` / `art_movement`
- Your multivalence CSV must have a painting ID column (one of `painting_id`,
  `image_id`, `artwork_id`, `id`) and a score column

---

## Phase 1 — Data Pipeline and Affective Geometry

### Step 1: Data Alignment and Coverage Check
**DAY 1 — Run this FIRST. Do not proceed until it passes.**

```bash
bash scripts/phase1_step1.sh
```

**What it does:**
- Normalises painting IDs across ArtEmis, APOLO, and your multivalence scores
- Joins all three datasets and validates join quality
- Checks WikiArt movement metadata coverage across ArtEmis paintings
- **Determines Path A or Path B** (the most important decision in the project)
- Builds the unified dataset with per-painting emotion distributions μᵢ

**Output:** `outputs/phase1/step1_summary.json`

**Check:** Open `step1_summary.json`. Verify:
- `mismatch_rate` < 0.001 for both APOLO and multivalence joins
- `path` is either "A" or "B"
- If Path B: read the Path A/B decision tree in the battle plan — Exp 2 must be redesigned

**If mismatch rate > 0.1%:** Do not proceed. Investigate the ID normalisation.
The code normalises by lowercasing and replacing spaces with underscores.
If your datasets use different conventions (e.g., WikiArt uses numeric IDs),
you will need to build a custom mapping table and pass it to the normalisation step.

---

### Step 2: Ricci Runtime Calibration
**DAY 1 — Run in parallel with or immediately after Step 1.**

```bash
bash scripts/phase1_step2.sh
```

**What it does:**
- Runs Ollivier-Ricci on a 1,000 node sample to measure wall-clock time
- Extrapolates to 10,000 nodes and recommends a `k` value
- If projected time > 20 hours, reduces k from 10 to 5

**Output:** `outputs/phase1/step2_ricci_calibration.json`

**Check:** If `recommended_k` differs from the value in `config.yaml`,
update `config.yaml` → `phase1.knn_k` before running Step 5.

---

### Step 3: Sentence-BERT Embeddings (overnight)
**DAY 1-2 — Launch and leave. It runs overnight.**

```bash
bash scripts/phase1_step3.sh
```

**What it does:**
- Embeds every ArtEmis verbal explanation utterance with `all-mpnet-base-v2`
  (this model version is specified in the paper reproducibility section — do not change it)
- Per painting, computes:
  - **Linguistic divergence**: mean pairwise cosine distance across utterances
  - **Mean utterance length**: control variable for partial correlation
  - **Type-token ratio (TTR)**: control variable for partial correlation

**Why these control variables?** The linguistic divergence measure could
be inflated by annotators who write longer or more vocabulary-rich explanations
rather than by genuine affective disagreement. The partial correlation in
Step 6 controls for this confound.

**Output:** `outputs/phase1/utterance_embeddings.npy` + `linguistic_divergence.parquet`

**Monitor:** `tail -f outputs/phase1/step3_nohup.log`

---

### Step 4: Stratified Sampling and Cost Matrices
**DAY 2 — Run after Step 1 completes (can overlap with Step 3).**

```bash
bash scripts/phase1_step4.sh
```

**What it does:**
- Clusters paintings by emotion distribution using **JSD k-medoids**
  (not Euclidean k-means — the probability simplex is not flat)
- Path A: stratified sampling by WikiArt movement × emotion cluster
- Path B: stratified sampling by emotion cluster only
- Builds all three cost matrices:
  - **A (Russell)**: Warriner 2013 norms on Russell valence-arousal plane
  - **B (Uniform)**: all off-diagonal entries equal — ablation baseline
  - **C (Data-driven)**: co-occurrence of emotions in ArtEmis — ablation baseline

**Why JSD k-medoids?** The emotion distributions μᵢ live on the probability
simplex Δ⁷. Euclidean distance is geometrically incorrect there. Jensen-Shannon
divergence is a proper metric on the simplex and gives more meaningful clusters.

**Output:** `outputs/phase1/subgraph_paintings.parquet` + three cost matrix `.npy` files + `strata_counts.csv`

**Note:** `strata_counts.csv` goes in the paper appendix to document the sampling.

---

### Step 5: W2 Distances, Affective Graph, and Ricci Curvature
**DAYS 2-4 — Launch and leave. This is the longest step (8-20 hours).**

```bash
bash scripts/phase1_step5.sh
```

**What it does:**
- Computes pairwise Wasserstein W₂ distances on the 10K stratified subgraph
  using Sinkhorn regularisation (ε=0.1) for tractability
- Builds the k-NN affective graph
- Computes **Ollivier-Ricci curvature** on all graph edges
- Stores per-node mean curvature as `node_ricci_targets.npy` —
  these are the **fixed training targets for the L_curv loss** in Phase 2
- Computes CLIP embeddings for all subgraph paintings
- Computes δ (delta) = 10th percentile of CLIP edge distance distribution
  — this is the semantic anchor threshold used in Phase 3

**Why are the Ricci targets fixed?** Computing Ollivier-Ricci online during
training is intractable (each computation requires solving a set of OT problems).
Instead, we pre-compute the graph curvature once and use it as a sparse
anchor-point regulariser during training.

**Output:**
- `outputs/phase1/w2_distances_subgraph.npy` — full W₂ distance matrix
- `outputs/phase1/affective_graph.gpickle` — the k-NN affective graph
- `outputs/phase1/node_ricci_targets.npy` — fixed L_curv targets
- `outputs/phase1/clip_embeddings_subgraph.npy` — CLIP embeddings
- `outputs/phase1/step5_summary.json` — includes the delta value

**Monitor:** `tail -f outputs/phase1/step5_nohup.log`

---

### Step 6: Validation Figures and Triplet Mining
**DAY 4 — Run after Step 5 completes.**

```bash
bash scripts/phase1_step6.sh
```

**What it does:**

**Validation (Fig 1a + 1b):**
- Fig 1a: Ricci vs H(μᵢ) — **APPENDIX ONLY, LABELLED AS SANITY CHECK**
  This correlation is structurally circular: both quantities are functions
  of the same Wasserstein graph geometry. It is included only as a
  construction-validity check, not as an empirical finding.
- Fig 1b: Three-step analysis for the **main text**:
  1. Partial correlation of Ricci with linguistic divergence,
     controlling for utterance length and TTR
  2. Mediation analysis testing whether H(μᵢ) mediates this correlation
  3. The **residual partial ρ after mediation** is the primary claim —
     this is the variance in Ricci curvature explained by linguistic divergence
     *beyond* what affective entropy alone explains

**Triplet mining:**
- Hard negatives: (anchor, positive: low W₂ + high CLIP-sim,
  negative: high W₂ + high CLIP-sim) — visually similar but emotionally distant
- Path A: 2×2 factorial pairs by (movement, emotion cluster)
- Path B: emotion-cluster distance pairs (close and far)

**Output:** All figures as PDFs + `triplets.npy` + factorial/emotion pair arrays

---

## Phase 2 — Dual Encoder Training

### Launch All 17 Training Runs
**DAY 5 — Edit GPU_IDS in the script first, then launch.**

```bash
# First: edit GPU_IDS in the script
nano scripts/phase2_train_all.sh

bash scripts/phase2_train_all.sh
```

**What it does:**
- Launches all 17 runs simultaneously (5 main seeds + 12 ablation seeds)
- Each run: freezes CLIP ViT-L/14, trains affective head + decoder
- Loss curriculum:
  - Epochs 1-10: L_OT only (triplet loss on W₂ rankings)
  - Epoch 10: adds L_perp (batch orthogonality vs frozen CLIP)
  - Epoch 20: adds L_curv (sparse anchor curvature regulariser)
- **Adaptive gamma**: if L_curv loss at epoch 30 is still > 80% of its
  epoch-20 value, gamma is automatically halved and the adjustment is logged.
  This means L_OT and L_curv may conflict — the adaptive rule handles it.
- Checkpoint figures generated at epochs 10, 30, 50

**Pre-committed robustness criterion for ablations C/D:**
Ablation mean within 0.5 × main model std = ROBUST.
This criterion was written before training ran. Do not change it after seeing results.

**Monitor:** `tail -f outputs/phase2/main/train_main_seed0.log`

**Expected outputs per run:** `model_final_{mode}_seed{seed}.pt` + loss history JSON

### Decoder Evaluation
**DAY 11 — After all training runs converge.**

```bash
bash scripts/phase2_evaluate_decoder.sh
```

**What it does:**
- Evaluates decoder on the held-out APOLO 4,718 paintings
- Reports W₂(μ̂, μ_true) mean ± std across 5 seeds
- Reports per-movement sample sizes and flags n < 30 strata as underpowered
- Generates decoder interpolation figures at epochs 10, 30, 50

**Decision point:** Compare mean decoder W₂ against the expected range in
`config.yaml → phase4.expected_ranges.decoder_w2_max`.
If above threshold: see Days 20-23 buffer.

---

## Phase 3 — Agentic Navigation System

**DAYS 5-11 — Run fully in parallel with Phase 2 training.**

```bash
bash scripts/phase3_build_agent.sh
```

**What it does:**
1. Generates 50 test queries via the LLM query parser (Claude API)
   — converts natural language emotional descriptions to μ* distributions
2. Pre-computes all-pairs shortest paths on the 10K subgraph for O(1) inference
3. Smoke-tests the full navigation loop

**The navigation algorithm (Theorem 1):**
The agent maintains a visited-node set. At each step:
1. Check W₂ convergence — if below threshold, stop
2. Curvature probe — flag current node if κ < τ (affective saddle point)
3. Get neighbours — **exclude all visited nodes** (this is the key step
   that makes Theorem 1 provably correct — no density argument needed)
4. Filter remaining candidates by CLIP drift ≤ δ (lexicographic constraint)
5. Among passing candidates, take step minimising W₂(decode(z), μ*)

**Why is convergence guaranteed?** Step 3 ensures each step visits a
distinct node. The graph has |V| = 10,000 nodes. Therefore the algorithm
terminates in at most |V| steps. No assumptions about filter density required.
This is Theorem 1.

**Delta (δ):** Set automatically to the 10th percentile of the empirical
CLIP edge distance distribution from Phase 1 (`step5_summary.json`).
This makes δ data-determined, not a free hyperparameter.

**Output:**
- `outputs/phase3/test_queries.json` — 50 test queries with parsed μ*
- `outputs/phase3/shortest_paths_cache.pkl` — pre-computed shortest paths

---

## Phase 4 — Experiments

**DAY 12 — Launch after Phases 2 and 3 complete.**

```bash
bash scripts/phase4_experiments.sh
```

**The 11 experiments:**

| # | Name | Figure | Key decision |
|---|------|--------|--------------|
| 1 | Retrieval P@k | Table 1 | Primary result — model vs all baselines |
| 2 | Hard negative separation | Fig 4 | Path A (2×2 factorial) or Path B |
| 3 | Ricci validation | Fig 5 | 3a is APPENDIX (circular); 3b is MAIN TEXT |
| 4 | Trajectory coherence | Fig 6 | Two curves: graph-step (theorem) + decoded (empirical) |
| 5 | Ablation table | Table 2 | Apply pre-committed robustness criterion |
| 6 | Cross-movement retrieval | Fig 7 | Should score higher than CLIP |
| 7 | Decoder interpolation | Fig 2 | Three checkpoints — compare epoch 10/30/50 |
| 8 | Beam search speedup | Supplementary | Empirical speedup vs BFS — no admissibility claimed |
| 9 | Trajectory gallery | Fig 8 | 4-6 most visually striking arcs |
| 10 | UMAP visualisation | Fig 9 | Affective vs CLIP space |
| 11 | Cost matrix sensitivity | Fig 10 | Honest report regardless of outcome |

**Statistical reporting (applies to all experiments):**
- Training variance: mean ± std across 5 seeds (main) or 3 seeds (ablations C/D)
- CI: cluster-robust bootstrap at WikiArt movement level (1,000 iterations)
- Multiple comparisons: BH-FDR correction across all 11 experiments
- Never write "significantly" without a test statistic
- Flag any stratum with n < 30 as insufficiently powered

**BH-FDR correction** is applied automatically. Results saved to
`outputs/phase4/bh_fdr_correction.json` with both uncorrected and
corrected p-values.

---

## Days 20–23 — Buffer and Re-run Protocol

This window is explicitly managed, not slack.

After Phase 4 completes, check: `outputs/phase4/days2023_buffer_check.json`

**Pre-specified expected ranges** (from `config.yaml → phase4.expected_ranges`):

| Metric | Expected | Action if outside |
|--------|----------|-------------------|
| P@5 vs CLIP delta | ≥ 0.05 | Re-run with higher L_OT weight |
| Ricci-ling. divergence ρ | \|ρ\| > 0.15 | Check sentence-BERT model version; weaken claim if still weak |
| Trajectory AUC delta | ≥ 0.10 geodesic > greedy | Check cycle detection; report as non-significant if gap < 0.10 |
| Decoder W₂ error | < 0.20 | Reduce decoder LR; retrain decoder head only |
| Ablation drop | > 1 std | Honest null if not — do not spin |

**Rule:** Any re-run that cannot complete before Day 23 end-of-day is dropped.
The result is reported with a limitation note. Do not compress the writing sprint.

---

## Phase 5 — Paper Writing

**DAYS 24–26 — Three days minimum.**

The code generates all figures and result JSONs. The writing sprint uses them.

**Day 24 morning:** Related work (3 hrs)
**Day 24 afternoon:** Method section — typeset all equations including
Theorem 1 with proof sketch in main text, full proof in appendix (3 sentences)

**Day 25 morning:** Introduction — 5-sentence abstract first, then 1-page intro
**Day 25 afternoon:** Experiments and Results — copy numbers from result JSONs

**Day 26 morning:** Limitations — required explicit statements:
- Exp 3a is structurally circular and appears in appendix only
- Linguistic divergence and Ricci share a causal ancestor (emotional ambiguity);
  we report the residual after mediating for H(μᵢ)
- L_curv and L_OT may conflict; γ is a logged hyperparameter with adaptive rule
- No human perceptual study validates trajectory quality
- Theorem 1 applies to graph-step convergence; decoded distribution monotonicity
  is approximate

**Day 26 afternoon:** Polish and submit.

**Submission checklist:**
- [ ] Every intro claim references a figure or table number
- [ ] Every figure is cited in text
- [ ] BH-FDR correction applied and both p-values reported
- [ ] Theorem 1 numbered and cited
- [ ] All equations numbered
- [ ] γ trajectory in supplementary
- [ ] Per-movement n in all stratified tables
- [ ] sentence-BERT model version (`all-mpnet-base-v2`) in reproducibility section
- [ ] WikiArt coverage rate in appendix
- [ ] Path A or Path B documented

---

## Key Design Decisions and Loopholes Addressed

### Why JSD k-medoids instead of Euclidean k-means?
The probability simplex Δ⁷ is not flat — Euclidean distance is
geometrically incorrect for clustering probability distributions.
Jensen-Shannon divergence is a proper metric on the simplex.

### Why is Exp 3a circular and why is it in the appendix?
The Ollivier-Ricci curvature is computed on a graph built from W₂ distances
over the same emotion distributions that define H(μᵢ). Correlating them is
structurally circular by construction. It is included only as a
construction-validity check. The primary finding is Exp 3b (partial
correlation with linguistic divergence after mediation).

### Why is δ set at the 10th percentile?
This makes δ a data-determined quantity, not a free hyperparameter.
With k=10 in the k-NN graph, at most 1 of 10 neighbours is pruned per step.
This makes Theorem 1 unconditional: convergence is guaranteed in ≤ |V| steps.

### Why is the A* admissibility claim dropped?
Admissibility requires the heuristic to never overestimate the true
cost-to-goal. This would require characterising the decoder's approximation
error precisely. Instead, we report empirical speedup vs BFS as the evidence
for "curvature-guided beam search" — honest and still a contribution.

### Why are ablations C/D run with 3 seeds?
With 1 seed, a result within 1 std of the main model could be noise rather
than genuine robustness. The pre-committed criterion (within 0.5 std =
robust) requires sufficient seeds to make the comparison meaningful.

---

## Troubleshooting

**"GraphRicciCurvature OT subproblem fails"**
Increase `sinkhorn_eps` in `config.yaml` from 0.1 to 0.5.
Higher epsilon = faster but less accurate Sinkhorn. Acceptable for curvature.

**"CLIP import fails"**
Install via: `pip install git+https://github.com/openai/CLIP.git`
Not available on PyPI.

**"Painting IDs don't match across datasets"**
Check the normalisation in step1: IDs are lowercased and spaces replaced
with underscores. If your datasets use numeric IDs or URL-encoded names,
build a custom `painting_id_map.csv` with two columns (`source_id`,
`painting_id`) and load it in step1 before the join.

**"Training diverges after adding L_curv"**
The adaptive gamma rule should catch this, but if it doesn't:
- Manually set `gamma_curv` to 0.01 in `config.yaml` and retrain
- Or delay `add_lcurv_epoch` to 30 instead of 20

**"Decoder W₂ error is too high on APOLO"**
This may indicate overfitting or the decoder learning to fit training
distributions only. Try: (1) lower the decoder learning rate multiplier
from 0.333 to 0.1, (2) add dropout to the decoder MLP.

**"APOLO files not found / wrong format"**
The code tries to load all `.csv` and `.json` files in the APOLO directory.
If APOLO uses a different format, adapt the loading block in `step1`.

**"Phase 3 agent: all neighbours visited before convergence"**
This can happen if the subgraph is too small for the emotional query.
Increase `subgraph_n` in `config.yaml` or reduce `convergence_w2` threshold.

---

*Battle Plan v4 — Four-Round Reviewed — All 26 vulnerabilities closed.*
