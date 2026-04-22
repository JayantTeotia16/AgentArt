"""
phase2/train.py

DAYS 5–11
- Trains the dual encoder with 3-term loss + decoder
- Supports multiple seeds and ablation modes
- All 17 runs are launched via the bash script in parallel

Usage (via bash script):
  python phase2/train.py --seed 0 --mode main
  python phase2/train.py --seed 0 --mode ablation_no_lperp
  python phase2/train.py --seed 0 --mode ablation_no_lcurv
  python phase2/train.py --seed 0 --mode ablation_uniform_cost
  python phase2/train.py --seed 0 --mode ablation_datadriven_cost
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.common import load_config, get_output_dir, setup_logger, save_json, ARTEMIS_EMOTIONS

# ── Model components ──────────────────────────────────────────────────────────

class AffectiveHead(nn.Module):
    """3-layer MLP: CLIP dim -> embed_dim."""
    def __init__(self, input_dim=768, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, embed_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class EmotionDecoder(nn.Module):
    """256 -> 8 softmax, trained with CE against mu_i."""
    def __init__(self, embed_dim=256, n_emotions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, n_emotions)
        )
    def forward(self, z):
        return F.softmax(self.net(z), dim=-1)

# ── Losses ────────────────────────────────────────────────────────────────────

def loss_ot(z_a, z_p, z_n, margin=0.5):
    """Soft triplet loss on affective embeddings."""
    d_ap = 1 - F.cosine_similarity(z_a, z_p)
    d_an = 1 - F.cosine_similarity(z_a, z_n)
    return F.relu(d_ap - d_an + margin).mean()

def loss_perp(z_aff, z_clip):
    """Batch orthogonality penalty between affective and CLIP embeddings."""
    z_aff_n = F.normalize(z_aff, dim=-1)
    z_clip_n = F.normalize(z_clip, dim=-1)
    cross = torch.mm(z_aff_n.T, z_clip_n) / z_aff_n.shape[0]
    return torch.norm(cross, p="fro") ** 2

def loss_curv(z_aff, anchor_idxs, kappa_targets, kappa_targets_tensor, k_neighbours=10):
    """
    Sparse anchor-point curvature regulariser.

    For each anchor in the batch, compute a kNN-based local curvature proxy
    in the current Z_aff space, then regress against pre-computed graph kappa.
    The proxy is the negated mean cosine distance to the k nearest neighbours
    within the batch: denser local neighbourhoods → higher proxy, matching the
    sign convention of Ollivier-Ricci curvature (positive = clustered).

    Both proxy and target are z-normalised within the batch so scales align.
    """
    B = z_aff.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=z_aff.device)

    # z_aff is already the batch of anchor embeddings (batch_size == len(anchor_idxs));
    # anchor_idxs are global painting indices, used only to look up kappa targets.
    dists = 1 - torch.mm(z_aff, z_aff.T)                  # [B, B]
    dists = dists + torch.eye(B, device=z_aff.device) * 1e6  # mask self
    k = min(k_neighbours, B - 1)
    knn_vals, _ = dists.topk(k, dim=-1, largest=False)    # [B, k] smallest distances
    embed_curvature = -knn_vals.mean(dim=-1)              # higher = denser = +curvature

    idx_tensor = torch.as_tensor(anchor_idxs, device=z_aff.device, dtype=torch.long)
    target = kappa_targets_tensor[idx_tensor]

    emb_norm = (embed_curvature - embed_curvature.mean()) / (embed_curvature.std() + 1e-8)
    tgt_norm = (target - target.mean()) / (target.std() + 1e-8)
    return F.mse_loss(emb_norm, tgt_norm)

def loss_decoder(mu_hat, mu_true):
    """Cross-entropy decoder loss against ground-truth emotion distribution."""
    mu_true = mu_true.clamp(min=1e-8)
    mu_true = mu_true / mu_true.sum(dim=-1, keepdim=True)
    return -(mu_true * torch.log(mu_hat.clamp(min=1e-8))).sum(dim=-1).mean()

# ── Dataset ───────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    def __init__(self, triplets, clip_embs, mu_matrix, kappa_targets, preprocess, img_root, mode="main"):
        self.triplets = triplets
        self.clip_embs = clip_embs  # pre-extracted CLIP features
        self.mu_matrix = mu_matrix
        self.kappa_targets = kappa_targets
        self.mode = mode

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return {
            "z_a": torch.tensor(self.clip_embs[a], dtype=torch.float32),
            "z_p": torch.tensor(self.clip_embs[p], dtype=torch.float32),
            "z_n": torch.tensor(self.clip_embs[n], dtype=torch.float32),
            "mu_a": torch.tensor(self.mu_matrix[a], dtype=torch.float32),
            "kappa_a": torch.tensor(self.kappa_targets[a], dtype=torch.float32),
            "anchor_idx": a,
        }

# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg, mode, seed, out_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg["phase2"]["device"] if torch.cuda.is_available() else "cpu")
    phase1_out = Path(cfg["paths"]["output_dir"]) / "phase1"

    log = setup_logger(f"train_{mode}_seed{seed}",
                       log_file=str(out_dir / f"train_{mode}_seed{seed}.log"))
    log.info(f"Training: mode={mode}, seed={seed}, device={device}")

    # Load pre-extracted CLIP features (from step5)
    clip_embs = np.load(phase1_out / "clip_embeddings_subgraph.npy")
    subgraph = pd.read_parquet(phase1_out / "subgraph_with_ricci.parquet")
    mu_matrix = np.array(subgraph["mu"].tolist(), dtype=np.float32)
    kappa_targets = np.load(phase1_out / "node_ricci_targets.npy").astype(np.float32)

    # Choose cost matrix for W2 (affects triplet ranking only for ablations)
    if mode == "ablation_uniform_cost":
        W2 = np.load(phase1_out / "w2_distances_uniform.npy") if \
             (phase1_out / "w2_distances_uniform.npy").exists() else \
             np.load(phase1_out / "w2_distances_subgraph.npy")
    elif mode == "ablation_datadriven_cost":
        W2 = np.load(phase1_out / "w2_distances_datadriven.npy") if \
             (phase1_out / "w2_distances_datadriven.npy").exists() else \
             np.load(phase1_out / "w2_distances_subgraph.npy")
    else:
        W2 = np.load(phase1_out / "w2_distances_subgraph.npy")

    triplets = np.load(phase1_out / "triplets.npy")
    dataset = TripletDataset(triplets, clip_embs, mu_matrix, kappa_targets, None, None, mode)
    loader = DataLoader(dataset, batch_size=cfg["phase2"]["batch_size"],
                        shuffle=True, num_workers=cfg["phase2"]["num_workers"])

    # Model — use pre-extracted CLIP features, so affective head maps from clip_dim
    clip_dim = clip_embs.shape[1]
    head = AffectiveHead(input_dim=clip_dim, embed_dim=cfg["phase2"]["embed_dim"]).to(device)
    decoder = EmotionDecoder(embed_dim=cfg["phase2"]["embed_dim"]).to(device)

    base_lr = cfg["phase2"]["base_lr"]
    decoder_lr = base_lr * cfg["phase2"]["decoder_lr_mult"]
    opt = torch.optim.AdamW([
        {"params": head.parameters(), "lr": base_lr},
        {"params": decoder.parameters(), "lr": decoder_lr},
    ], weight_decay=1e-4)

    lam_perp = cfg["phase2"]["lambda_perp"]
    gamma = cfg["phase2"]["gamma_curv"]
    lcurv_ep20_val = None  # track for adaptive gamma

    kappa_tensor = torch.tensor(kappa_targets, dtype=torch.float32, device=device)
    history = []

    for epoch in range(1, cfg["phase2"]["num_epochs"] + 1):
        head.train(); decoder.train()
        epoch_losses = {"L_OT": [], "L_perp": [], "L_curv": [], "L_dec": [], "total": []}

        for batch in loader:
            z_a_raw = batch["z_a"].to(device)
            z_p_raw = batch["z_p"].to(device)
            z_n_raw = batch["z_n"].to(device)
            mu_a = batch["mu_a"].to(device)
            anchor_idxs = batch["anchor_idx"].tolist()

            # Forward
            z_a = head(z_a_raw)
            z_p = head(z_p_raw)
            z_n = head(z_n_raw)
            mu_hat = decoder(z_a)

            # Losses
            L_OT = loss_ot(z_a, z_p, z_n, margin=cfg["phase2"]["triplet_margin"])
            L_dec = loss_decoder(mu_hat, mu_a)
            total = L_OT + L_dec

            # Curriculum additions
            use_lperp = epoch >= cfg["phase2"]["add_lperp_epoch"]
            use_lcurv = epoch >= cfg["phase2"]["add_lcurv_epoch"]
            if mode == "ablation_no_lperp":
                use_lperp = False
            if mode == "ablation_no_lcurv":
                use_lcurv = False

            L_perp_val = torch.tensor(0.0, device=device)
            L_curv_val = torch.tensor(0.0, device=device)

            if use_lperp:
                # Use raw CLIP features as g_phi (frozen)
                L_perp_val = loss_perp(z_a, F.normalize(z_a_raw, dim=-1)) * lam_perp
                total = total + L_perp_val

            if use_lcurv:
                L_curv_val = loss_curv(z_a, anchor_idxs, kappa_targets, kappa_tensor) * gamma
                total = total + L_curv_val

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()

            epoch_losses["L_OT"].append(L_OT.item())
            epoch_losses["L_perp"].append(L_perp_val.item() if hasattr(L_perp_val, "item") else float(L_perp_val))
            epoch_losses["L_curv"].append(L_curv_val.item() if hasattr(L_curv_val, "item") else float(L_curv_val))
            epoch_losses["L_dec"].append(L_dec.item())
            epoch_losses["total"].append(total.item())

        means = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        history.append({"epoch": epoch, **means, "gamma": gamma})
        log.info(f"Epoch {epoch:3d} | L_OT={means['L_OT']:.4f} L_perp={means['L_perp']:.4f} "
                 f"L_curv={means['L_curv']:.4f} L_dec={means['L_dec']:.4f} gamma={gamma:.5f}")

        # Adaptive gamma: check at epoch 30
        if epoch == cfg["phase2"]["add_lcurv_epoch"]:
            lcurv_ep20_val = means["L_curv"]
        if epoch == 30 and lcurv_ep20_val is not None and lcurv_ep20_val > 0:
            conflict_thresh = cfg["phase2"]["lcurv_conflict_threshold"]
            if means["L_curv"] > conflict_thresh * lcurv_ep20_val:
                gamma = gamma / 2.0
                log.warning(f"  L_curv conflict detected at epoch 30. Halving gamma to {gamma:.6f}")

        # Checkpoint decoder interpolation figures
        if epoch in cfg["phase2"]["checkpoint_epochs"]:
            ckpt_path = out_dir / f"checkpoint_ep{epoch}_{mode}_seed{seed}.pt"
            torch.save({"head": head.state_dict(), "decoder": decoder.state_dict(),
                        "gamma": gamma, "epoch": epoch}, ckpt_path)
            log.info(f"  Checkpoint saved: {ckpt_path}")

    # Save final model and history
    torch.save({"head": head.state_dict(), "decoder": decoder.state_dict(),
                "gamma": gamma, "epoch": cfg["phase2"]["num_epochs"]},
               out_dir / f"model_final_{mode}_seed{seed}.pt")
    save_json(history, out_dir / f"loss_history_{mode}_seed{seed}.json")
    log.info(f"Training complete. Model saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="main",
        choices=["main","ablation_no_lperp","ablation_no_lcurv",
                 "ablation_uniform_cost","ablation_datadriven_cost"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = get_output_dir(cfg, "phase2") / args.mode
    out.mkdir(parents=True, exist_ok=True)
    train(cfg, args.mode, args.seed, out)

if __name__ == "__main__":
    main()
