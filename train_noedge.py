import argparse
import copy
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# from trainer_noedge import (
#     N_PCA_FEATURES, set_seed, build_geo_dict_kdtree,
#     load_or_build_dataset, HybridGNNClassifier, evaluate_loader,
# )

from trainer_noedge_spatial import (
    N_PCA_FEATURES, set_seed, build_geo_dict_kdtree,
    load_or_build_dataset, HybridGNNClassifier, evaluate_loader,
)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t).pow(self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def train(args):
    set_seed(42)

    if not os.path.exists(args.geo):
        print(f"ERROR: geometry file not found: {args.geo}", file=sys.stderr)
        sys.exit(1)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)
    print(f"Geometry: {len(geo)} DOMs, KDTree built")

    dataset = load_or_build_dataset(
        cache_dir                 = args.cache_dir,
        nue_dbs                   = args.nue_dbs,
        tau_dbs                   = args.tau_dbs,
        kd_tree                   = kd_tree,
        string_ids_geo            = string_ids_geo,
        max_events_per_class      = args.max_events,
        charge_threshold          = args.charge_threshold,
    )

    indices = np.arange(len(dataset))
    labels  = np.array([dataset[i].y.item() for i in range(len(dataset))])

    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_frac, stratify=labels, random_state=42
    )

    train_ds = dataset[train_idx.tolist()]
    val_ds   = dataset[val_idx.tolist()]
    print(f"Split data:  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridGNNClassifier(gnn_hidden_dim=64, pca_dim=N_PCA_FEATURES, fusion_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=args.lr_min)

    n_pos      = max(int((labels[train_idx] == 1).sum()), 1)
    n_neg      = max(int((labels[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # n_pos      = max(int((labels[train_idx] == 1).sum()), 1)
    # n_neg      = max(int((labels[train_idx] == 0).sum()), 1)
    # pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    # criterion  = FocalLoss(gamma=3.0, pos_weight=None)

    best_val_auc   = -np.inf
    best_state     = None
    best_epoch     = -1
    patience       = args.patience
    epochs_no_gain = 0
    os.makedirs(args.save_dir, exist_ok=True)

    train_losses = []
    val_losses   = []
    val_aucs     = []
    lrs          = [] 

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1:03d}", unit="batch", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            logits  = model(batch)
            targets = batch.y.view(-1)
            loss    = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)

        val_loss, val_auc, _, _ = evaluate_loader(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc if not np.isnan(val_auc) else 0.0)
        lrs.append(current_lr)

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1

        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | " f"AUC: {val_auc_str} | LR: {current_lr:.2e} | Best: {best_val_auc:.4f}")

        if epochs_no_gain >= patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "hybrid_weights.pt"))
    joblib.dump(dataset.pca_scaler, os.path.join(args.save_dir, "pca_scaler.joblib"))

    print(f"\nBest val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
    ax.plot(train_losses, color="cyan", label="Train")
    ax.plot(val_losses, color="orange", label="Val")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.set_title("Training Curves", color="white")
    ax.legend(facecolor="black", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curves.png"), dpi=150, facecolor="black")
    plt.close()
    fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="black")
    ax2.plot(lrs, color="lime")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Learning Rate", color="white")
    ax2.set_yscale("log")
    ax2.set_title("Learning Rate Schedule", color="white")
    ax2.set_facecolor("black")
    ax2.tick_params(colors="white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "lr_curve.png"), dpi=150, facecolor="black")
    plt.close()
    print(f"Saved model, scaler, and plot to {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training for PCA+GNN Tau vs Electron Neutrino Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tau_dbs", nargs="+", required=True, help="Tau DB paths")
    parser.add_argument("--nue_dbs", nargs="+", required=True, help="Nue DB paths")
    parser.add_argument("--geo", default="/path/to/geometry_clean.csv", help="Geometry file")
    parser.add_argument("--max_events", type=int, default=None, help="Max events per class")
    parser.add_argument("--charge_threshold", type=float, default=0.1, help="Charge threshold")
    parser.add_argument("--save_dir", default="./output_hybrid", help="Output directory")
    parser.add_argument("--cache_dir", default="./dataset_cache", help="Directory to cache the built dataset")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for cosine decay scheduler")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


5tev-combine.log
add_secondary-5TeV.log
c2.log
c3.log
commands.txt
gemini/add_secondary-65tev.log
gemini/add_secondary.log
gemini/filter_trainfiles.py
gemini/filtered_tau.txt
gemini/merge-5TeV-moresamples.py
gemini/merge-65TeV-moresamples.py
nue_train_65_event_counts.csv
pca_feature_scan_5TeV.csv
redundant/trained_models/
string-level/infer.py