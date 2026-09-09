import argparse
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
from sklearn.metrics import roc_curve
from torch_geometric.loader import DataLoader

from train import (
    N_PCA_FEATURES, set_seed, build_geo_dict_kdtree,
    HybridNeutrinoDataset, HybridGNNClassifier, HybridDataLoader, evaluate_loader,
)


def infer(args):
    set_seed(42)

    if not os.path.exists(args.geo):
        print(f"ERROR: geometry file not found: {args.geo}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"ERROR: weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.scaler):
        print(f"ERROR: scaler file not found: {args.scaler}", file=sys.stderr)
        sys.exit(1)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)
    print(f"Geometry: {len(geo)} DOMs, KDTree built")

    pca_scaler = joblib.load(args.scaler)

    dataset = HybridNeutrinoDataset(
        nue_dbs                   = args.nue_dbs,
        tau_dbs                   = args.tau_dbs,
        kd_tree                   = kd_tree,
        string_ids_geo            = string_ids_geo,
        max_events_per_class      = args.max_events,
        charge_threshold          = args.charge_threshold,
        pca_scaler                = pca_scaler,  # transform only, don't refit
    )

    test_loader_gnn = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = HybridDataLoader(test_loader_gnn, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridGNNClassifier(gnn_hidden_dim=64, pca_dim=N_PCA_FEATURES, fusion_dim=32).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    test_loss, test_auc, test_preds, test_trues = evaluate_loader(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}")

    np.savez(
        os.path.join(args.save_dir, "test_predictions.npz"),
        preds=test_preds, trues=test_trues,
    )

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    fpr, tpr, _ = roc_curve(test_trues, test_preds)
    ax.plot(fpr, tpr, color="cyan", lw=2, label=f"AUC={test_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("FPR", color="white")
    ax.set_ylabel("TPR", color="white")
    ax.set_title("ROC Curve (Test)", color="white")
    ax.legend(facecolor="black", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "test_roc.png"), dpi=150, facecolor="black")
    plt.close()

    print(f"Saved predictions and ROC plot to {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid PCA+GNN Tau vs Electron Neutrino Classifier -- inference on held-out test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tau_dbs", nargs="+", required=True, help="Tau test DB paths")
    parser.add_argument("--nue_dbs", nargs="+", required=True, help="Nue test DB paths")
    parser.add_argument("--geo", default="/path/to/geometry_clean.csv", help="Geometry file")
    parser.add_argument("--weights", required=True, help="Path to hybrid_weights.pt from training")
    parser.add_argument("--scaler", required=True, help="Path to pca_scaler.joblib from training")
    parser.add_argument("--max_events", type=int, default=None, help="Max events per class")
    parser.add_argument("--charge_threshold", type=float, default=0.1, help="Charge threshold")
    parser.add_argument("--save_dir", default="./output_hybrid_test", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args)
