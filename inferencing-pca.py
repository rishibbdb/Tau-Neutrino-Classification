"""
diagnose_hybrid.py

Load a trained HybridGNNClassifier + its saved PCA scaler, run inference on
tau/nue test DBs, and produce:
  - ROC, confusion matrix, score distribution, calibration plots
  - SHAP attribution over the 13 PCA_FEATURE_NAMES (GNN branch held fixed
    per event -- see module docstring in compute_shap_values)

Imports feature/graph construction directly from hybrid_common.py to
guarantee exact parity with what produced the checkpoint. Do not
reimplement stream_events_with_truth / compute_pca_features /
event_to_string_graph here -- copies of these have drifted from the
training version before and silently corrupted inference.
"""
import argparse
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, brier_score_loss,
)
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from hybrid_common import (
    N_PCA_FEATURES, PCA_FEATURE_NAMES,
    build_geo_dict_kdtree, HybridNeutrinoDataset, HybridGNNClassifier,
    HybridDataLoader,
)


def load_model_and_scaler(model_path, scaler_path, device):
    model = HybridGNNClassifier(gnn_hidden_dim=64, pca_dim=N_PCA_FEATURES, fusion_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dataset, device, batch_size=64):
    loader_gnn = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loader = HybridDataLoader(loader_gnn, dataset)

    all_probs, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for batch, pca_feats in loader:
            batch = batch.to(device)
            pca_feats = pca_feats.to(device)
            logits = model(batch, pca_feats)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_trues.extend(batch.y.view(-1).cpu().numpy())

    return np.array(all_probs), np.array(all_trues)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_diagnostics(probs, trues, save_dir):
    auc = roc_auc_score(trues, probs)
    fpr, tpr, _ = roc_curve(trues, probs)

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    ax.plot(fpr, tpr, color="tab:blue", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150, facecolor="white")
    plt.close()

    preds_bin = (probs >= 0.5).astype(int)
    cm = confusion_matrix(trues, preds_bin)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    ConfusionMatrixDisplay(cm, display_labels=["nue", "tau"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (thr=0.5)")
    ax.grid(False)  # grid lines would cut through the matrix cells
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, facecolor="white")
    plt.close()

    # Score distribution as overlaid KDE lines instead of bars
    from scipy.stats import gaussian_kde
    x_grid = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    for label, name, color in [(0, "nue", "tab:orange"), (1, "tau", "tab:blue")]:
        vals = probs[trues == label]
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        ax.plot(x_grid, kde(x_grid), color=color, label=name, linewidth=2)
        ax.fill_between(x_grid, kde(x_grid), color=color, alpha=0.15)
    ax.set_xlabel("P(tau)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "score_distribution.png"), dpi=150, facecolor="white")
    plt.close()

    frac_pos, mean_pred = calibration_curve(trues, probs, n_bins=10, strategy="quantile")
    brier = brier_score_loss(trues, probs)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    ax.plot(mean_pred, frac_pos, marker="o", color="tab:blue", label=f"Brier = {brier:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("Mean predicted P(tau)")
    ax.set_ylabel("Observed fraction tau")
    ax.set_title("Calibration")
    ax.legend()
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration.png"), dpi=150, facecolor="white")
    plt.close()

    print(f"AUC: {auc:.4f} | Brier: {brier:.4f}")
    print(f"Saved diagnostic plots to {save_dir}")


# ---------------------------------------------------------------------------
# SHAP -- local attribution over PCA_FEATURE_NAMES, GNN branch held fixed
# per event (raw pulse-level features aren't attributable through a
# variable-topology GNN, so this explains what the PCA branch adds on top
# of each event's fixed GNN embedding).
# ---------------------------------------------------------------------------

def get_gnn_features(model, batch):
    """Replays HybridGNNClassifier's GNN branch up to the fusion concat."""
    x, edge_index, b = batch.x, batch.edge_index, batch.batch
    x = model.node_mlp(x)
    x = model.drop_gnn(model.act(model.bn1(model.conv1(x, edge_index))))
    x = model.drop_gnn(model.act(model.bn2(model.conv2(x, edge_index))))
    x = model.drop_gnn(model.act(model.bn3(model.conv3(x, edge_index))))
    gnn_output = global_mean_pool(x, b)
    mean_q = global_mean_pool(batch.x[:, 6], b).unsqueeze(1)
    return torch.cat([gnn_output, mean_q], dim=1)


class PCABranchWrapper(nn.Module):
    """pca_mlp + fusion, with the GNN embedding fixed to one event."""
    def __init__(self, model, gnn_features_fixed):
        super().__init__()
        self.pca_mlp = model.pca_mlp
        self.fusion = model.fusion
        self.gnn_features_fixed = gnn_features_fixed

    def forward(self, pca_features):
        n = pca_features.shape[0]
        gnn_rep = self.gnn_features_fixed.repeat(n, 1)
        pca_out = self.pca_mlp(pca_features)
        fused = torch.cat([gnn_rep, pca_out], dim=1)
        return torch.sigmoid(self.fusion(fused))  # keep (n, 1) -- GradientExplainer needs 2D output


def compute_shap_values(model, dataset, device, n_events=100, background_size=100, seed=42):
    model.eval()
    rng = np.random.default_rng(seed)

    n_total = len(dataset)
    event_idx = rng.choice(n_total, size=min(n_events, n_total), replace=False)
    bg_idx = rng.choice(n_total, size=min(background_size, n_total), replace=False)

    background = torch.tensor(
        dataset.pca_features_scaled[bg_idx], dtype=torch.float, device=device
    )

    shap_rows = []
    for i in event_idx:
        single = Batch.from_data_list([dataset[int(i)]]).to(device)
        with torch.no_grad():
            gnn_feat = get_gnn_features(model, single)
        wrapper = PCABranchWrapper(model, gnn_feat).to(device)

        instance = torch.tensor(
            dataset.pca_features_scaled[i:i + 1], dtype=torch.float, device=device
        )
        explainer = shap.GradientExplainer(wrapper, background)
        sv = explainer.shap_values(instance)
        shap_rows.append(np.array(sv).reshape(-1))

    shap_array = np.stack(shap_rows)
    feature_array = dataset.pca_features_scaled[event_idx]
    return shap_array, feature_array, event_idx


def plot_shap(shap_array, feature_array, save_dir):
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_array, feature_array, feature_names=PCA_FEATURE_NAMES, show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=150)
    plt.close(fig)

    mean_abs = np.abs(shap_array).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.barh(
        [PCA_FEATURE_NAMES[j] for j in order][::-1],
        mean_abs[order][::-1],
        color="tab:blue",
    )
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("PCA Branch Feature Importance")
    ax.set_facecolor("white")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_importance.png"), dpi=150, facecolor="white")
    plt.close()

    print(f"Saved SHAP plots to {save_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # os.makedirs(args.save_dir, exist_ok=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model, scaler = load_model_and_scaler(args.model_path, args.scaler_path, device)

    # geo = pd.read_csv(args.geo)
    # kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)

    # dataset = HybridNeutrinoDataset(
    #     nue_dbs=args.nue_dbs, tau_dbs=args.tau_dbs,
    #     kd_tree=kd_tree, string_ids_geo=string_ids_geo,
    #     max_events_per_class=args.max_events,
    #     charge_threshold=args.charge_threshold,
    #     pca_scaler=scaler,  # reuse training-time scaler -- transform, not fit
    # )

    # probs, trues = run_inference(model, dataset, device, batch_size=args.batch_size)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_model_and_scaler(args.model_path, args.scaler_path, device)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)

    nue_dbs = resolve_db_inputs(args.nue_dbs, args.max_db_files)
    tau_dbs = resolve_db_inputs(args.tau_dbs, args.max_db_files)

    print(f"Using {len(nue_dbs)} nue DB files")
    print(f"Using {len(tau_dbs)} tau DB files")

    dataset = HybridNeutrinoDataset(
        nue_dbs=nue_dbs, tau_dbs=tau_dbs,
        kd_tree=kd_tree, string_ids_geo=string_ids_geo,
        max_events_per_class=args.max_events,
        charge_threshold=args.charge_threshold,
        pca_scaler=scaler,  # reuse training-time scaler -- transform, not fit
    )

    probs, trues = run_inference(model, dataset, device, batch_size=args.batch_size)
    pd.DataFrame({"true_label": trues, "prob_tau": probs}).to_csv(
        os.path.join(args.save_dir, "scores.csv"), index=False
    )

    plot_diagnostics(probs, trues, args.save_dir)

    shap_array, feature_array, event_idx = compute_shap_values(
        model, dataset, device,
        n_events=args.n_shap_events, background_size=args.shap_background,
    )
    plot_shap(shap_array, feature_array, args.save_dir)
    
def resolve_db_inputs(inputs, max_db_files=None):
    """Expand file/directory inputs into a sorted list of DB files."""
    db_files = []
    allowed_suffixes = {".db", ".sqlite", ".sqlite3"}

    for item in inputs:
        path = Path(item)
        if path.is_dir():
            matches = sorted(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in allowed_suffixes
            )
            db_files.extend(matches)
        elif path.is_file():
            db_files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {item}")

    db_files = [str(p) for p in db_files]

    if max_db_files is not None:
        db_files = db_files[:max_db_files]

    if not db_files:
        raise ValueError("No database files found from the provided inputs.")

    return db_files

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--scaler_path", required=True)
    p.add_argument("--tau_dbs", nargs="+", required=True)
    p.add_argument("--nue_dbs", nargs="+", required=True)
    p.add_argument("--max_db_files", type=int, default=None, help="Maximum number of DB files to use from each class input.")
    p.add_argument("--geo", required=True)
    p.add_argument("--max_events", type=int, default=None)
    p.add_argument("--charge_threshold", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--save_dir", default="./diagnostics")
    p.add_argument("--n_shap_events", type=int, default=100)
    p.add_argument("--shap_background", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

#python /mnt/scratch/baburish/doublepulse/gnn/Analysis/inferencing-pca.py --model_path pca_65TeV_16batchsize/hybrid_weights.pt --scaler_path pca_65TeV_16batchsize/pca_scaler.joblib --nue_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nue_test/ --tau_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nutau_test/ --max_db_files 20 --geo geometry_clean.csv --save_dir ./inference_pca_65TeV_Run20files

#python inferencing-pca.py --model_path pca_65TeV_16batchsize/hybrid_weights.pt --scaler_path pca_65TeV_16batchsize/pca_scaler.joblib --tau_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nutau_test/nutau_gemini_65TeV_lowerbound_skimmed_chunk_022633.000114_0000999_0000.db_test.db --nue_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nue_test/nue_gemini_65TeV_lowerbound_skimmed_chunk_022612.013261_0000.db_test.db --geo geometry_clean.csv --save_dir ./inference_pca_65TeV_Run2