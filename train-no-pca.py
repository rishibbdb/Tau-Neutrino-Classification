"""
train_gnn_only_baseline.py

Minimal GNN-only baseline (no PCA fusion) for nutau vs nue classification.
Purpose: establish whether raw string-level graph structure carries any
separating signal above the ~0.51 AUC floor found for hand-engineered
PCA features on this dataset.
"""

import argparse
import copy
import glob
import os
import random
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from tqdm import tqdm


GNN_FEATURE_NAMES = [
    "str_x", "str_y", "z_centroid", "z_asym", "t_mean", "t_asym",
    "q_log", "n_doms", "t_spread", "z_rel", "t_first", "q_frac",
]
N_GNN_FEATURES = len(GNN_FEATURE_NAMES)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_geo_dict_kdtree(geo_df):
    coords = geo_df[["dom_x", "dom_y", "dom_z"]].values.astype(float)
    kd_tree = KDTree(coords, leafsize=40)
    string_ids = geo_df["string"].values.astype(int)
    return kd_tree, string_ids


def map_pulse_to_string(x, y, z, kd_tree, string_ids, max_distance=50.0):
    distance, idx = kd_tree.query([x, y, z])
    if distance > max_distance:
        return -1
    return int(string_ids[idx])


def stream_events_with_truth(db_file):
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"DB file not found: {db_file}")

    conn = sqlite3.connect(db_file)
    query = """
        SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge
        FROM   CleanedROIPulses p
        JOIN   truth            t ON p.event_no = t.event_no
        ORDER  BY p.event_no
    """
    cursor = conn.cursor()
    cursor.execute(query)
    current_event = None
    buffer = []

    for row in cursor:
        event_no = row[0]
        pulse = row[:6]

        if current_event is None:
            current_event = event_no

        if event_no != current_event:
            yield current_event, buffer
            buffer = []
            current_event = event_no

        buffer.append(pulse)

    if buffer:
        yield current_event, buffer

    conn.close()


# --------------------------------------------------------------------------
# Graph construction (string-level)
# --------------------------------------------------------------------------

def event_to_string_graph(rows, kd_tree, string_ids_geo, k=16,
                           t_scale=500.0, xyz_scale=500.0, charge_threshold=0.0):
    rows = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    t0 = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits, q_hits = t_hits[mask], q_hits[mask]
    t_hits = t_hits - t0

    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits = t_hits[q_mask], q_hits[q_mask]

    def _fallback():
        return Data(x=torch.zeros((1, N_GNN_FEATURES), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long))

    if len(t_hits) < 2:
        return _fallback()

    string_ids = np.array([
        map_pulse_to_string(xi, yi, zi, kd_tree, string_ids_geo)
        for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    if len(t_hits) < 2:
        return _fallback()

    q_total = q_hits.sum() + 1e-6
    z_q_center = np.sum(q_hits * z_hits) / q_total

    node_feats = []
    unique_strings = np.unique(string_ids)

    for s in unique_strings:
        mask_s = string_ids == s
        q_s = q_hits[mask_s]
        t_s = t_hits[mask_s]
        z_s = z_hits[mask_s]
        x_s = x_hits[mask_s]
        y_s = y_hits[mask_s]
        q_sum_s = q_s.sum() + 1e-6

        str_x = x_s[0] / xyz_scale
        str_y = y_s[0] / xyz_scale
        z_centroid = np.sum(q_s * z_s) / q_sum_s
        z_mid = np.median(z_s)
        q_top = q_s[z_s >= z_mid].sum()
        q_bot = q_s[z_s < z_mid].sum()
        z_asym = (q_top - q_bot) / q_sum_s
        t_mean_s = np.sum(q_s * t_s) / q_sum_s
        q_early = q_s[t_s <= 750.0].sum()
        q_late = q_s[t_s > 750.0].sum()
        t_asym = (q_early - q_late) / q_sum_s
        q_log = np.log1p(q_sum_s)
        n_doms = len(np.unique(z_s)) / 60.0
        t_spread = float(t_s.std()) / t_scale if len(t_s) > 1 else 0.0
        z_rel = (z_centroid - z_q_center) / xyz_scale
        t_first = float(t_s.min()) / t_scale
        q_frac = q_sum_s / q_total

        node_feats.append([str_x, str_y, z_centroid / xyz_scale, z_asym, t_mean_s / t_scale,
                            t_asym, q_log, n_doms, t_spread, z_rel, t_first, q_frac])

    node_feats = np.array(node_feats, dtype=np.float32)
    x_tensor = torch.tensor(node_feats, dtype=torch.float)
    pos = x_tensor[:, [0, 1, 2]]

    k_actual = min(k, x_tensor.size(0) - 1)
    if k_actual < 1:
        return _fallback()
    edge_index = knn_graph(pos, k=k_actual, loop=False)

    return Data(x=x_tensor, edge_index=edge_index)


def collect_graphs_from_dir(db_dir, kd_tree, string_ids_geo, n_files=20, pattern="*.db",
                             max_events_per_file=None, charge_threshold=0.1, label=0,
                             min_nodes=3):
    db_paths = sorted(glob.glob(os.path.join(db_dir, pattern)))[:n_files]
    if not db_paths:
        raise FileNotFoundError(f"No DB files matching {pattern} in {db_dir}")

    graphs = []
    n_seen = 0
    n_skipped = 0

    for p in tqdm(db_paths, desc=f"loading label={label}", unit="file"):
        n_this = 0
        for eid, rows in stream_events_with_truth(p):
            if max_events_per_file and n_this >= max_events_per_file:
                break
            n_this += 1
            n_seen += 1

            g = event_to_string_graph(rows, kd_tree, string_ids_geo,
                                       charge_threshold=charge_threshold)
            if g.x.shape[0] >= min_nodes:
                g.y = torch.tensor([float(label)])
                graphs.append(g)
            else:
                n_skipped += 1

    print(f"  {db_dir}: seen={n_seen}  kept={len(graphs)}  skipped(<{min_nodes} nodes)={n_skipped}")
    return graphs


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class GNNOnlyClassifier(nn.Module):
    def __init__(self, gnn_hidden_dim=64):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(N_GNN_FEATURES, gnn_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gnn_hidden_dim),
        )
        self.conv1 = TransformerConv(gnn_hidden_dim, gnn_hidden_dim, heads=4)
        self.conv2 = TransformerConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)
        self.conv3 = TransformerConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)

        self.bn1 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.act = nn.ReLU()
        self.drop_gnn = nn.Dropout(0.4)

        gnn_output_dim = gnn_hidden_dim * 4 + 1
        self.head = nn.Sequential(
            nn.Linear(gnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_mlp(x)
        x = self.drop_gnn(self.act(self.bn1(self.conv1(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn2(self.conv2(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn3(self.conv3(x, edge_index))))
        gnn_output = global_mean_pool(x, batch)
        mean_q = global_mean_pool(data.x[:, 6], batch).unsqueeze(1)
        gnn_features = torch.cat([gnn_output, mean_q], dim=1)
        return self.head(gnn_features).squeeze(-1)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y.view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            trues.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) >= 2 else float("nan")
    return avg_loss, auc, np.array(preds), np.array(trues)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(args):
    set_seed(42)

    if not os.path.exists(args.geo):
        print(f"ERROR: geometry file not found: {args.geo}", file=sys.stderr)
        sys.exit(1)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)
    print(f"Geometry: {len(geo)} DOMs, KDTree built")

    tau_graphs = collect_graphs_from_dir(
        args.tau_dir, kd_tree, string_ids_geo, n_files=args.n_files,
        pattern=args.tau_pattern, max_events_per_file=args.max_events_per_file,
        charge_threshold=args.charge_threshold, label=1,
    )
    nue_graphs = collect_graphs_from_dir(
        args.nue_dir, kd_tree, string_ids_geo, n_files=args.n_files,
        pattern=args.nue_pattern, max_events_per_file=args.max_events_per_file,
        charge_threshold=args.charge_threshold, label=0,
    )

    print(f"\nDataset summary:")
    print(f"  Tau: {len(tau_graphs)}")
    print(f"  Nue: {len(nue_graphs)}")

    all_graphs = tau_graphs + nue_graphs
    labels = np.array([g.y.item() for g in all_graphs])

    if len(all_graphs) == 0:
        raise RuntimeError("No graphs collected -- check DB paths and patterns.")

    indices = np.arange(len(all_graphs))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_frac, stratify=labels, random_state=42
    )
    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs = [all_graphs[i] for i in val_idx]
    print(f"Split data:  train: {len(train_graphs)}  val: {len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GNNOnlyClassifier(gnn_hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    n_pos = max(int((labels[train_idx] == 1).sum()), 1)
    n_neg = max(int((labels[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    epochs_no_gain = 0
    os.makedirs(args.save_dir, exist_ok=True)

    train_losses, val_losses, val_aucs = [], [], []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1:03d}", unit="batch", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            targets = batch.y.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        val_loss, val_auc, _, _ = evaluate_loader(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc if not np.isnan(val_auc) else 0.0)

        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1

        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
              f"| AUC: {val_auc_str} | Best: {best_val_auc:.4f}")

        if epochs_no_gain >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "gnn_only_weights.pt"))
    print(f"\nBest val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
    ax.plot(train_losses, color="cyan", label="Train")
    ax.plot(val_losses, color="orange", label="Val")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.set_title("GNN-only Training Curves (no PCA fusion)", color="white")
    ax.legend(facecolor="black", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "gnn_only_training_curves.png"), dpi=150, facecolor="black")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(val_aucs, color="tab:green")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="chance")
    ax.axhline(0.5146, color="tab:red", linestyle=":", lw=1, label="PCA-only baseline (0.515)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUC")
    ax.legend()
    ax.set_title("GNN-only val AUC vs. PCA-only baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "gnn_only_val_auc.png"), dpi=150)
    plt.close()

    print(f"Saved model and plots to {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN-only baseline (no PCA fusion) for nutau vs nue classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tau_dir", required=True, help="Directory with tau DB files")
    parser.add_argument("--nue_dir", required=True, help="Directory with nue DB files")
    parser.add_argument("--tau_pattern", default="nutau_gemini_ftp_65TeV_*.db")
    parser.add_argument("--nue_pattern", default="nue_gemini_ftp_65TeV_*.db")
    parser.add_argument("--geo", default="/path/to/geometry_clean.csv")
    parser.add_argument("--n_files", type=int, default=20, help="DB files per class")
    parser.add_argument("--max_events_per_file", type=int, default=None)
    parser.add_argument("--charge_threshold", type=float, default=0.1)
    parser.add_argument("--save_dir", default="./output_gnn_only")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)