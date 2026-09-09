import argparse
import copy
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch_cluster import knn_graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import unbatch


# ── Feature definitions ───────────────────────────────────────────────────────

GNN_FEATURE_NAMES = [
    "str_x", "str_y", "z_centroid", "z_asym", "t_mean", "t_asym",
    "q_log", "n_doms", "t_spread", "z_rel", "t_first", "q_frac",
    "distance_to_cascade", "log_energy"
]
N_GNN_FEATURES = len(GNN_FEATURE_NAMES)

PCA_FEATURE_NAMES = [
    'pc1_var', 'pc2_var', 'pc3_var', 'elongation',
    'depth_loading', 'time_loading', 'charge_loading',
    'n_strings', 'n_doms', 'n_hits', 'total_charge',
    'log_energy', 'pc1_pc2_sum'
]
N_PCA_FEATURES = len(PCA_FEATURE_NAMES)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Geometry ──────────────────────────────────────────────────────────────────

def build_geo_dict_kdtree(geo_df):
    coords     = geo_df[['dom_x', 'dom_y', 'dom_z']].values.astype(float)
    kd_tree    = KDTree(coords, leafsize=40)
    string_ids = geo_df['string'].values.astype(int)
    return kd_tree, string_ids

def map_pulse_to_string(x, y, z, kd_tree, string_ids, max_distance=50.0):
    distance, idx = kd_tree.query([x, y, z])
    if distance > max_distance:
        return -1
    return int(string_ids[idx])


# ── Data streaming ────────────────────────────────────────────────────────────

def stream_events_with_truth(db_file,
                              energy_threshold_min_tev=None,
                              energy_threshold_max_tev=None):
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"DB file not found: {db_file}")

    if energy_threshold_min_tev is None:
        energy_threshold_min_tev = 0.05

    conn = sqlite3.connect(db_file)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(truth)").fetchall()]

    for col in ("cascade_reco_energy_tev", "cascade_vertex_fit_x"):
        if col not in cols:
            raise RuntimeError(f"'{col}' missing from truth table in {db_file}")

    where = f"WHERE t.cascade_reco_energy_tev >= {energy_threshold_min_tev}"
    if energy_threshold_max_tev is not None:
        where += f" AND t.cascade_reco_energy_tev <= {energy_threshold_max_tev}"

    query = f"""
        SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
               t.cascade_reco_energy_tev,
               t.cascade_vertex_fit_x, t.cascade_vertex_fit_y, t.cascade_vertex_fit_z
        FROM   CleanedROIPulses p
        JOIN   truth            t ON p.event_no = t.event_no
        {where}
        ORDER  BY p.event_no
    """

    cursor        = conn.cursor()
    cursor.execute(query)
    current_event = None
    buffer        = []
    meta_store    = {}

    for row in cursor:
        eid = row[0]
        if current_event is None:
            current_event = eid
        if eid != current_event:
            yield current_event, buffer, meta_store.get(current_event, {})
            buffer        = []
            current_event = eid
        buffer.append(row[:6])
        if eid not in meta_store:
            meta_store[eid] = {
                "cascade_reco_energy_tev": float(row[6]),
                "cascade_vertex_fit_x":    float(row[7]),
                "cascade_vertex_fit_y":    float(row[8]),
                "cascade_vertex_fit_z":    float(row[9]),
            }

    if buffer:
        yield current_event, buffer, meta_store.get(current_event, {})

    conn.close()


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_pca_features(rows, kd_tree, string_ids_geo, meta=None,
                          t_scale=500.0, xyz_scale=500.0, charge_threshold=0.0):
    rows   = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    t0   = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits = t_hits[mask] - t0
    q_hits = q_hits[mask]

    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits          = t_hits[q_mask], q_hits[q_mask]

    string_ids = np.array([
        map_pulse_to_string(xi, yi, zi, kd_tree, string_ids_geo)
        for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits     = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    _invalid = {
        'is_valid': False,
        'pc1_var': 0.0, 'pc2_var': 0.0, 'pc3_var': 1.0, 'elongation': 1.0,
        'depth_loading': 0.0, 'time_loading': 0.0, 'charge_loading': 0.0,
        'n_strings': 0.0, 'n_doms': 0.0, 'n_hits': 0.0,
        'total_charge': 0.0, 'log_energy': 0.0, 'pc1_pc2_sum': 0.0,
    }
    if len(t_hits) < 4 or len(np.unique(string_ids)) < 2:
        return _invalid

    charges = q_hits
    weights = charges / charges.sum()
    X_spatial  = np.column_stack([x_hits, y_hits, z_hits])
    X_centered = X_spatial - np.average(X_spatial, weights=weights, axis=0)
    cov_w      = np.cov(X_centered.T, aweights=weights)
    evals, _   = np.linalg.eigh(cov_w)
    evals      = evals[::-1]
    total_var  = evals.sum() + 1e-10
    pc1_var    = float(evals[0] / total_var)
    pc2_var    = float(evals[1] / total_var)
    pc3_var    = float(evals[2] / total_var)
    elongation = float(evals[0] / (evals[2] + 1e-10))

    X_temp   = np.column_stack([t_hits, q_hits, z_hits])
    X_tscale = StandardScaler().fit_transform(X_temp)
    cov_t    = np.cov(X_tscale.T)
    evals_t, evecs_t = np.linalg.eigh(cov_t)
    evecs_t  = evecs_t[:, np.argsort(evals_t)[::-1]]
    pc1_load = evecs_t[:, 0]

    cascade_reco_energy_tev = float(meta.get("cascade_reco_energy_tev", 0.05)) if meta else 0.05
    log_energy = np.log10(max(cascade_reco_energy_tev, 0.05)) / 6.0

    return {
        'is_valid':       True,
        'pc1_var':        pc1_var,
        'pc2_var':        pc2_var,
        'pc3_var':        pc3_var,
        'elongation':     min(elongation, 50.0),
        'depth_loading':  float(np.abs(pc1_load[2])),
        'time_loading':   float(np.abs(pc1_load[0])),
        'charge_loading': float(np.abs(pc1_load[1])),
        'n_strings':      float(len(np.unique(string_ids))),
        'n_doms':         float(len(np.unique(np.column_stack([x_hits, y_hits, z_hits]), axis=0))),
        'n_hits':         float(len(t_hits)),
        'total_charge':   float(charges.sum()),
        'log_energy':     log_energy,
        'pc1_pc2_sum':    pc1_var + pc2_var,
    }


def event_to_string_graph(rows, kd_tree, string_ids_geo, meta=None,
                           t_scale=500.0, xyz_scale=500.0, charge_threshold=0.0):
    rows   = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    t0   = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits = t_hits[mask] - t0
    q_hits = q_hits[mask]

    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits          = t_hits[q_mask], q_hits[q_mask]

    def _fallback():
        return Data(
            x          = torch.zeros((1, N_GNN_FEATURES), dtype=torch.float),
            edge_index = torch.zeros((2, 0), dtype=torch.long),
        )

    if len(t_hits) < 2:
        return _fallback()

    string_ids = np.array([
        map_pulse_to_string(xi, yi, zi, kd_tree, string_ids_geo)
        for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits     = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    if len(t_hits) < 2:
        return _fallback()

    q_total    = q_hits.sum() + 1e-6
    z_q_center = np.sum(q_hits * z_hits) / q_total

    cascade_reco_energy_tev = float(meta.get("cascade_reco_energy_tev", 0.05)) if meta else 0.05
    log_energy = np.log10(max(cascade_reco_energy_tev, 0.05)) / 6.0

    cx = float(meta.get("cascade_vertex_fit_x", 0.0)) if meta else 0.0
    cy = float(meta.get("cascade_vertex_fit_y", 0.0)) if meta else 0.0
    cz = float(meta.get("cascade_vertex_fit_z", 0.0)) if meta else 0.0
    cx_n, cy_n, cz_n = cx / xyz_scale, cy / xyz_scale, cz / xyz_scale

    node_feats = []
    for s in np.unique(string_ids):
        m       = string_ids == s
        q_s     = q_hits[m];  t_s = t_hits[m]
        z_s     = z_hits[m];  x_s = x_hits[m];  y_s = y_hits[m]
        q_sum_s = q_s.sum() + 1e-6

        str_x      = x_s[0] / xyz_scale
        str_y      = y_s[0] / xyz_scale
        str_z      = z_s[0] / xyz_scale
        z_centroid = np.sum(q_s * z_s) / q_sum_s
        z_mid      = np.median(z_s)
        z_asym     = (q_s[z_s >= z_mid].sum() - q_s[z_s < z_mid].sum()) / q_sum_s
        t_mean_s   = np.sum(q_s * t_s) / q_sum_s
        t_asym     = (q_s[t_s <= 750.].sum() - q_s[t_s > 750.].sum()) / q_sum_s
        q_log      = np.log1p(q_sum_s)
        n_doms_s   = len(np.unique(z_s)) / 60.0
        t_spread   = float(t_s.std()) / t_scale if len(t_s) > 1 else 0.0
        z_rel      = (z_centroid - z_q_center) / xyz_scale
        t_first    = float(t_s.min()) / t_scale
        q_frac     = q_sum_s / q_total
        dist_casc  = np.sqrt(
            (str_x - cx_n)**2 + (str_y - cy_n)**2 + (str_z - cz_n)**2
        )
        node_feats.append([
            str_x, str_y, z_centroid / xyz_scale, z_asym,
            t_mean_s / t_scale, t_asym, q_log, n_doms_s,
            t_spread, z_rel, t_first, q_frac, dist_casc, log_energy
        ])

    x_t  = torch.tensor(np.array(node_feats, dtype=np.float32), dtype=torch.float)
    pos  = x_t[:, [0, 1, 2]]
    k    = min(16, x_t.size(0) - 1)
    if k < 1:
        return _fallback()

    return Data(x=x_t, edge_index=knn_graph(pos, k=k, loop=False))


# ── Dataset ───────────────────────────────────────────────────────────────────

class HybridNeutrinoDataset(InMemoryDataset):
    def __init__(self, nue_dbs, tau_dbs, kd_tree, string_ids_geo,
                 max_events_per_class=None, charge_threshold=0.0,
                 energy_threshold_min_tev=None, energy_threshold_max_tev=None):
        super().__init__()

        if isinstance(nue_dbs, str): nue_dbs = [nue_dbs]
        if isinstance(tau_dbs, str): tau_dbs = [tau_dbs]

        data_list        = []
        pca_features_list = []

        def _load(dbs, label, max_n, desc):
            count = 0
            for db in tqdm(dbs, desc=f"  {desc} files", unit="file"):
                for eid, rows, meta in tqdm(
                    stream_events_with_truth(db,
                        energy_threshold_min_tev=energy_threshold_min_tev,
                        energy_threshold_max_tev=energy_threshold_max_tev),
                    desc=f"    {os.path.basename(db)}", unit="ev", leave=False
                ):
                    if max_n and count >= max_n:
                        break
                    pca_f = compute_pca_features(
                        rows, kd_tree, string_ids_geo, meta=meta,
                        charge_threshold=charge_threshold)
                    if not pca_f['is_valid']:
                        continue
                    g   = event_to_string_graph(
                        rows, kd_tree, string_ids_geo, meta=meta,
                        charge_threshold=charge_threshold)
                    g.y = torch.tensor([float(label)])
                    data_list.append(g)
                    pca_features_list.append(pca_f)
                    count += 1
                if max_n and count >= max_n:
                    break
            return count

        print(f"Loading tau events...")
        tau_count = _load(tau_dbs, 1, max_events_per_class, "tau")
        print(f"  -> {tau_count} tau events")

        nue_max = tau_count if (max_events_per_class and tau_count < max_events_per_class) \
                  else max_events_per_class
        print(f"Loading nue events (max={nue_max})...")
        nue_count = _load(nue_dbs, 0, nue_max, "nue")
        print(f"  -> {nue_count} nue events")

        if not data_list:
            raise RuntimeError("Dataset is empty.")

        print("Collating...")
        self.data, self.slices = self.collate(data_list)

        self.pca_features_array = np.array(
            [[f[n] for n in PCA_FEATURE_NAMES] for f in pca_features_list],
            dtype=np.float32
        )
        self.pca_scaler          = StandardScaler()
        self.pca_features_scaled = self.pca_scaler.fit_transform(self.pca_features_array)

    def get_pca_features(self, idx):
        return torch.tensor(self.pca_features_scaled[idx], dtype=torch.float)


class HybridDataLoader:
    """Yields (batch, pca_features) tuples in order."""
    def __init__(self, gnn_loader, dataset):
        self.gnn_loader = gnn_loader
        self.dataset    = dataset

    def __iter__(self):
        self.iter_gnn    = iter(self.gnn_loader)
        self.batch_start = 0
        return self

    def __next__(self):
        batch = next(self.iter_gnn)
        end   = self.batch_start + batch.num_graphs
        pca   = torch.stack([
            self.dataset.get_pca_features(i)
            for i in range(self.batch_start, end)
        ])
        self.batch_start = end
        return batch, pca

    def __len__(self):
        return len(self.gnn_loader)


# ── GNN Encoder (shared by AE and classifier) ─────────────────────────────────

class GNNEncoder(nn.Module):
    """
    TransformerConv encoder → global pool → latent vector.
    Shared between the autoencoder pretraining and the classifier.
    """
    def __init__(self, hidden_dim=64, latent_dim=64):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(N_GNN_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.conv1 = TransformerConv(hidden_dim,     hidden_dim, heads=4)
        self.conv2 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4)
        self.conv3 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4)

        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(0.4)

        # Pool → latent
        # +1 for mean_q appended after pooling
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_mlp(x)
        x = self.drop(self.act(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(self.act(self.bn2(self.conv2(x, edge_index))))
        x = self.drop(self.act(self.bn3(self.conv3(x, edge_index))))

        pooled = global_mean_pool(x, batch)                          # (B, hidden*4)
        mean_q = global_mean_pool(data.x[:, 6], batch).unsqueeze(1)  # (B, 1)  q_log
        return self.project(torch.cat([pooled, mean_q], dim=1))      # (B, latent_dim)


# ── Autoencoder ───────────────────────────────────────────────────────────────

class GNNDecoder(nn.Module):
    """
    Decode latent vector → (mean, std) of each node feature across the graph.
    Permutation-invariant target: avoids reconstructing the variable-size node matrix.
    """
    def __init__(self, latent_dim=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_GNN_FEATURES * 2),   # mean + std per feature
        )

    def forward(self, z):
        out = self.net(z)
        return out[:, :N_GNN_FEATURES], out[:, N_GNN_FEATURES:]   # pred_mean, pred_std


class GNNAutoencoder(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.encoder = GNNEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = GNNDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def forward(self, data):
        z                    = self.encoder(data)
        pred_mean, pred_std  = self.decoder(z)
        return z, pred_mean, pred_std


def graph_node_stats(data):
    """Per-graph mean and std of node features. Permutation-invariant AE target."""
    graphs     = unbatch(data.x, data.batch)
    true_means = torch.stack([g.mean(0) for g in graphs])
    true_stds  = torch.stack([
        g.std(0) if g.shape[0] > 1
        else torch.zeros(N_GNN_FEATURES, device=g.device)
        for g in graphs
    ])
    return true_means, true_stds


# ── Classifier (encoder + PCA branch + fusion) ────────────────────────────────

class HybridClassifier(nn.Module):
    """
    Reuses a pretrained GNNEncoder.
    freeze_encoder=True  → linear probe (only fusion head trained)
    freeze_encoder=False → full fine-tune (encoder updated at lower lr)
    """
    def __init__(self, encoder, latent_dim=64, pca_dim=N_PCA_FEATURES,
                 fusion_dim=32, freeze_encoder=False):
        super().__init__()

        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.pca_mlp = nn.Sequential(
            nn.Linear(pca_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3),
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, data, pca_features):
        z       = self.encoder(data)                           # (B, latent_dim)
        pca_out = self.pca_mlp(pca_features)                   # (B, fusion_dim)
        return self.fusion(torch.cat([z, pca_out], dim=1)).squeeze(-1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss   = 0.0
    with torch.no_grad():
        for batch, pca_feats in loader:
            batch     = batch.to(device)
            pca_feats = pca_feats.to(device)
            logits    = model(batch, pca_feats)
            targets   = batch.y.view(-1)
            loss      = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            trues.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) >= 2 else float("nan")
    return avg_loss, auc, np.array(preds), np.array(trues)


# ── Stage 1: autoencoder pretraining ─────────────────────────────────────────

def pretrain_autoencoder(dataset, args, device):
    print("\n" + "=" * 60)
    print("Stage 1 — GNN Autoencoder pretraining")
    print("=" * 60)

    indices              = np.arange(len(dataset))
    train_idx, val_idx   = train_test_split(indices, test_size=0.15, random_state=42)

    train_loader = DataLoader(dataset[train_idx.tolist()],
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(dataset[val_idx.tolist()],
                              batch_size=args.batch_size, num_workers=4)

    ae        = GNNAutoencoder(hidden_dim=64, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=args.ae_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    best_val   = np.inf
    best_state = None
    no_gain    = 0
    ae_train_losses, ae_val_losses = [], []

    for epoch in range(args.ae_epochs):
        ae.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"AE {epoch+1:03d}", leave=False):
            batch = batch.to(device)
            true_means, true_stds = graph_node_stats(batch)
            optimizer.zero_grad()
            _, pred_mean, pred_std = ae(batch)
            loss = (nn.functional.mse_loss(pred_mean, true_means) +
                    0.5 * nn.functional.mse_loss(pred_std, true_stds))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)

        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                true_means, true_stds = graph_node_stats(batch)
                _, pred_mean, pred_std = ae(batch)
                loss = (nn.functional.mse_loss(pred_mean, true_means) +
                        0.5 * nn.functional.mse_loss(pred_std, true_stds))
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)

        ae_train_losses.append(train_loss)
        ae_val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(ae.state_dict())
            no_gain    = 0
        else:
            no_gain += 1

        print(f"AE {epoch+1:3d} | train: {train_loss:.5f} | val: {val_loss:.5f} | best: {best_val:.5f}")

        if no_gain >= args.ae_patience:
            print(f"AE early stop at epoch {epoch+1}")
            break

    ae.load_state_dict(best_state)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(ae.encoder.state_dict(),
               os.path.join(args.save_dir, "ae_encoder_weights.pt"))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="black")
    ax.plot(ae_train_losses, color="cyan",   label="Train")
    ax.plot(ae_val_losses,   color="orange", label="Val")
    ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("Recon loss", color="white")
    ax.set_title("Autoencoder pretraining", color="white")
    ax.legend(facecolor="black", labelcolor="white")
    ax.set_facecolor("black"); ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "ae_curves.png"), dpi=150, facecolor="black")
    plt.close()
    print(f"Encoder weights saved to {args.save_dir}/ae_encoder_weights.pt")

    return ae.encoder   # pretrained, ready to plug into classifier


# ── Stage 2: classifier fine-tuning ──────────────────────────────────────────

def finetune_classifier(dataset, pretrained_encoder, args, device):
    print("\n" + "=" * 60)
    print("Stage 2 — Classifier fine-tuning")
    print("=" * 60)

    indices = np.arange(len(dataset))
    labels  = np.array([dataset[i].y.item() for i in range(len(dataset))])

    holdout      = args.val_frac + args.test_frac
    train_idx, holdout_idx = train_test_split(
        indices, test_size=holdout, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size    = args.test_frac / holdout,
        stratify     = labels[holdout_idx],
        random_state = 42,
    )
    print(f"Split — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    def make_loader(idx, shuffle):
        gnn = DataLoader(dataset[idx.tolist()], batch_size=args.batch_size,
                         shuffle=shuffle, num_workers=4, pin_memory=shuffle)
        return HybridDataLoader(gnn, dataset)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    model = HybridClassifier(
        encoder        = pretrained_encoder.to(device),
        latent_dim     = args.latent_dim,
        pca_dim        = N_PCA_FEATURES,
        fusion_dim     = 32,
        freeze_encoder = args.freeze_encoder,
    ).to(device)

    # Two param groups: encoder at lower lr to preserve pretrained features
    if not args.freeze_encoder:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(),  "lr": args.lr * 0.1},
            {"params": model.pca_mlp.parameters(),  "lr": args.lr},
            {"params": model.fusion.parameters(),   "lr": args.lr},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    n_pos      = max(int((labels[train_idx] == 1).sum()), 1)
    n_neg      = max(int((labels[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc   = -np.inf
    best_state = None
    no_gain    = 0
    train_losses, val_losses, val_aucs = [], [], []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1:03d}", leave=False)
        for batch, pca_feats in pbar:
            batch     = batch.to(device)
            pca_feats = pca_feats.to(device)
            optimizer.zero_grad()
            logits  = model(batch, pca_feats)
            loss    = criterion(logits, batch.y.view(-1))
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

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_gain    = 0
        else:
            no_gain += 1

        print(f"Epoch {epoch+1:3d} | train: {train_loss:.4f} | val: {val_loss:.4f} "
              f"| auc: {val_auc:.4f} | best: {best_auc:.4f}")

        if no_gain >= args.patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "hybrid_weights.pt"))

    test_loss, test_auc, test_preds, test_trues = evaluate_loader(
        model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f}  |  Test AUC: {test_auc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="black")
    axes[0].plot(train_losses, color="cyan",   label="Train")
    axes[0].plot(val_losses,   color="orange", label="Val")
    axes[0].set_xlabel("Epoch", color="white"); axes[0].set_ylabel("Loss", color="white")
    axes[0].set_title("Classifier training", color="white")
    axes[0].legend(facecolor="black", labelcolor="white")
    axes[0].set_facecolor("black"); axes[0].tick_params(colors="white")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    fpr, tpr, _ = roc_curve(test_trues, test_preds)
    axes[1].plot(fpr, tpr, color="cyan", lw=2, label=f"AUC={test_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[1].set_xlabel("FPR", color="white"); axes[1].set_ylabel("TPR", color="white")
    axes[1].set_title("ROC Curve", color="white")
    axes[1].legend(facecolor="black", labelcolor="white")
    axes[1].set_facecolor("black"); axes[1].tick_params(colors="white")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "hybrid_curves.png"), dpi=150, facecolor="black")
    plt.close()

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def train(args):
    set_seed(42)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)
    print(f"Geometry: {len(geo)} DOMs loaded")

    dataset = HybridNeutrinoDataset(
        nue_dbs                  = args.nue_dbs,
        tau_dbs                  = args.tau_dbs,
        kd_tree                  = kd_tree,
        string_ids_geo           = string_ids_geo,
        max_events_per_class     = args.max_events,
        charge_threshold         = args.charge_threshold,
        energy_threshold_min_tev = getattr(args, "energy_threshold_min_tev", None),
        energy_threshold_max_tev = getattr(args, "energy_threshold_max_tev", None),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pretrained_encoder = pretrain_autoencoder(dataset, args, device)
    finetune_classifier(dataset, pretrained_encoder, args, device)

    print(f"\nAll outputs saved to {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AE-pretrained Hybrid PCA+GNN Neutrino Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tau_dbs", nargs="+", required=True)
    parser.add_argument("--nue_dbs", nargs="+", required=True)
    parser.add_argument("--geo",     default="/path/to/geometry_clean.csv")
    parser.add_argument("--max_events",              type=int,   default=None)
    parser.add_argument("--charge_threshold",        type=float, default=0.1)
    parser.add_argument("--energy_threshold_min_tev",type=float, default=None)
    parser.add_argument("--energy_threshold_max_tev",type=float, default=None)
    parser.add_argument("--save_dir",                default="./output_hybrid")
    # AE
    parser.add_argument("--latent_dim",   type=int,   default=64)
    parser.add_argument("--ae_epochs",    type=int,   default=50)
    parser.add_argument("--ae_lr",        type=float, default=1e-3)
    parser.add_argument("--ae_patience",  type=int,   default=15)
    # Classifier
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights during classifier training")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--val_frac",    type=float, default=0.15)
    parser.add_argument("--test_frac",   type=float, default=0.15)
    parser.add_argument("--patience",    type=int,   default=35)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)