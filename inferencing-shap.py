import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

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


# ── 1. Wrap the model so SHAP can call it with just PCA features ──────────────
# We freeze a representative GNN embedding and vary only PCA inputs.
# This isolates PCA feature importance while holding GNN output constant.


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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_geo_dict_kdtree(geo_df):
    """Build KDTree for nearest-neighbor geometry mapping."""
    coords = geo_df[['dom_x', 'dom_y', 'dom_z']].values.astype(float)
    kd_tree = KDTree(coords, leafsize=40)
    string_ids = geo_df['string'].values.astype(int)
    return kd_tree, string_ids

def map_pulse_to_string(x, y, z, kd_tree, string_ids, max_distance=50.0):
    """Map pulse to nearest string."""
    distance, idx = kd_tree.query([x, y, z])
    if distance > max_distance:
        return -1
    return int(string_ids[idx])

def stream_events_with_truth(db_file, energy_threshold_min_tev=None, energy_threshold_max_tev=None):
    """Stream events row by row from SQLite, joining truth info."""
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"DB file not found: {db_file}")

    if energy_threshold_min_tev is None:
        energy_threshold_min_tev = 0.05
    
    conn = sqlite3.connect(db_file)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(truth)").fetchall()]

    if "cascade_reco_energy_tev" not in cols:
        raise RuntimeError(f"'cascade_reco_energy_tev' column missing from truth table in {db_file}")
    if "cascade_vertex_fit_x" not in cols:
        raise RuntimeError(f"'cascade_vertex_fit_x' column missing from truth table in {db_file}")

    where_clause = f"WHERE t.cascade_reco_energy_tev >= {energy_threshold_min_tev}"
    if energy_threshold_max_tev is not None:
        where_clause += f" AND t.cascade_reco_energy_tev <= {energy_threshold_max_tev}"

    query = f"""
        SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
               t.cascade_reco_energy_tev, t.cascade_vertex_fit_x, 
               t.cascade_vertex_fit_y, t.cascade_vertex_fit_z
        FROM   CleanedROIPulses p
        JOIN   truth            t ON p.event_no = t.event_no
        {where_clause}
        ORDER  BY p.event_no
    """

    cursor        = conn.cursor()
    cursor.execute(query)
    current_event = None
    buffer        = []
    meta_store    = {}

    for row in cursor:
        event_no = row[0]
        pulse    = row[:6]
        cascade_reco_energy_tev = row[6]
        cascade_vertex_fit_x    = row[7]
        cascade_vertex_fit_y    = row[8]
        cascade_vertex_fit_z    = row[9]

        if current_event is None:
            current_event = event_no

        if event_no != current_event:
            yield current_event, buffer, meta_store.get(current_event, {})
            buffer        = []
            current_event = event_no

        buffer.append(pulse)

        if event_no not in meta_store:
            meta_store[event_no] = {
                "cascade_reco_energy_tev": float(cascade_reco_energy_tev),
                "cascade_vertex_fit_x": float(cascade_vertex_fit_x),
                "cascade_vertex_fit_y": float(cascade_vertex_fit_y),
                "cascade_vertex_fit_z": float(cascade_vertex_fit_z),
            }

    if buffer:
        yield current_event, buffer, meta_store.get(current_event, {})

    conn.close()

def compute_pca_features(rows, kd_tree, string_ids_geo, meta=None, 
                         t_scale=500.0, xyz_scale=500.0, charge_threshold=0.0):
    """
    Compute PCA-informed features from raw event data.
    Returns dict with 13 PCA-derived features.
    """
    rows   = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    # Time window
    t0   = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits, q_hits          = t_hits[mask], q_hits[mask]
    t_hits                  = t_hits - t0

    # Charge threshold
    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits          = t_hits[q_mask], q_hits[q_mask]

    # Map to strings
    string_ids = np.array([
        map_pulse_to_string(xi, yi, zi, kd_tree, string_ids_geo)
        for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits     = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    # Check if we have enough data
    if len(t_hits) < 4 or len(np.unique(string_ids)) < 2:
        return {
            'is_valid': False,
            'pc1_var': 0.0, 'pc2_var': 0.0, 'pc3_var': 1.0,
            'elongation': 1.0, 'depth_loading': 0.0, 'time_loading': 0.0,
            'charge_loading': 0.0, 'n_strings': 0.0, 'n_doms': 0.0, 'n_hits': 0.0,
            'total_charge': 0.0, 'log_energy': 0.0, 'pc1_pc2_sum': 0.0
        }

    X_spatial = np.column_stack([x_hits, y_hits, z_hits])
    
    # Charge-weighted covariance
    charges = q_hits
    weights = charges / charges.sum()
    X_centered = X_spatial - np.average(X_spatial, weights=weights, axis=0)
    cov_weighted = np.cov(X_centered.T, aweights=weights)
    eigenvals_spatial, eigenvecs_spatial = np.linalg.eigh(cov_weighted)
    idx = np.argsort(eigenvals_spatial)[::-1]
    eigenvals_spatial = eigenvals_spatial[idx]
    
    # Variance ratios
    total_var = eigenvals_spatial.sum() + 1e-10
    pc1_var = float(eigenvals_spatial[0] / total_var)
    pc2_var = float(eigenvals_spatial[1] / total_var)
    pc3_var = float(eigenvals_spatial[2] / total_var)
    elongation = float(eigenvals_spatial[0] / (eigenvals_spatial[2] + 1e-10))

    X_temporal = np.column_stack([t_hits, q_hits, z_hits])
    scaler = StandardScaler()
    X_temporal_scaled = scaler.fit_transform(X_temporal)
    
    cov_temporal = np.cov(X_temporal_scaled.T)
    eigenvals_temporal, eigenvecs_temporal = np.linalg.eigh(cov_temporal)
    idx_t = np.argsort(eigenvals_temporal)[::-1]
    eigenvecs_temporal = eigenvecs_temporal[:, idx_t]
    
    pc1_loadings = eigenvecs_temporal[:, 0]
    depth_loading = float(np.abs(pc1_loadings[2]))
    time_loading = float(np.abs(pc1_loadings[0]))
    charge_loading = float(np.abs(pc1_loadings[1]))

    cascade_reco_energy_tev = float(meta.get("cascade_reco_energy_tev", 0.05)) if meta else 0.05
    log_energy = np.log10(max(cascade_reco_energy_tev, 0.05)) / 6.0

    n_strings = len(np.unique(string_ids))
    n_doms = len(np.unique(np.column_stack([x_hits, y_hits, z_hits]), axis=0))
    n_hits = len(t_hits)
    total_charge = float(charges.sum())

    return {
        'is_valid': True,
        'pc1_var': pc1_var,
        'pc2_var': pc2_var,
        'pc3_var': pc3_var,
        'elongation': min(elongation, 50.0),
        'depth_loading': depth_loading,
        'time_loading': time_loading,
        'charge_loading': charge_loading,
        'n_strings': float(n_strings),
        'n_doms': float(n_doms),
        'n_hits': float(n_hits),
        'total_charge': total_charge,
        'log_energy': log_energy,
        'pc1_pc2_sum': pc1_var + pc2_var,
    }

def event_to_string_graph(rows, kd_tree, string_ids_geo, meta=None,
                           k=16, t_scale=500.0, xyz_scale=500.0,
                           charge_threshold=0.0):
    """Convert raw pulse rows for one event into a PyG Data object."""
    rows   = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    t0   = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits, q_hits          = t_hits[mask], q_hits[mask]
    t_hits                  = t_hits - t0

    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits          = t_hits[q_mask], q_hits[q_mask]

    def _fallback():
        return Data(
            x          = torch.zeros((1, N_GNN_FEATURES), dtype=torch.float),
            edge_index = torch.zeros((2, 0),              dtype=torch.long),
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
    
    cascade_vertex_fit_x = float(meta.get("cascade_vertex_fit_x", 0.0)) if meta else 0.0
    cascade_vertex_fit_y = float(meta.get("cascade_vertex_fit_y", 0.0)) if meta else 0.0
    cascade_vertex_fit_z = float(meta.get("cascade_vertex_fit_z", 0.0)) if meta else 0.0
    
    cascade_vertex_x_norm = cascade_vertex_fit_x / xyz_scale
    cascade_vertex_y_norm = cascade_vertex_fit_y / xyz_scale
    cascade_vertex_z_norm = cascade_vertex_fit_z / xyz_scale

    node_feats     = []
    unique_strings = np.unique(string_ids)

    for s in unique_strings:
        mask_s  = string_ids == s
        q_s     = q_hits[mask_s]
        t_s     = t_hits[mask_s]
        z_s     = z_hits[mask_s]
        x_s     = x_hits[mask_s]
        y_s     = y_hits[mask_s]
        q_sum_s = q_s.sum() + 1e-6

        str_x      = x_s[0] / xyz_scale
        str_y      = y_s[0] / xyz_scale
        str_z      = z_s[0] / xyz_scale
        z_centroid = np.sum(q_s * z_s) / q_sum_s
        z_mid      = np.median(z_s)
        q_top      = q_s[z_s >= z_mid].sum()
        q_bot      = q_s[z_s <  z_mid].sum()
        z_asym     = (q_top - q_bot) / q_sum_s
        t_mean_s   = np.sum(q_s * t_s) / q_sum_s
        q_early    = q_s[t_s <= 750.0].sum()
        q_late     = q_s[t_s >  750.0].sum()
        t_asym     = (q_early - q_late) / q_sum_s
        q_log      = np.log1p(q_sum_s)
        n_doms     = len(np.unique(z_s)) / 60.0
        t_spread   = float(t_s.std()) / t_scale if len(t_s) > 1 else 0.0
        z_rel      = (z_centroid - z_q_center) / xyz_scale
        t_first    = float(t_s.min()) / t_scale
        q_frac     = q_sum_s / q_total
        
        distance_to_cascade = np.sqrt(
            (str_x - cascade_vertex_x_norm)**2 +
            (str_y - cascade_vertex_y_norm)**2 +
            (str_z / xyz_scale - cascade_vertex_z_norm)**2
        )

        node_feats.append([
            str_x, str_y, z_centroid / xyz_scale, z_asym, t_mean_s / t_scale,
            t_asym, q_log, n_doms, t_spread, z_rel, t_first, q_frac,
            distance_to_cascade, log_energy
        ])

    node_feats = np.array(node_feats, dtype=np.float32)
    x_tensor   = torch.tensor(node_feats, dtype=torch.float)
    pos        = x_tensor[:, [0, 1, 2]]
    
    k_actual   = min(16, x_tensor.size(0) - 1)
    if k_actual < 1:
        return _fallback()
    edge_index = knn_graph(pos, k=k_actual, loop=False)
    
    return Data(x=x_tensor, edge_index=edge_index)

class HybridNeutrinoDataset(InMemoryDataset):
    def __init__(self, nue_dbs, tau_dbs, kd_tree, string_ids_geo,
                 max_events_per_class=None, charge_threshold=0.0,
                 energy_threshold_min_tev=None, energy_threshold_max_tev=None):
        super().__init__()

        if isinstance(nue_dbs, str): nue_dbs = [nue_dbs]
        if isinstance(tau_dbs, str): tau_dbs = [tau_dbs]

        for path in nue_dbs + tau_dbs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"DB file not found: {path}")

        data_list = []
        pca_features_list = []
        
        # --- TAU ---
        print(f"Loading tau events from {len(tau_dbs)} file(s)...")
        tau_count = 0
        for db in tqdm(tau_dbs, desc="  tau files", unit="file"):
            for eid, rows, meta in tqdm(
                stream_events_with_truth(db, 
                                        energy_threshold_min_tev=energy_threshold_min_tev,
                                        energy_threshold_max_tev=energy_threshold_max_tev),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if max_events_per_class and tau_count >= max_events_per_class:
                    break
                
                # GNN graph
                g = event_to_string_graph(
                    rows, kd_tree, string_ids_geo, meta=meta, charge_threshold=charge_threshold
                )
                g.y = torch.tensor([1.0])
                
                # PCA features
                pca_feats = compute_pca_features(
                    rows, kd_tree, string_ids_geo, meta=meta, charge_threshold=charge_threshold
                )
                
                if pca_feats['is_valid']:
                    data_list.append(g)
                    pca_features_list.append(pca_feats)
                    tau_count += 1
            
            if max_events_per_class and tau_count >= max_events_per_class:
                break
        print(f"  -> Loaded {tau_count} tau events")

        # Nue
        print(f"Loading nue events from {len(nue_dbs)} file(s)...")
        nue_max_events = max_events_per_class
        if max_events_per_class and tau_count < max_events_per_class:
            print(f"  Auto-balancing: limiting nue to {tau_count} events")
            nue_max_events = tau_count
        
        nue_count = 0
        for db in tqdm(nue_dbs, desc="  nue files", unit="file"):
            for eid, rows, meta in tqdm(
                stream_events_with_truth(db,
                                        energy_threshold_min_tev=energy_threshold_min_tev,
                                        energy_threshold_max_tev=energy_threshold_max_tev),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if nue_max_events and nue_count >= nue_max_events:
                    break
                
                g = event_to_string_graph(
                    rows, kd_tree, string_ids_geo, meta=meta, charge_threshold=charge_threshold
                )
                g.y = torch.tensor([0.0])
                
                pca_feats = compute_pca_features(
                    rows, kd_tree, string_ids_geo, meta=meta, charge_threshold=charge_threshold
                )
                
                if pca_feats['is_valid']:
                    data_list.append(g)
                    pca_features_list.append(pca_feats)
                    nue_count += 1
            
            if nue_max_events and nue_count >= nue_max_events:
                break
        print(f"  -> Loaded {nue_count} nue events")

        if len(data_list) == 0:
            raise RuntimeError("Dataset is empty -- check DB paths and table names.")

        print(f"\nDataset summary:")
        print(f"  Tau:  {tau_count}")
        print(f"  Nue:  {nue_count}")
        if tau_count != nue_count:
            ratio = max(tau_count, nue_count) / min(tau_count, nue_count)
            print(f"Ratio: 1:{ratio:.2f}")
        else:
            print(f"Ratio: 1:1 (balanced)")

        print("Collating GNN dataset...")
        self.data, self.slices = self.collate(data_list)
        
        self.pca_features_array = np.array([
            [feats[name] for name in PCA_FEATURE_NAMES]
            for feats in pca_features_list
        ], dtype=np.float32)
        
        self.pca_scaler = StandardScaler()
        self.pca_features_scaled = self.pca_scaler.fit_transform(self.pca_features_array)
        
    def get_pca_features(self, idx):
        return torch.tensor(self.pca_features_scaled[idx], dtype=torch.float)


class HybridGNNClassifier(nn.Module):
    
    def __init__(self, gnn_hidden_dim=64, pca_dim=13, fusion_dim=32):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(N_GNN_FEATURES, gnn_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gnn_hidden_dim),
        )

        self.conv1 = TransformerConv(gnn_hidden_dim,     gnn_hidden_dim, heads=4)
        self.conv2 = TransformerConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)
        self.conv3 = TransformerConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)

        self.bn1 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.act = nn.ReLU()
        self.drop_gnn = nn.Dropout(0.4)
        
        gnn_output_dim = gnn_hidden_dim * 4 + 1

        self.pca_mlp = nn.Sequential(
            nn.Linear(pca_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3),
        )
        pca_output_dim = fusion_dim

        self.fusion = nn.Sequential(
            nn.Linear(gnn_output_dim + pca_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, data, pca_features):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_mlp(x)
        x = self.drop_gnn(self.act(self.bn1(self.conv1(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn2(self.conv2(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn3(self.conv3(x, edge_index))))
        gnn_output = global_mean_pool(x, batch)
        mean_q = global_mean_pool(data.x[:, 6], batch).unsqueeze(1)
        gnn_features = torch.cat([gnn_output, mean_q], dim=1)

        pca_output = self.pca_mlp(pca_features)

        fused = torch.cat([gnn_features, pca_output], dim=1)
        return self.fusion(fused).squeeze(-1)

class HybridDataLoader:
    """Wrapper to yield (batch, pca_features) tuples."""
    
    def __init__(self, gnn_loader, dataset):
        self.gnn_loader = gnn_loader
        self.dataset = dataset
        self.indices = []
        self.current_idx = 0
        
        for batch in gnn_loader:
            # Extract indices from batch (PyG stores them internally)
            batch_size = batch.num_graphs
            self.indices.extend(list(range(self.current_idx, self.current_idx + batch_size)))
            self.current_idx += batch_size
    
    def __iter__(self):
        self.iter_gnn = iter(self.gnn_loader)
        self.batch_counter = 0
        return self
    
    def __next__(self):
        batch = next(self.iter_gnn)
        
        batch_indices = list(range(
            self.batch_counter,
            min(self.batch_counter + batch.num_graphs, len(self.dataset))
        ))
        self.batch_counter += len(batch_indices)
        
        pca_feats = torch.stack([
            self.dataset.get_pca_features(idx)
            for idx in batch_indices
        ])
        
        return batch, pca_feats
    
    def __len__(self):
        return len(self.gnn_loader)

def load_model(model_path, device=None):
    """Load trained HybridGNNClassifier from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HybridGNNClassifier(
        gnn_hidden_dim=64,
        pca_dim=N_PCA_FEATURES,
        fusion_dim=32
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def build_frozen_gnn_embeddings(model, data_list, device, batch_size=64):
    """Pre-compute GNN embeddings for a set of events. Returns (N, gnn_dim) tensor."""
    from torch_geometric.nn import global_mean_pool

    model.eval()
    embeddings = []

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x, edge_index, b = batch.x, batch.edge_index, batch.batch

            x = model.node_mlp(x)
            x = model.drop_gnn(model.act(model.bn1(model.conv1(x, edge_index))))
            x = model.drop_gnn(model.act(model.bn2(model.conv2(x, edge_index))))
            x = model.drop_gnn(model.act(model.bn3(model.conv3(x, edge_index))))
            gnn_out = global_mean_pool(x, b)
            mean_q  = global_mean_pool(batch.x[:, 6], b).unsqueeze(1)
            gnn_features = torch.cat([gnn_out, mean_q], dim=1)  # (B, gnn_dim+1)
            embeddings.append(gnn_features.cpu())

    return torch.cat(embeddings, dim=0)   # (N, gnn_dim+1)


class FusionWrapper(torch.nn.Module):
    """Fusion head only — takes frozen GNN embedding + PCA features."""
    def __init__(self, model, frozen_gnn_embeddings):
        super().__init__()
        self.pca_mlp  = model.pca_mlp
        self.fusion   = model.fusion
        self.frozen   = frozen_gnn_embeddings   # (N, gnn_dim+1), on CPU

    def forward_numpy(self, pca_np):
        """Called by SHAP. pca_np shape: (B, N_PCA_FEATURES)."""
        pca_t = torch.tensor(pca_np, dtype=torch.float32)
        N     = pca_t.shape[0]

        # Repeat or sample frozen GNN embeddings to match batch size.
        # Using the mean embedding keeps GNN contribution constant across
        # all SHAP perturbations — isolates PCA importance cleanly.
        gnn_mean = self.frozen.mean(0, keepdim=True).expand(N, -1)

        pca_out = self.pca_mlp(pca_t)
        fused   = torch.cat([gnn_mean, pca_out], dim=1)
        logits  = self.fusion(fused).squeeze(-1)
        probs   = torch.sigmoid(logits)
        return probs.detach().numpy()


# ── 2. Compute SHAP values ────────────────────────────────────────────────────

def compute_pca_shap(
    model_path,
    db_path,
    geo_path,
    charge_threshold=0.1,
    energy_threshold_min_tev=None,
    energy_threshold_max_tev=None,
    n_background=200,     # background sample for SHAP explainer
    n_explain=500,        # events to explain
    max_events=2000,
    batch_size=64,
    save_dir="./output_hybrid",
):
    import os
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu")   # SHAP runs on CPU
    model, _ = load_model(model_path, device)

    import pandas as pd
    geo = pd.read_csv(geo_path)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)

    # Collect events
    data_list      = []
    pca_feats_list = []

    for eid, rows, meta in tqdm(
        stream_events_with_truth(
            db_path,
            energy_threshold_min_tev=energy_threshold_min_tev,
            energy_threshold_max_tev=energy_threshold_max_tev,
        ),
        unit="ev",
    ):
        if len(data_list) >= max_events:
            break
        pca_f = compute_pca_features(rows, kd_tree, string_ids_geo, meta=meta,
                                     charge_threshold=charge_threshold)
        if not pca_f["is_valid"]:
            continue
        graph = event_to_string_graph(rows, kd_tree, string_ids_geo, meta=meta,
                                      charge_threshold=charge_threshold)
        data_list.append(graph)
        pca_feats_list.append([pca_f[name] for name in PCA_FEATURE_NAMES])

    print(f"Collected {len(data_list)} events")

    pca_array  = np.array(pca_feats_list, dtype=np.float32)
    pca_scaled = StandardScaler().fit_transform(pca_array)

    # Frozen GNN embeddings
    print("Computing frozen GNN embeddings...")
    frozen_embeddings = build_frozen_gnn_embeddings(model, data_list, device, batch_size)

    wrapper = FusionWrapper(model, frozen_embeddings)
    wrapper.eval()

    # Background and explanation sets
    bg_idx  = np.random.choice(len(pca_scaled), size=min(n_background, len(pca_scaled)), replace=False)
    exp_idx = np.random.choice(len(pca_scaled), size=min(n_explain,    len(pca_scaled)), replace=False)

    background = pca_scaled[bg_idx]
    explain_X  = pca_scaled[exp_idx]

    # KernelExplainer works on any black-box function
    print(f"Fitting SHAP KernelExplainer on {len(background)} background samples...")
    explainer   = shap.KernelExplainer(wrapper.forward_numpy, background)

    print(f"Computing SHAP values for {len(explain_X)} events (this takes a few minutes)...")
    shap_values = explainer.shap_values(explain_X, nsamples=256)
    # shap_values shape: (n_explain, N_PCA_FEATURES)

    return shap_values, explain_X, pca_scaled


# ── 3. Plot ───────────────────────────────────────────────────────────────────

def plot_shap(shap_values, explain_X, save_dir="./output_hybrid"):
    import os

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="black")

    # --- Bar plot: mean |SHAP| per feature ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    axes[0].barh(
        [PCA_FEATURE_NAMES[i] for i in order[::-1]],
        mean_abs[order[::-1]],
        color="cyan", alpha=0.85
    )
    axes[0].set_xlabel("Mean |SHAP value|", color="white")
    axes[0].set_title("PCA Feature Importance", color="white")
    axes[0].set_facecolor("black")
    axes[0].tick_params(colors="white")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_color("white")
    axes[0].spines["left"].set_color("white")

    # --- Scatter: SHAP value vs feature value (top 6 features) ---
    top6 = order[:6]
    colors_scatter = ["cyan", "orange", "lime", "magenta", "yellow", "red"]

    for rank, (feat_idx, color) in enumerate(zip(top6, colors_scatter)):
        feat_vals  = explain_X[:, feat_idx]
        shap_vals  = shap_values[:, feat_idx]
        axes[1].scatter(feat_vals, shap_vals, s=4, alpha=0.4, color=color,
                        label=PCA_FEATURE_NAMES[feat_idx])

    axes[1].axhline(0, color="gray", lw=0.8, linestyle="--")
    axes[1].set_xlabel("Feature value (scaled)", color="white")
    axes[1].set_ylabel("SHAP value → P(tau)", color="white")
    axes[1].set_title("SHAP vs Feature Value (top 6)", color="white")
    axes[1].legend(facecolor="black", labelcolor="white", markerscale=3, fontsize=8)
    axes[1].set_facecolor("black")
    axes[1].tick_params(colors="white")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["bottom"].set_color("white")
    axes[1].spines["left"].set_color("white")

    plt.tight_layout()
    out = os.path.join(save_dir, "shap_pca_importance.png")
    plt.savefig(out, dpi=150, facecolor="black")
    plt.close()
    print(f"Saved: {out}")

    # Ranked summary to stdout
    print("\nPCA feature importance ranking:")
    for rank, i in enumerate(order):
        print(f"  {rank+1:2d}. {PCA_FEATURE_NAMES[i]:<25s}  mean|SHAP| = {mean_abs[i]:.5f}")


# ── Usage ─────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     shap_values, explain_X, pca_scaled = compute_pca_shap(
#         model_path               = "/mnt/scratch/baburish/doublepulse/gnn/pca_hybrid/hybrid_weights.pt",
#         db_path                  = "/mnt/research/IceCube/lownutau/taurus_sqlite/nutau/nutau_taurus_5TeV_lowerbound_skimmed_chunk_022859.003469_0000.db",
#         geo_path                 = "/mnt/scratch/baburish/doublepulse/gnn/geometry_clean.csv",
#         charge_threshold         = 0.1,
#         energy_threshold_min_tev = 0.05,
#         n_background             = 50,
#         n_explain                = 200,
#     )
#     plot_shap(shap_values, explain_X)

if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    geo_path   = "/mnt/scratch/baburish/doublepulse/gnn/geometry_clean.csv"
    model_path = "/mnt/scratch/baburish/doublepulse/gnn/pca_hybrid/hybrid_weights.pt"
    nue_db     = "/mnt/research/IceCube/lownutau/taurus_sqlite/nue/nue_taurus_5TeV_lowerbound_skimmed_chunk_022856.002730_0004.db"
    tau_db     = "/mnt/research/IceCube/lownutau/taurus_sqlite/nutau/nutau_taurus_5TeV_lowerbound_skimmed_chunk_022859.003469_0000.db"

    import pandas as pd
    geo = pd.read_csv(geo_path)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)

    device = torch.device("cpu")
    model, _ = load_model(model_path, device)

    # ── Collect events from both classes ──────────────────────────────────────
    data_list      = []
    pca_feats_list = []
    labels         = []

    for db, label in [(nue_db, 0), (tau_db, 1)]:
        count = 0
        for eid, rows, meta in tqdm(
            stream_events_with_truth(db, energy_threshold_min_tev=0.05),
            unit="ev", desc=f"{'nue' if label==0 else 'tau'}",
        ):
            if count >= 1000:
                break
            pca_f = compute_pca_features(rows, kd_tree, string_ids_geo, meta=meta,
                                         charge_threshold=0.1)
            if not pca_f["is_valid"]:
                continue
            graph = event_to_string_graph(rows, kd_tree, string_ids_geo, meta=meta,
                                          charge_threshold=0.1)
            data_list.append(graph)
            pca_feats_list.append([pca_f[name] for name in PCA_FEATURE_NAMES])
            labels.append(label)
            count += 1

    labels    = np.array(labels)
    pca_array = np.array(pca_feats_list, dtype=np.float32)
    pca_scaled = StandardScaler().fit_transform(pca_array)

    print(f"Total events: {len(data_list)}  (nue={( labels==0).sum()}  tau={(labels==1).sum()})")

    # ── Frozen GNN embeddings ─────────────────────────────────────────────────
    print("Computing frozen GNN embeddings...")
    frozen_embeddings = build_frozen_gnn_embeddings(model, data_list, device, batch_size=64)

    wrapper = FusionWrapper(model, frozen_embeddings)
    wrapper.eval()

    # ── SHAP — use 100 background, explain all 400 ────────────────────────────
    n_background = 100
    bg_idx  = np.random.choice(len(pca_scaled), size=n_background, replace=False)
    background  = pca_scaled[bg_idx]

    print(f"Fitting KernelExplainer on {n_background} background samples...")
    explainer   = shap.KernelExplainer(wrapper.forward_numpy, background)

    print(f"Computing SHAP values for {len(pca_scaled)} events...")
    shap_values = explainer.shap_values(pca_scaled, nsamples=128)  # nsamples lowered to avoid OOM
    # shape: (400, N_PCA_FEATURES)

    # ── Split by class ────────────────────────────────────────────────────────
    nue_mask = labels == 0
    tau_mask = labels == 1

    shap_nue = shap_values[nue_mask]
    shap_tau = shap_values[tau_mask]
    X_nue    = pca_scaled[nue_mask]
    X_tau    = pca_scaled[tau_mask]

    # ── Plot ──────────────────────────────────────────────────────────────────
    mean_abs_nue = np.abs(shap_nue).mean(axis=0)
    mean_abs_tau = np.abs(shap_tau).mean(axis=0)
    mean_abs_all = np.abs(shap_values).mean(axis=0)
    order        = np.argsort(mean_abs_all)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="black")

    def style_ax(ax, title):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.set_title(title, color="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    feat_labels = [PCA_FEATURE_NAMES[i] for i in order[::-1]]

    # Panel 1: overall importance
    axes[0].barh(feat_labels, mean_abs_all[order[::-1]], color="cyan", alpha=0.85)
    axes[0].set_xlabel("Mean |SHAP|")
    style_ax(axes[0], "Overall importance")

    # Panel 2: nue vs tau side by side
    y      = np.arange(N_PCA_FEATURES)
    height = 0.35
    axes[1].barh(y + height/2, mean_abs_nue[order[::-1]], height, color="dodgerblue", alpha=0.85, label="nue")
    axes[1].barh(y - height/2, mean_abs_tau[order[::-1]], height, color="orange",     alpha=0.85, label="tau")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(feat_labels)
    axes[1].set_xlabel("Mean |SHAP|")
    axes[1].legend(facecolor="black", labelcolor="white")
    style_ax(axes[1], "nue vs tau")

    # Panel 3: mean signed SHAP (direction of influence)
    mean_signed_nue = shap_nue.mean(axis=0)
    mean_signed_tau = shap_tau.mean(axis=0)
    axes[2].barh(y + height/2, mean_signed_nue[order[::-1]], height, color="dodgerblue", alpha=0.85, label="nue")
    axes[2].barh(y - height/2, mean_signed_tau[order[::-1]], height, color="orange",     alpha=0.85, label="tau")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(feat_labels)
    axes[2].axvline(0, color="gray", lw=0.8, linestyle="--")
    axes[2].set_xlabel("Mean SHAP  (+ → tau, − → nue)")
    axes[2].legend(facecolor="black", labelcolor="white")
    style_ax(axes[2], "Decision direction")

    plt.tight_layout()
    plt.savefig("shap_combined.png", dpi=150, facecolor="black")
    plt.close()
    print("Saved: shap_combined.png")

    # ── Ranked summary ────────────────────────────────────────────────────────
    print(f"\n{'Feature':<25} {'Overall':>10} {'nue':>10} {'tau':>10} {'direction'}")
    print("-" * 65)
    for i in order:
        direction = "→ tau" if shap_values[:, i].mean() > 0 else "→ nue"
        print(f"{PCA_FEATURE_NAMES[i]:<25} {mean_abs_all[i]:>10.5f} {mean_abs_nue[i]:>10.5f} {mean_abs_tau[i]:>10.5f}  {direction}")