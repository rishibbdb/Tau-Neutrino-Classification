import hashlib
import os
import random
import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch_cluster import knn_graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool

GNN_FEATURE_NAMES = [
    "x_norm", "y_norm", "z_norm", "q_log", "q_frac",
    "t_first", "t_mean", "t_spread", "n_pulses",
    "dom_depth_rank", "z_rel", "distance_to_cascade", "log_energy"
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
    """Single-point nearest-string lookup. Kept for compatibility with other
    scripts; the dataset-building path below uses the batched version."""
    distance, idx = kd_tree.query([x, y, z])
    if distance > max_distance:
        return -1
    return int(string_ids[idx])


def map_pulses_to_strings(x, y, z, kd_tree, string_ids_geo, max_distance=50.0):
    """Vectorized string mapping: one batched KDTree query for all hits in
    an event instead of one query per hit."""
    points = np.column_stack([x, y, z])
    distances, idx = kd_tree.query(points, workers=-1)
    strings = string_ids_geo[idx]
    return np.where(distances <= max_distance, strings, -1).astype(int)


def stream_events_with_truth(db_file):
    """Stream events row by row from SQLite, joining truth info.

    No energy/vertex columns required from truth -- meta is returned empty,
    and downstream code (preprocess_event_hits / compute_pca_features /
    event_to_dom_graph) already falls back to its defaults (log_energy from
    0.05 TeV, vertex at origin) when meta doesn't carry those keys.
    """
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"DB file not found: {db_file}")

    conn = sqlite3.connect(db_file)

    query = """
    SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
           r.cascade_vertex_fit_x, r.cascade_vertex_fit_y, r.cascade_vertex_fit_z,
           r.cascade_reco_energy_tev
    FROM   CleanedROIPulses p
    JOIN   truth t ON p.event_no = t.event_no
    JOIN   reco  r ON p.event_no = r.event_no
    ORDER  BY p.event_no
    """

    cursor        = conn.cursor()
    cursor.execute(query)
    current_event = None
    buffer        = []
    current_meta    = {}

    for row in cursor:
        event_no = row[0]
        pulse    = row[:6]
        meta     = {
            "cascade_vertex_fit_x": row[6],
            "cascade_vertex_fit_y": row[7],
            "cascade_vertex_fit_z": row[8],
            "cascade_reco_energy_tev": row[9],
        }
        if current_event is None:
            current_event = event_no
            current_meta  = meta

        if event_no != current_event:
            yield current_event, buffer, current_meta
            buffer        = []
            current_event = event_no
            current_meta  = meta

        buffer.append(pulse)

    if buffer:
        yield current_event, buffer, current_meta

    conn.close()


def preprocess_event_hits(rows, kd_tree, string_ids_geo,
                           charge_threshold=0.0, time_window=1500.0):
    """Parse raw pulse rows once and return filtered, string-tagged hit
    arrays -- shared by compute_pca_features and event_to_dom_graph so the
    time/charge masking and KDTree string-mapping aren't each done twice
    per event."""
    rows = np.array(rows, dtype=object)
    x = rows[:, 1].astype(float)
    y = rows[:, 2].astype(float)
    z = rows[:, 3].astype(float)
    t = rows[:, 4].astype(float)
    q = rows[:, 5].astype(float)

    if len(t) == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty.astype(int)

    t0   = t.min()
    mask = (t >= t0) & (t <= t0 + time_window)
    x, y, z, t, q = x[mask], y[mask], z[mask], t[mask] - t0, q[mask]

    q_mask = q > charge_threshold
    x, y, z, t, q = x[q_mask], y[q_mask], z[q_mask], t[q_mask], q[q_mask]

    if len(t) == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty.astype(int)

    string_ids = map_pulses_to_strings(x, y, z, kd_tree, string_ids_geo)
    valid = string_ids >= 0
    return x[valid], y[valid], z[valid], t[valid], q[valid], string_ids[valid]


def compute_pca_features(x_hits, y_hits, z_hits, t_hits, q_hits, string_ids, meta=None):
    """Compute PCA-informed features from preprocessed hit arrays. Returns dict with 13 features."""
    if len(t_hits) < 4 or len(np.unique(string_ids)) < 2:
        return {
            'is_valid': False,
            'pc1_var': 0.0, 'pc2_var': 0.0, 'pc3_var': 1.0,
            'elongation': 1.0, 'depth_loading': 0.0, 'time_loading': 0.0,
            'charge_loading': 0.0, 'n_strings': 0.0, 'n_doms': 0.0, 'n_hits': 0.0,
            'total_charge': 0.0, 'log_energy': 0.0, 'pc1_pc2_sum': 0.0
        }

    X_spatial = np.column_stack([x_hits, y_hits, z_hits])
    charges = q_hits
    weights = charges / charges.sum()
    X_centered = X_spatial - np.average(X_spatial, weights=weights, axis=0)
    cov_weighted = np.cov(X_centered.T, aweights=weights)
    eigenvals_spatial, eigenvecs_spatial = np.linalg.eigh(cov_weighted)
    idx = np.argsort(eigenvals_spatial)[::-1]
    eigenvals_spatial = eigenvals_spatial[idx]

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


def event_to_dom_graph(x_hits, y_hits, z_hits, t_hits, q_hits, string_ids,
                        meta=None, k=8, t_scale=500.0, xyz_scale=500.0):
    """Convert preprocessed per-event hit arrays into a DOM-level PyG Data
    object. One node per DOM (not per string) -- pulses landing on the same
    (dom_x, dom_y, dom_z) are aggregated into that DOM's node, so per-DOM
    timing/charge structure and depth ordering survive instead of being
    pooled into string-level summaries. k-NN edges in real 3D space
    naturally favor same-string vertical neighbors (~17 m DOM spacing)
    over cross-string neighbors (~125 m string spacing).
    """

    def _fallback():
        return Data(
            x          = torch.zeros((1, N_GNN_FEATURES), dtype=torch.float),
            edge_index = torch.zeros((2, 0),              dtype=torch.long),
        )

    if len(t_hits) < 2:
        return _fallback()

    q_total    = q_hits.sum() + 1e-6
    z_q_center = np.sum(q_hits * z_hits) / q_total

    cascade_reco_energy_tev = float(meta.get("cascade_reco_energy_tev", 0.05)) if meta else 0.05
    log_energy = np.log10(max(cascade_reco_energy_tev, 0.05)) / 6.0

    cascade_vertex_x_norm = (float(meta.get("cascade_vertex_fit_x", 0.0)) if meta else 0.0) / xyz_scale
    cascade_vertex_y_norm = (float(meta.get("cascade_vertex_fit_y", 0.0)) if meta else 0.0) / xyz_scale
    cascade_vertex_z_norm = (float(meta.get("cascade_vertex_fit_z", 0.0)) if meta else 0.0) / xyz_scale

    dom_keys    = list(zip(string_ids.tolist(), z_hits.tolist()))
    unique_doms = sorted(set(dom_keys))
    dom_index   = {key: i for i, key in enumerate(unique_doms)}

    depth_rank = {}
    for s in np.unique(string_ids):
        z_on_string = sorted({z for (sid, z) in unique_doms if sid == s})
        n = len(z_on_string)
        for rank, z in enumerate(z_on_string):
            depth_rank[(s, z)] = rank / max(n - 1, 1)

    node_feats = [None] * len(unique_doms)
    for (sid, z), idx in dom_index.items():
        m = (string_ids == sid) & (z_hits == z)
        x_d = x_hits[m][0]
        y_d = y_hits[m][0]
        t_d, q_d = t_hits[m], q_hits[m]
        q_sum_d  = q_d.sum() + 1e-6

        node_feats[idx] = [
            x_d / xyz_scale,
            y_d / xyz_scale,
            z / xyz_scale,
            np.log1p(q_sum_d),
            q_sum_d / q_total,
            float(t_d.min()) / t_scale,
            float(np.sum(q_d * t_d) / q_sum_d) / t_scale,
            float(t_d.std()) / t_scale if len(t_d) > 1 else 0.0,
            len(t_d) / 10.0,
            depth_rank[(sid, z)],
            (z - z_q_center) / xyz_scale,
            np.sqrt((x_d / xyz_scale - cascade_vertex_x_norm) ** 2 +
                    (y_d / xyz_scale - cascade_vertex_y_norm) ** 2 +
                    (z   / xyz_scale - cascade_vertex_z_norm) ** 2),
            log_energy,
        ]

    node_feats = np.array(node_feats, dtype=np.float32)
    x_tensor   = torch.tensor(node_feats, dtype=torch.float)
    pos        = x_tensor[:, [0, 1, 2]]

    k_actual = min(k, x_tensor.size(0) - 1)
    if k_actual < 1:
        return _fallback()
    edge_index = knn_graph(pos, k=k_actual, loop=False)

    return Data(x=x_tensor, edge_index=edge_index)


class HybridNeutrinoDataset(InMemoryDataset):
    def __init__(self, nue_dbs, tau_dbs, kd_tree, string_ids_geo,
                 max_events_per_class=None, charge_threshold=0.0,
                 pca_scaler=None):
        """
        pca_scaler: pass a pre-fit StandardScaler (e.g. from training) to transform
        with it instead of fitting a new one. Leave None to fit-and-transform
        (only appropriate for the training set).
        """
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
                stream_events_with_truth(db),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if max_events_per_class and tau_count >= max_events_per_class:
                    break

                x, y, z, t, q, string_ids = preprocess_event_hits(
                    rows, kd_tree, string_ids_geo, charge_threshold=charge_threshold
                )

                g = event_to_dom_graph(x, y, z, t, q, string_ids, meta=meta)
                g.y = torch.tensor([1.0])

                pca_feats = compute_pca_features(x, y, z, t, q, string_ids, meta=meta)

                if pca_feats['is_valid']:
                    data_list.append(g)
                    pca_features_list.append(pca_feats)
                    tau_count += 1

            if max_events_per_class and tau_count >= max_events_per_class:
                break
        print(f"  -> Loaded {tau_count} tau events")

        # --- NUE ---
        print(f"Loading nue events from {len(nue_dbs)} file(s)...")
        nue_max_events = max_events_per_class
        if max_events_per_class and tau_count < max_events_per_class:
            print(f"  Auto-balancing: limiting nue to {tau_count} events")
            nue_max_events = tau_count

        nue_count = 0
        for db in tqdm(nue_dbs, desc="  nue files", unit="file"):
            for eid, rows, meta in tqdm(
                stream_events_with_truth(db),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if nue_max_events and nue_count >= nue_max_events:
                    break

                x, y, z, t, q, string_ids = preprocess_event_hits(
                    rows, kd_tree, string_ids_geo, charge_threshold=charge_threshold
                )

                g = event_to_dom_graph(x, y, z, t, q, string_ids, meta=meta)
                g.y = torch.tensor([0.0])

                pca_feats = compute_pca_features(x, y, z, t, q, string_ids, meta=meta)

                if pca_feats['is_valid']:
                    data_list.append(g)
                    pca_features_list.append(pca_feats)
                    nue_count += 1

            if nue_max_events and nue_count >= nue_max_events:
                break
        print(f"  -> Loaded {nue_count} nue events")

        if tau_count != nue_count:
            n_min = min(tau_count, nue_count)
            print(f"Class imbalance detected (tau={tau_count}, nue={nue_count}) "
                  f"-- balancing both to {n_min}")

            tau_indices = list(range(tau_count))                      # tau entries are first
            nue_indices = list(range(tau_count, tau_count + nue_count))  # nue entries follow

            rng = random.Random(42)
            if tau_count > n_min:
                tau_indices = rng.sample(tau_indices, n_min)
            if nue_count > n_min:
                nue_indices = rng.sample(nue_indices, n_min)

            keep_indices = sorted(tau_indices + nue_indices)
            data_list = [data_list[i] for i in keep_indices]
            pca_features_list = [pca_features_list[i] for i in keep_indices]

            tau_count = len(tau_indices)
            nue_count = len(nue_indices)

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

        pca_features_array = np.array([
            [feats[name] for name in PCA_FEATURE_NAMES]
            for feats in pca_features_list
        ], dtype=np.float32)

        if pca_scaler is None:
            self.pca_scaler = StandardScaler()
            pca_features_scaled = self.pca_scaler.fit_transform(pca_features_array)
        else:
            self.pca_scaler = pca_scaler
            pca_features_scaled = self.pca_scaler.transform(pca_features_array)

        self.pca_features_array = pca_features_array
        self.pca_features_scaled = pca_features_scaled

        for g, feats_row in zip(data_list, pca_features_scaled):
            g.pca_feats = torch.tensor(feats_row, dtype=torch.float).unsqueeze(0)

        print("Collating dataset...")
        self.data, self.slices = self.collate(data_list)

    def get_pca_features(self, idx):
        """Kept for inspection/debugging. Not needed for training --
        batch.pca_feats is populated automatically by PyG's DataLoader."""
        return torch.tensor(self.pca_features_scaled[idx], dtype=torch.float)


def _dataset_cache_key(nue_dbs, tau_dbs, max_events_per_class, charge_threshold):
    key_str = repr({
        "nue_dbs": sorted(nue_dbs),
        "tau_dbs": sorted(tau_dbs),
        "max_events_per_class": max_events_per_class,
        "charge_threshold": charge_threshold,
    })
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def load_or_build_dataset(cache_dir, nue_dbs, tau_dbs, kd_tree, string_ids_geo,
                           max_events_per_class=None, charge_threshold=0.0):
    """Build HybridNeutrinoDataset once and cache it to disk, keyed by the
    construction arguments -- repeat runs with the same DB paths/settings
    load instantly instead of re-running SQL streaming + PCA + graph
    building. A different charge_threshold, max_events, or DB list gets its
    own cache file rather than silently reusing a stale one."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _dataset_cache_key(nue_dbs, tau_dbs, max_events_per_class, charge_threshold)
    cache_path = os.path.join(cache_dir, f"dataset_{key}.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached dataset: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"No cache at {cache_path} -- building dataset from scratch")
    dataset = HybridNeutrinoDataset(
        nue_dbs=nue_dbs, tau_dbs=tau_dbs,
        kd_tree=kd_tree, string_ids_geo=string_ids_geo,
        max_events_per_class=max_events_per_class,
        charge_threshold=charge_threshold,
    )
    torch.save(dataset, cache_path)
    print(f"Cached dataset to {cache_path}")
    return dataset

class HybridGNNClassifier(nn.Module):
    def __init__(self, gnn_hidden_dim=64, pca_dim=13, fusion_dim=32):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(N_GNN_FEATURES, gnn_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gnn_hidden_dim),
        )

        self.conv1 = GATConv(gnn_hidden_dim,     gnn_hidden_dim, heads=4)
        self.conv2 = GATConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)
        self.conv3 = GATConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)

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
    def forward(self, data):
        gnn_features = self.get_gnn_embedding(data)
        pca_output = self.pca_mlp(data.pca_feats)
        fused = torch.cat([gnn_features, pca_output], dim=1)
        return self.fusion(fused).squeeze(-1)

    def get_gnn_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_mlp(x)
        x = self.drop_gnn(self.act(self.bn1(self.conv1(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn2(self.conv2(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn3(self.conv3(x, edge_index))))
        gnn_output = global_add_pool(x, batch)
        mean_q = global_mean_pool(data.x[:, 3], batch).unsqueeze(1)  # index 3 == q_log
        return torch.cat([gnn_output, mean_q], dim=1)


def evaluate_loader(model, loader, criterion, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    preds, trues = [], []
    total_loss   = 0.0

    with torch.no_grad():
        for batch in loader:
            batch   = batch.to(device)
            logits  = model(batch)
            targets = batch.y.view(-1)
            loss    = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            trues.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) >= 2 else float("nan")
    return avg_loss, auc, np.array(preds), np.array(trues)