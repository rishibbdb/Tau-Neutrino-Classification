"""
infer.py

Load a trained HybridGNNClassifier + its saved PCA scaler, run inference on
tau/nue test DBs, and produce:
  - ROC, confusion matrix, score distribution, calibration plots
  - SHAP attribution over the 13 PCA_FEATURE_NAMES (GNN branch held fixed
    per event -- see module docstring in compute_shap_values)

Imports feature/graph construction directly from trainer.py to guarantee
exact parity with what produced the checkpoint. Do not reimplement
stream_events_with_truth / compute_pca_features / event_to_dom_graph here --
copies of these have drifted from the training version before and silently
corrupted inference. Likewise, the GNN branch is replayed via
model.get_gnn_embedding() rather than a hand-copied forward pass, for the
same reason.
"""
import argparse
import os
from pathlib import Path
import sqlite3
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

# from trainer import (
#     N_PCA_FEATURES, PCA_FEATURE_NAMES,
#     build_geo_dict_kdtree, HybridNeutrinoDataset, HybridGNNClassifier,
# )
from trainer_noedge import (
    N_PCA_FEATURES, PCA_FEATURE_NAMES,
    build_geo_dict_kdtree, HybridNeutrinoDataset, HybridGNNClassifier,
)


def load_exclusions(path):
    """Parse a file like add_secondary-65tev.log: '#'-prefixed lines and
    blanks are ignored, everything else is treated as a DB path. Matches
    by basename so it's robust to differing directory prefixes."""
    excl = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            excl.add(os.path.basename(line))
    return excl


def apply_exclusions(db_paths, excl):
    kept = [p for p in db_paths if os.path.basename(p) not in excl]
    n_removed = len(db_paths) - len(kept)
    if n_removed:
        print(f"Excluded {n_removed} files via exclusion list")
    return kept

def extract_reco_energy(dataset):
    """Recover physical reco energy (TeV) from the unscaled PCA feature array --
    same event order as run_inference's probs/trues, since both come from the
    same non-shuffled dataset."""
    log_energy_idx = PCA_FEATURE_NAMES.index("log_energy")
    log_energy_raw = dataset.pca_features_array[:, log_energy_idx]  # unscaled
    return 10 ** (log_energy_raw * 6.0)  # inverts log10(E)/6 from compute_pca_features

def load_model_and_scaler(model_path, scaler_path, device):
    model = HybridGNNClassifier(gnn_hidden_dim=64, pca_dim=N_PCA_FEATURES, fusion_dim=32).to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        # a resume-style checkpoint.pt rather than the bare hybrid_weights.pt --
        # unwrap it so you can point diagnostics at an in-progress checkpoint too.
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler

def plot_energy_vs_score(energy, probs, trues, save_dir, threshold=0.5):
    correct = (probs >= threshold).astype(int) == trues.astype(int)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="white")
    for label, name, color in [(0, "nue", "tab:orange"), (1, "tau", "tab:blue")]:
        mask = trues == label
        ax.scatter(energy[mask & correct], probs[mask & correct],
                   s=14, color=color, alpha=0.6, label=f"{name} (correct)")
        ax.scatter(energy[mask & ~correct], probs[mask & ~correct],
                   s=26, facecolors="none", edgecolors=color, linewidths=1.3,
                   label=f"{name} (misclassified)")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Reco energy (TeV)")
    ax.set_ylabel("P(tau)")
    ax.set_title("Score vs. Reco Energy")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "energy_vs_score.png"), dpi=150, facecolor="white")
    plt.close()
    print(f"Saved energy_vs_score.png to {save_dir}")
# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, dataset, device, batch_size=64):
    # pca_feats lives on each Data object now, so a plain DataLoader carries
    # it through batching automatically -- no HybridDataLoader wrapper needed.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
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
            gnn_feat = model.get_gnn_embedding(single)
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

def filter_valid_dbs(db_paths, required_table="reco"):
    valid, skipped = [], []
    for p in db_paths:
        try:
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            has_table = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (required_table,),
            ).fetchone() is not None
            con.close()
        except sqlite3.Error as e:
            print(f"  SKIP (unreadable): {p} ({e})")
            skipped.append(p)
            continue
        if has_table:
            valid.append(p)
        else:
            print(f"  SKIP (no '{required_table}' table): {p}")
            skipped.append(p)
    return valid, skipped

def energy_binned_table(energy, probs, trues, threshold=0.7,
                         bins=(65, 100, 200, 500, 1000, 2500, 6000, np.inf)):
    from sklearn.metrics import roc_auc_score

    bin_idx = np.digitize(energy, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        n = mask.sum()
        if n < 10:
            continue

        y_true = trues[mask]
        y_prob = probs[mask]
        y_pred = (y_prob >= threshold).astype(int)

        n_tau = (y_true == 1).sum()
        n_nue = (y_true == 0).sum()
        tau_frac = n_tau / n if n > 0 else np.nan

        # AUC only defined if both classes present in this bin
        auc = roc_auc_score(y_true, y_prob) if (n_tau > 0 and n_nue > 0) else np.nan

        recovery = (y_pred[y_true == 1] == 1).mean() if n_tau > 0 else np.nan
        rejection = (y_pred[y_true == 0] == 0).mean() if n_nue > 0 else np.nan

        rows.append({
            "bin_low": bins[b],
            "bin_high": bins[b + 1],
            "n_events": n,
            "n_tau": n_tau,
            "n_nue": n_nue,
            "tau_frac": round(tau_frac, 3),
            "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
            "tau_recovery": round(recovery, 4) if not np.isnan(recovery) else np.nan,
            "nue_rejection": round(rejection, 4) if not np.isnan(rejection) else np.nan,
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_model_and_scaler(args.model_path, args.scaler_path, device)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)

    nue_dbs = resolve_db_inputs(args.nue_dbs, args.max_db_files)
    tau_dbs = resolve_db_inputs(args.tau_dbs, args.max_db_files)

    if args.exclude_list:
        excl = load_exclusions(args.exclude_list)
        nue_dbs = apply_exclusions(nue_dbs, excl)
        tau_dbs = apply_exclusions(tau_dbs, excl)

    nue_dbs, nue_skipped = filter_valid_dbs(nue_dbs)
    tau_dbs, tau_skipped = filter_valid_dbs(tau_dbs)

    print(f"Using {len(nue_dbs)} nue DB files ({len(nue_skipped)} skipped)")
    print(f"Using {len(tau_dbs)} tau DB files ({len(tau_skipped)} skipped)")

    dataset = HybridNeutrinoDataset(
        nue_dbs=nue_dbs, tau_dbs=tau_dbs,
        kd_tree=kd_tree, string_ids_geo=string_ids_geo,
        max_events_per_class=args.max_events,
        charge_threshold=args.charge_threshold,
        pca_scaler=scaler,  # reuse training-time scaler -- transform, not fit
    )

    probs, trues = run_inference(model, dataset, device, batch_size=args.batch_size)
    energy = extract_reco_energy(dataset)

    table = energy_binned_table(energy, probs, trues, threshold=args.threshold)
    table.to_csv(os.path.join(args.save_dir, f"energy_binned_metrics_thresh{args.threshold}.csv"), index=False)

    pd.DataFrame({
        "true_label": trues,
        "prob_tau": probs,
        "reco_energy_tev": energy,
    }).to_csv(os.path.join(args.save_dir, "scores.csv"), index=False)

    plot_diagnostics(probs, trues, args.save_dir)
    plot_energy_vs_score(energy, probs, trues, args.save_dir, threshold=args.threshold)

    shap_array, feature_array, event_idx = compute_shap_values(
        model, dataset, device,
        n_events=args.n_shap_events, background_size=args.shap_background,
    )
    plot_shap(shap_array, feature_array, args.save_dir)


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
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for the energy-binned recovery/rejection table.")
    p.add_argument("--shap_background", type=int, default=100)
    p.add_argument("--exclude_list", default=None, help="Text file of DB paths to skip (lines starting with # or blank are ignored).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

#python /mnt/scratch/baburish/doublepulse/gnn/Analysis/inferencing-pca.py --model_path pca_65TeV_16batchsize/hybrid_weights.pt --scaler_path pca_65TeV_16batchsize/pca_scaler.joblib --nue_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nue_test/ --tau_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nutau_test/ --max_db_files 20 --geo geometry_clean.csv --save_dir /trained_model/dom_level_train_GAT_infer

#python inferencing-pca.py --model_path pca_65TeV_16batchsize/hybrid_weights.pt --scaler_path pca_65TeV_16batchsize/pca_scaler.joblib --tau_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nutau_test/nutau_gemini_65TeV_lowerbound_skimmed_chunk_022633.000114_0000999_0000.db_test.db --nue_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nue_test/nue_gemini_65TeV_lowerbound_skimmed_chunk_022612.013261_0000.db_test.db --geo geometry_clean.csv --save_dir ./inference_pca_65TeV_Run2



# python infer.py --model_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/dom_level_train_GAT/hybrid_weights.pt --scaler_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/dom_level_train_GAT/pca_scaler.joblib --nue_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nue_test/nue_gemini_65TeV_lowerbound_skimmed_chunk_02261* --tau_dbs /mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nutau_test/nutau_gemini_65TeV_lowerbound_skimmed_chunk_02263* --max_db_files 10 --geo data/geometry_clean.csv --save_dir trained_model/dom_level_train_GAT_infer_20files



#python infer.py --model_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac/hybrid_weights.pt --scaler_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac/pca_scaler.joblib --nue_dbs /mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nue_test/nue_gemini_65TeV_lowerbound_skimmed_chunk_02261* --tau_dbs /mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nutau_test/nutau_gemini_65TeV_lowerbound_skimmed_chunk_02263* --geo geometry_clean.csv --save_dir trained_model/infer_NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac



# python infer.py --model_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac/hybrid_weights.pt --scaler_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac/pca_scaler.joblib --nue_dbs /mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nue_test/*65TeV* --tau_dbs /mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nutau_test/*65TeV* --geo geometry_clean.csv --save_dir trained_model/infer_NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac --max_events 10000


# python infer.py --model_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac/hybrid_weights.pt --scaler_path /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac/pca_scaler.joblib --nue_dbs /mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nue_test/*65TeV* --tau_dbs /mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nutau_test/*65TeV* --geo geometry_clean.csv --save_dir trained_model/infer_NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac --max_events 10000 --exclude_list /mnt/scratch/baburish/doublepulse/gnn/Analysis/gemini/add_secondary-65tev.log