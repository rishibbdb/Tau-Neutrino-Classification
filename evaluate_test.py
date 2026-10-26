"""Evaluate a trained run on an independent TEST sample (the Virgo *_test directories).

Rebuilds the test dataset with exactly the run's preprocessing (trainer module, charge threshold,
DOM charge cut, minimum strings) but WITHOUT 1:1 balancing, so the test sample keeps its natural
composition. Loads the run's best weights and writes, into --out_dir:

    eval_test.npz   same keys as the run's eval_val.npz (preds, trues, y_aux, aux_pred, ...), so
                    plot_eval.py --eval_file eval_test.npz can plot it
    scores.csv      true_label, prob_tau, reco_energy_tev, true_log10E, true_coszen
                    (format of plot_infer.ipynb / notebooks/abcd_threshold.ipynb)

Example:
    python evaluate_test.py --run_dir virgo_65TeV/virgo65_gat_arch_feats_aux1 \
        --tau_dbs data/combined_nutau_65TeV_virgo_test.db --nue_dbs data/combined_nue_65TeV_virgo_test.db \
        --cache_dir dataset_cache_65TeV --out_dir virgo_65TeV/virgo65_gat_arch_feats_aux1/test
"""
import argparse
import importlib
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from train import evaluate_with_aux


def build_model(tm, cfg, state):
    """Reconstruct the run's model from run_config.json and its saved weights."""
    params = inspect.signature(tm.HybridGNNClassifier.__init__).parameters
    kw = {"use_pca": not cfg.get("no_pca", False)}
    if cfg.get("aux_weight", 0) > 0:
        kw["n_aux"] = tm.N_AUX
    if "readout" in params:   # trainer_domenc options
        kw.update(readout=cfg.get("gat_readout", "sum"), residual=cfg.get("gat_residual", False),
                  bright_feats=cfg.get("bright_feats", False), extra_edges=cfg.get("extra_edges", False),
                  full_graph=cfg.get("full_graph", False), time_unit_ns=cfg.get("time_unit_ns") or None,
                  pulse_act=cfg.get("pulse_act", "relu"))
        if cfg.get("use_image") or cfg.get("image_only"):
            kw.update(use_image=True, use_gnn=not cfg.get("image_only", False))
    if "n_node_features" in params:
        n_in = state["node_mlp.0.weight"].shape[1]
        if hasattr(tm, "node_features_in_weights"):
            kw["n_node_features"] = tm.node_features_in_weights(state, bright_feats=cfg.get("bright_feats", False))
        else:
            kw["n_node_features"] = int(n_in)
    model = tm.HybridGNNClassifier(gnn_hidden_dim=64, pca_dim=tm.N_PCA_FEATURES, fusion_dim=32, **kw)
    model.load_state_dict(state)
    return model, kw.get("n_node_features")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="Trained run (run_config.json, hybrid_weights.pt)")
    p.add_argument("--tau_dbs", nargs="+", required=True)
    p.add_argument("--nue_dbs", nargs="+", required=True)
    p.add_argument("--cache_dir", required=True, help="Dataset cache dir (test caches get their own key)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--geo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry_clean.csv"))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--balance", action="store_true", help="Balance the test sample 1:1 (default: natural composition)")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if os.path.exists(os.path.join(args.out_dir, "eval_test.npz")):
        sys.exit(f"{args.out_dir}/eval_test.npz exists -- not overwriting")
    cfg = json.load(open(os.path.join(args.run_dir, "run_config.json")))
    if not cfg.get("no_pca", False):
        sys.exit("run uses the PCA branch: test evaluation needs its fitted PCA scaler, not supported here")
    tm = importlib.import_module(cfg["trainer"])
    print(f"run {args.run_dir} | trainer {cfg['trainer']}")

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = tm.build_geo_dict_kdtree(geo)
    build_kw = dict(charge_threshold=cfg.get("charge_threshold", 0.1))
    if cfg.get("min_dom_charge", 0) > 0:
        build_kw["min_dom_charge"] = cfg["min_dom_charge"]
    if cfg.get("min_strings", 2) != 2:
        build_kw["min_strings"] = cfg["min_strings"]
    ds = tm.load_or_build_dataset(cache_dir=args.cache_dir, nue_dbs=args.nue_dbs, tau_dbs=args.tau_dbs,
                                  kd_tree=kd_tree, string_ids_geo=string_ids_geo, max_events_per_class=None,
                                  balance=args.balance, **build_kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(os.path.join(args.run_dir, "hybrid_weights.pt"), map_location="cpu")
    model, n_node = build_model(tm, cfg, state)
    if n_node is not None and n_node != ds._data.x.shape[1]:
        sys.exit(f"model expects {n_node} node features, test dataset has {ds._data.x.shape[1]} "
                 "(run trained before distance_to_cascade was removed?)")
    model.to(device).eval()

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
    criterion = torch.nn.BCEWithLogitsLoss()
    _, auc, preds, trues, _, y_aux, aux_pred = evaluate_with_aux(model, loader, criterion, device)

    run_npz = np.load(os.path.join(args.run_dir, "eval_val.npz"))
    norm = run_npz["aux_train_norm"] if "aux_train_norm" in run_npz.files else np.array([np.nan, np.nan])
    if aux_pred is not None and np.all(np.isfinite(norm)):
        aux_pred[:, 0] = aux_pred[:, 0] * norm[1] + norm[0]       # back to cached target units
    mean_e, std_e = getattr(tm, "AUX_LOG10E_MEAN", np.nan), getattr(tm, "AUX_LOG10E_STD", np.nan)

    os.makedirs(args.out_dir, exist_ok=True)
    extra = {}
    if y_aux is not None:
        extra["y_aux"] = y_aux
    if aux_pred is not None:
        extra["aux_pred"] = aux_pred
    np.savez(os.path.join(args.out_dir, "eval_test.npz"), preds=preds, trues=trues,
             best_epoch=run_npz["best_epoch"], aux_log10e_mean=mean_e, aux_log10e_std=std_e,
             aux_train_norm=norm, **extra)

    # per-event table: reco energy from the (constant per event) log_energy node feature
    names = list(tm.GNN_FEATURE_NAMES)
    if ds._data.x.shape[1] == len(names) + 1:
        names = names[:11] + ["distance_to_cascade"] + names[11:]
    first_node = ds.slices["x"][:-1]
    log_e_feat = ds._data.x[first_node, names.index("log_energy")].numpy()
    out = pd.DataFrame({"true_label": trues.astype(int), "prob_tau": preds,
                        "reco_energy_tev": 10 ** (6 * log_e_feat)})
    if y_aux is not None:
        out["true_log10E"] = y_aux[:, 0] * std_e + mean_e
        out["true_coszen"] = y_aux[:, 3]
    out.to_csv(os.path.join(args.out_dir, "scores.csv"), index=False)
    print(f"test events: {len(trues)} (nutau {int(trues.sum())}, nue {int(len(trues) - trues.sum())}) | "
          f"test AUC {auc:.4f} | written to {args.out_dir}")


if __name__ == "__main__":
    main()
