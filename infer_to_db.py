"""Score SQLite event files with a trained network and write the per-event scores into the
files' `reco` table as new columns. Works for any trained run (5 TeV or 65 TeV network): the
trainer module, model options, event preprocessing and auxiliary-target scaling are all read
from the run directory.

For every event in `reco`:
  * preprocess exactly as in the run's training (pulse charge threshold, DOM charge cut, minimum
    strings; can be overridden, see below),
  * build the event graph with the run's trainer module and score it with the run's best weights.

Columns added to reco (PREFIX = --prefix, default "gnn"; use e.g. gnn5 / gnn65 to keep the scores of
both networks in one file):
    PREFIX_prob_tau            nutau score (sigmoid of the logit); NULL if the event was not scored
    PREFIX_logit               raw network output
    PREFIX_valid               1 = passed the run's event selection and was scored, 0 = not scored
    PREFIX_n_doms              DOMs left after the charge cuts
  and for runs trained with auxiliary targets:
    PREFIX_pred_log10_energy   predicted log10(primary energy / GeV)
    PREFIX_pred_zenith, PREFIX_pred_azimuth   predicted direction [rad]
plus a table PREFIX_model_info (key, value) with the model run directory, its run_config.json and the
preprocessing used (skip with --no_info).

Safety: each file is copied to --backup_dir before it is modified (unless --no_backup); all changes
to a file are made in one transaction (nothing changes if anything fails); existing PREFIX_ columns
are only replaced with --overwrite.

Runs without run_config.json (trained before it existed, e.g. trained_model/GAT_*): pass --trainer
(e.g. trainer_noedge); the PCA branch and auxiliary head are detected from the weights, the PCA
features are scaled with the run's pca_scaler.joblib, and the event selection is that trainer's own
(>= 4 pulses on >= 2 strings for PCA runs), with --charge_threshold (default 0.1 PE, the train.py default).

Examples:
    python infer_to_db.py --run_dir trained_model5TeV/full_nodist_aux1_bs64 --prefix gnn5  --db file.db
    python infer_to_db.py --run_dir trained_model/GAT_0p3dropout_weightdecay5e4_64batchsize \
        --trainer trainer_noedge --prefix gnn65 --db file.db
"""
import argparse
import datetime
import importlib
import inspect
import json
import os
import shutil
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

import trainer as base   # shared preprocessing / event selection
from evaluate_test import build_model

ANA = os.path.dirname(os.path.abspath(__file__))

PULSE_QUERY = """
SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
       r.cascade_vertex_fit_x, r.cascade_vertex_fit_y, r.cascade_vertex_fit_z, r.cascade_reco_energy_tev
FROM   CleanedROIPulses p JOIN reco r ON p.event_no = r.event_no
ORDER  BY p.event_no
"""


class SerialKD:
    """Same KD-tree query as training, single-threaded (the trainer asks for all cores per event)."""
    def __init__(self, tree):
        self.tree = tree
        self.data = tree.data

    def query(self, pts, workers=1, **kw):
        return self.tree.query(pts, workers=1, **kw)


def stream_events(conn):
    """Yield (event_no, pulse rows [event_no, x, y, z, t, q], meta) grouped by event."""
    current, rows, meta = None, [], {}
    for r in conn.execute(PULSE_QUERY):
        if r[0] != current:
            if rows:
                yield current, rows, meta
            current, rows = r[0], []
            meta = {"cascade_vertex_fit_x": r[6] or 0.0, "cascade_vertex_fit_y": r[7] or 0.0,
                    "cascade_vertex_fit_z": r[8] or 0.0,
                    "cascade_reco_energy_tev": r[9] if r[9] is not None else 0.05}
        rows.append(r[:6])
    if rows:
        yield current, rows, meta


def score_events(conn, model, tm, pre_kw, min_strings, aux_norm, kd, sid_geo, device, batch_size, pca_scaler=None):
    """DataFrame with one row per reco event: event_no, prob_tau, logit, valid, n_doms (+ aux predictions).
    pca_scaler: the run's fitted StandardScaler for models with the PCA branch (None = no PCA branch)."""
    graph_fn = getattr(tm, "event_to_dom_pulse_graph", None) or tm.event_to_dom_graph
    # the trainer module's own preprocessing (legacy modules have no DOM-charge cut)
    pre_fn = tm.preprocess_event_hits
    pre_args = {k: v for k, v in pre_kw.items() if k in inspect.signature(pre_fn).parameters}
    has_aux = getattr(model, "aux_head", None) is not None
    out = {ev: dict(event_no=ev, prob_tau=None, logit=None, valid=0, n_doms=0)
           for (ev,) in conn.execute("SELECT event_no FROM reco")}
    if has_aux:
        for o in out.values():
            o.update(pred_log10_energy=None, pred_zenith=None, pred_azimuth=None)
    pending = []

    def flush():
        if not pending:
            return
        b = Batch.from_data_list([g for _, g in pending]).to(device)
        with torch.no_grad():
            logits, aux = model(b, return_aux=True) if has_aux else (model(b), None)
        logits = logits.cpu().numpy()
        aux = aux.cpu().numpy() if aux is not None else None
        for k, (ev, _) in enumerate(pending):
            o = out[ev]
            o.update(logit=float(logits[k]), prob_tau=float(1 / (1 + np.exp(-logits[k]))), valid=1)
            if aux is not None:
                e = aux[k, 0] * aux_norm[1] + aux_norm[0] if np.all(np.isfinite(aux_norm)) else aux[k, 0]
                o["pred_log10_energy"] = float(e * tm.AUX_LOG10E_STD + tm.AUX_LOG10E_MEAN)
                d = aux[k, 1:4] / max(np.linalg.norm(aux[k, 1:4]), 1e-9)
                o["pred_zenith"] = float(np.arccos(np.clip(d[2], -1, 1)))
                o["pred_azimuth"] = float(np.arctan2(d[1], d[0]) % (2 * np.pi))
        pending.clear()

    for ev, rows, meta in stream_events(conn):
        x, y, z, t, q, s = pre_fn(rows, kd, sid_geo, **pre_args)
        out[ev]["n_doms"] = len(set(zip(s.tolist(), z.tolist()))) if len(t) else 0
        if pca_scaler is not None:
            # PCA runs: selection = the trainer's PCA validity (>= 4 pulses on >= 2 strings)
            pca = tm.compute_pca_features(x, y, z, t, q, s, meta=meta)
            if not pca["is_valid"]:
                continue
            g = graph_fn(x, y, z, t, q, s, meta=meta)
            vals = np.array([[pca[n] for n in tm.PCA_FEATURE_NAMES]], dtype=np.float32)
            g.pca_feats = torch.tensor(pca_scaler.transform(vals), dtype=torch.float)
        elif base.event_is_valid(z, t, s, min_strings=min_strings):
            g = graph_fn(x, y, z, t, q, s, meta=meta)
        else:
            continue
        pending.append((ev, g))
        if len(pending) >= batch_size:
            flush()
    flush()
    return pd.DataFrame(list(out.values()))


def write_to_reco(db, df, prefix, info, overwrite):
    """Add/replace PREFIX_* columns in reco from df (by event_no), in one transaction."""
    cols = [c for c in df.columns if c != "event_no"]
    types = {c: ("INTEGER" if c in ("valid", "n_doms") else "REAL") for c in cols}
    conn = sqlite3.connect(db, isolation_level=None)   # explicit transaction control (DDL included)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = {r[1] for r in conn.execute("PRAGMA table_info(reco)")}
        for c in cols:
            if f"{prefix}_{c}" not in existing:
                conn.execute(f'ALTER TABLE reco ADD COLUMN "{prefix}_{c}" {types[c]}')
        conn.execute(f"CREATE TEMP TABLE scores (event_no INTEGER PRIMARY KEY, "
                     + ", ".join(f'"{c}" {types[c]}' for c in cols) + ")")
        rows = df[["event_no"] + cols].astype(object).where(df[["event_no"] + cols].notna(), None).values.tolist()
        conn.executemany(f"INSERT INTO temp.scores VALUES ({', '.join('?' * (len(cols) + 1))})", rows)
        conn.execute("UPDATE reco SET " + ", ".join(
            f'"{prefix}_{c}" = (SELECT s."{c}" FROM temp.scores s WHERE s.event_no = reco.event_no)' for c in cols))
        if info is not None:
            conn.execute(f'DROP TABLE IF EXISTS "{prefix}_model_info"')
            conn.execute(f'CREATE TABLE "{prefix}_model_info" (key TEXT PRIMARY KEY, value TEXT)')
            conn.executemany(f'INSERT INTO "{prefix}_model_info" VALUES (?, ?)', list(info.items()))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="Trained run (run_config.json, hybrid_weights.pt, eval_val.npz)")
    p.add_argument("--db", nargs="+", required=True, help="SQLite file(s) to score (reco gets new columns)")
    p.add_argument("--prefix", default="gnn", help="Column prefix, e.g. gnn5 / gnn65 for the 5 / 65 TeV networks")
    p.add_argument("--trainer", default=None,
                   help="Trainer module for runs without run_config.json (e.g. trainer_noedge for trained_model/GAT_*)")
    p.add_argument("--charge_threshold", type=float, default=None,
                   help="Pulse charge threshold [PE] for runs without run_config.json (default 0.1, the train.py default)")
    p.add_argument("--overwrite", action="store_true", help="Replace existing PREFIX_ columns")
    p.add_argument("--min_dom_charge", type=float, default=None,
                   help="Override the run's DOM charge cut [PE] (outside the training conditions; recorded in the info table)")
    p.add_argument("--min_strings", type=int, default=None, help="Override the run's minimum number of strings")
    p.add_argument("--backup_dir", default=os.path.join(ANA, "backup", "infer_db"))
    p.add_argument("--no_backup", action="store_true", help="Do not copy the files before modifying them")
    p.add_argument("--no_info", action="store_true", help="Do not write the PREFIX_model_info table")
    p.add_argument("--geo", default=os.path.join(ANA, "geometry_clean.csv"))
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    state = torch.load(os.path.join(args.run_dir, "hybrid_weights.pt"), map_location="cpu")
    cfg_path = os.path.join(args.run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        if args.trainer and args.trainer != cfg["trainer"]:
            sys.exit(f"--trainer {args.trainer} contradicts run_config.json ({cfg['trainer']})")
    else:
        if not args.trainer:
            sys.exit(f"{args.run_dir} has no run_config.json: pass --trainer (e.g. trainer_noedge)")
        cfg = {"trainer": args.trainer, "no_pca": not any(k.startswith("pca_mlp.") for k in state),
               "aux_weight": 1.0 if any(k.startswith("aux_head.") for k in state) else 0.0,
               "charge_threshold": 0.1 if args.charge_threshold is None else args.charge_threshold,
               "min_dom_charge": 0.0, "min_strings": 2, "legacy_run_without_config": True}
        print(f"no run_config.json: using {cfg}")
    tm = importlib.import_module(cfg["trainer"])
    pca_scaler = None
    if not cfg.get("no_pca", False):
        sc_path = os.path.join(args.run_dir, "pca_scaler.joblib")
        if not os.path.exists(sc_path):
            sys.exit("run uses the PCA branch but has no pca_scaler.joblib")
        pca_scaler = joblib.load(sc_path)
    model, n_node = build_model(tm, cfg, state)
    if n_node is not None and n_node != tm.N_GNN_FEATURES:
        sys.exit(f"model was trained with {n_node} node features; the current code builds {tm.N_GNN_FEATURES}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    npz_path = os.path.join(args.run_dir, "eval_val.npz")
    npz = np.load(npz_path) if os.path.exists(npz_path) else {}
    aux_norm = npz["aux_train_norm"] if "aux_train_norm" in getattr(npz, "files", []) else np.array([np.nan, np.nan])

    pre_kw = dict(charge_threshold=cfg.get("charge_threshold", 0.1),
                  min_dom_charge=cfg.get("min_dom_charge", 0.0) if args.min_dom_charge is None else args.min_dom_charge)
    min_strings = cfg.get("min_strings", 2) if args.min_strings is None else args.min_strings
    overridden = args.min_dom_charge is not None or args.min_strings is not None
    if overridden:
        print("WARNING: event selection overridden -- scores are outside the network's training conditions")

    geo = pd.read_csv(args.geo)
    kd_tree, sid_geo = tm.build_geo_dict_kdtree(geo)
    if hasattr(tm, "set_geometry"):
        tm.set_geometry(kd_tree, sid_geo)
    kd = SerialKD(kd_tree)

    for db in args.db:
        if not os.access(db, os.W_OK) or not os.access(os.path.dirname(os.path.abspath(db)), os.W_OK):
            sys.exit(f"{db}: file or its directory not writable")
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as ro:
            existing = {r[1] for r in ro.execute("PRAGMA table_info(reco)")}
            clash = sorted(c for c in existing if c.startswith(f"{args.prefix}_"))
            if clash and not args.overwrite:
                sys.exit(f"{db}: reco already has {clash} (use --overwrite or another --prefix)")
            df = score_events(ro, model, tm, pre_kw, min_strings, aux_norm, kd, sid_geo, device, args.batch_size,
                              pca_scaler)
            n_pulse_only = ro.execute("SELECT COUNT(DISTINCT event_no) FROM CleanedROIPulses "
                                      "WHERE event_no NOT IN (SELECT event_no FROM reco)").fetchone()[0]
        if not args.no_backup:
            os.makedirs(args.backup_dir, exist_ok=True)
            dst = os.path.join(args.backup_dir, f"{os.path.basename(db)}.{datetime.datetime.now():%Y%m%d-%H%M%S}")
            shutil.copy2(db, dst)
            print(f"backup: {dst}")
        info = None if args.no_info else {
            "model_run_dir": os.path.abspath(args.run_dir), "trainer": cfg["trainer"],
            "run_config": json.dumps(cfg), "charge_threshold_pe": str(pre_kw["charge_threshold"]),
            "min_dom_charge_pe": str(pre_kw["min_dom_charge"]), "min_strings": str(min_strings),
            "selection_overridden": str(overridden), "scored_on": f"{datetime.datetime.now():%Y-%m-%dT%H:%M:%S}",
            "n_reco_events": str(len(df)), "n_scored": str(int(df["valid"].sum()))}
        write_to_reco(db, df, args.prefix, info, args.overwrite)
        print(f"{db}: {len(df)} reco events, {int(df['valid'].sum())} scored -> reco.{args.prefix}_* "
              f"({len(df.columns) - 1} columns)" + (f"; {n_pulse_only} events with pulses but no reco row skipped"
                                                    if n_pulse_only else ""))
        print(df.to_string(index=False, max_rows=20))


if __name__ == "__main__":
    main()
