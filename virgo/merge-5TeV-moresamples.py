import os
import glob
import random
import sqlite3
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

OFFSET = 1_000_000


def _eligible_pool(all_files, start_index, min_pe):
    """Pass 1: events passing the same cuts as the full combine, with their true
    direction/energy. Reads only the small reco/truth tables.
    Returns a DataFrame [file_index, event_no, coszen, log_e], file by file in
    ascending event_no."""
    parts = []
    for i, full_path in enumerate(all_files):
        conn = sqlite3.connect(f"file:{full_path}?mode=ro", uri=True)
        reco_ok = {r[0] for r in conn.execute(
            "SELECT event_no FROM reco WHERE brightest_om_brightest_string_PE > ?;", (min_pe,))}
        truth = pd.read_sql_query(
            "SELECT event_no, zenith, primary_energy FROM truth WHERE seed_string IS NOT NULL;", conn)
        conn.close()
        truth = truth[truth["event_no"].isin(reco_ok)].sort_values("event_no")
        print(f"  {os.path.basename(full_path)}: {len(truth)} eligible events")
        parts.append(pd.DataFrame({
            "file_index": start_index + i,
            "event_no": truth["event_no"].to_numpy(),
            "coszen": np.cos(truth["zenith"].to_numpy()),
            "log_e": np.log10(truth["primary_energy"].to_numpy()),
        }))
    return pd.concat(parts, ignore_index=True)


def _write_events(all_files, output_db, pid, start_index, sampled):
    """Pass 2: copy only the (file_index, event_no) pairs in `sampled` from each
    source into output_db, with the usual pid overwrite and event_no offset.
    The event list goes into a temp table, so each source table is scanned once
    (no huge IN (...) parameter lists) and never loaded whole into memory."""
    combined_conn = sqlite3.connect(output_db)
    valid_files, skipped_empty = [], []

    for i, full_path in enumerate(tqdm(all_files, desc=os.path.basename(output_db))):
        file_index = start_index + i
        events = sorted(int(ev) for fi, ev in sampled if fi == file_index)
        if not events:
            skipped_empty.append(full_path)
            continue
        print(f"  {os.path.basename(full_path)}: sampled {len(events)} events")

        conn = sqlite3.connect(f"file:{full_path}?mode=ro", uri=True)
        conn.execute("CREATE TEMP TABLE pick(event_no INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO pick VALUES (?)", [(e,) for e in events])
        dfs = {
            table: pd.read_sql_query(
                f"SELECT * FROM {table} WHERE event_no IN (SELECT event_no FROM temp.pick);", conn)
            for table in ("truth", "reco", "CleanedROIPulses")
        }
        conn.close()

        dfs["truth"]["pid"] = pid
        event_offset = file_index * OFFSET
        for table, df in dfs.items():
            df["event_no"] = df["event_no"] + event_offset
            df.to_sql(table, combined_conn, if_exists="append", index=False)
        valid_files.append(full_path)

    combined_conn.close()
    return valid_files, skipped_empty


def _sample_dbs(all_files, output_db, pid, start_index, min_pe, n_events, seed):
    """Randomly sample n_events across all_files (same cuts as the full combine)
    and write only those events into output_db."""
    pool_df = _eligible_pool(all_files, start_index, min_pe)
    pool = list(zip(pool_df["file_index"].tolist(), pool_df["event_no"].tolist()))

    if n_events > len(pool):
        print(f"Requested {n_events} events but only {len(pool)} eligible; taking all.")
        n_events = len(pool)
    sampled = random.Random(seed).sample(pool, n_events)

    valid_files, skipped_empty = _write_events(all_files, output_db, pid, start_index, sampled)
    print(f"\n[{output_db}] Sampled {n_events} events from {len(valid_files)} files (seed={seed})")
    return valid_files, skipped_empty, start_index + len(all_files)


def combine_matched(specs, n_events, seed=42, min_pe=10, bins=(20, 20)):
    """Sample n_events per flavour such that all flavours share the same
    distribution of true (cos zenith, log10 primary energy).

    specs: list of dicts with db_dir, patterns, output_db, pid (one per flavour).
    Events are binned in `bins` = (cos zenith bins on [-1, 1], log10 E
    quantile bins of the combined pool); every flavour gets the same number of
    events per bin, proportional to the smallest flavour count in that bin.
    Truth is only used to choose events -- nothing is added to the output DBs.
    """
    pools = []
    for spec in specs:
        if os.path.exists(spec["output_db"]):
            os.remove(spec["output_db"])
        files = sorted(set().union(*[glob.glob(os.path.join(spec["db_dir"], p)) for p in spec["patterns"]]))
        print(f"[{spec['output_db']}] Found {len(files)} matching files")
        spec["files"] = files
        pools.append(_eligible_pool(files, 0, min_pe))

    all_pool = pd.concat(pools, ignore_index=True)
    cz_edges = np.linspace(-1, 1, bins[0] + 1)
    le_edges = np.quantile(all_pool["log_e"], np.linspace(0, 1, bins[1] + 1))
    for p in pools:
        cz_bin = np.clip(np.digitize(p["coszen"], cz_edges) - 1, 0, bins[0] - 1)
        le_bin = np.clip(np.digitize(p["log_e"], le_edges) - 1, 0, bins[1] - 1)
        p["cell"] = cz_bin * bins[1] + le_bin

    counts = pd.concat([p["cell"].value_counts() for p in pools], axis=1).fillna(0).min(axis=1).astype(int)
    counts = counts[counts > 0]
    total = int(counts.sum())
    if n_events > total:
        print(f"Requested {n_events} matched events per flavour but only {total} available; taking {total}.")
        n_events = total
    # per-cell quota: proportional to the common (min) count, summing exactly to n_events
    raw = counts * n_events / total
    quota = np.floor(raw).astype(int)
    remainder = n_events - int(quota.sum())
    quota[(raw - quota).sort_values(ascending=False).index[:remainder]] += 1

    rng = np.random.default_rng(seed)
    for spec, p in zip(specs, pools):
        picked = []
        for cell, k in quota[quota > 0].items():
            idx = p.index[p["cell"] == cell]
            picked.append(p.loc[rng.choice(idx, k, replace=False), ["file_index", "event_no"]])
        picked = pd.concat(picked)
        sampled = list(zip(picked["file_index"].tolist(), picked["event_no"].tolist()))
        valid_files, _ = _write_events(spec["files"], spec["output_db"], spec["pid"], 0, sampled)
        print(f"[{spec['output_db']}] wrote {len(sampled)} kinematically matched events "
              f"from {len(valid_files)} files (bins={bins}, seed={seed})")


def combine_dbs(db_dir, patterns, output_db, pid, start_index=0, append=False, min_pe=10,
                n_events=None, seed=42):
    """Combine all files matching any of `patterns` in db_dir into output_db.

    start_index: file_index to start counting from (so event_no offsets don't
                 collide with a previous call into the same output_db).
    append: if False (default), wipes output_db first. Set True to add into
            an existing output_db from a prior call.
    n_events: if set, write only a random sample of n_events (drawn across all
              matching files, with `seed`) instead of every event.
    """
    if not append and os.path.exists(output_db):
        os.remove(output_db)

    all_files = sorted(set().union(
        *[glob.glob(os.path.join(db_dir, p)) for p in patterns]
    ))
    print(f"[{output_db}] Found {len(all_files)} matching files across patterns={patterns}")

    if n_events is not None:
        return _sample_dbs(all_files, output_db, pid, start_index, min_pe, n_events, seed)

    combined_conn = sqlite3.connect(output_db)
    valid_files, skipped_empty = [], []

    for i, full_path in enumerate(tqdm(all_files, desc=os.path.basename(output_db))):
        file_index = start_index + i
        try:
            conn = sqlite3.connect(full_path)
            try:
                df_truth = pd.read_sql_query("SELECT * FROM truth;", conn)
            except Exception:
                df_truth = pd.DataFrame()
            try:
                df_pulses = pd.read_sql_query("SELECT * FROM CleanedROIPulses;", conn)
            except Exception:
                df_pulses = pd.DataFrame()
            try:
                df_reco = pd.read_sql_query("SELECT * FROM reco;", conn)
            except Exception:
                df_reco = pd.DataFrame()
            conn.close()

            if df_reco.empty:
                skipped_empty.append(full_path)
                continue
            df_reco = df_reco[df_reco["brightest_om_brightest_string_PE"] > min_pe]
            keep = df_reco["event_no"].unique()
            df_truth = df_truth[df_truth["event_no"].isin(keep)]
            if not df_pulses.empty:
                df_pulses = df_pulses[df_pulses["event_no"].isin(keep)]

            if df_truth.empty and df_pulses.empty and df_reco.empty:
                skipped_empty.append(full_path)
                continue

            if not df_truth.empty and "seed_string" in df_truth.columns:
                df_truth = df_truth.dropna(subset=["seed_string"])

            if df_truth.empty:
                skipped_empty.append(full_path)
                continue

            df_truth["pid"] = pid

            event_offset = file_index * OFFSET
            df_truth["event_no"] = df_truth["event_no"] + event_offset
            if not df_pulses.empty:
                df_pulses["event_no"] = df_pulses["event_no"] + event_offset
            if not df_reco.empty:
                df_reco["event_no"] = df_reco["event_no"] + event_offset
            valid_files.append(full_path)

            df_truth.to_sql("truth", combined_conn, if_exists="append", index=False)
            if not df_pulses.empty:
                df_pulses.to_sql("CleanedROIPulses", combined_conn, if_exists="append", index=False)
            if not df_reco.empty:
                df_reco.to_sql("reco", combined_conn, if_exists="append", index=False)

        except Exception as e:
            print(f"Error with file {full_path}: {e}")
            skipped_empty.append(full_path)

    combined_conn.close()

    print(f"\n[{output_db}] Valid: {len(valid_files)}, Skipped: {len(skipped_empty)}")
    if skipped_empty:
        print("Skipped files:")
        for f in skipped_empty:
            print("  ", f)

    return valid_files, skipped_empty, start_index + len(all_files)


def sample_and_append(secondary_dir, output_db, pid, start_index, n_sample=350,
                       pattern="*5TeV*", seed=42, log_path="/mnt/scratch/baburish/doublepulse/gnn/Analysis/add_secondary-5TeV.log"):
    """Randomly sample n_sample files matching `pattern` in secondary_dir and
    append them into output_db, continuing offsets from start_index.

    Logs every file actually used (i.e. successfully read and appended) to
    `log_path`, along with any sampled files that were skipped.
    """
    all_secondary = sorted(glob.glob(os.path.join(secondary_dir, pattern)))
    print(f"{secondary_dir}: {len(all_secondary)} files matching {pattern}")

    random.seed(seed)
    sampled = random.sample(all_secondary, min(n_sample, len(all_secondary)))
    sampled_basenames = [os.path.basename(f) for f in sampled]

    valid_files, skipped_empty, next_index = combine_dbs(
        db_dir=secondary_dir,
        patterns=sampled_basenames,
        output_db=output_db,
        pid=pid,
        start_index=start_index,
        append=True,
    )

    with open(log_path, "a") as f:
        f.write(f"\n# sample_and_append: {len(sampled_basenames)} files sampled from "
                f"{secondary_dir} matching {pattern} (seed={seed}) -> {output_db}\n")
        f.write(f"# used ({len(valid_files)}):\n")
        for fp in valid_files:
            f.write(f"{fp}\n")
        if skipped_empty:
            f.write(f"# sampled but skipped ({len(skipped_empty)}):\n")
            for fp in skipped_empty:
                f.write(f"{fp}\n")

    return valid_files, skipped_empty, next_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=None,
                        help="Write a random sample of this many events per flavor "
                             "(e.g. 1000) to *_<N>.db instead of the full combined DBs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-kinematics", action="store_true",
                        help="With --n-events: choose the events so nutau and nue have the same true "
                             "(cos zenith, log10 E) distribution; writes *_<N>_matched.db.")
    args = parser.parse_args()

    outdir = "/mnt/scratch/baburish/doublepulse/gnn/Analysis/data/"

    suffix = ""
    if args.n_events is not None:
        n = args.n_events
        suffix = f"_{n // 1000}k" if n % 1000 == 0 else f"_{n}"
    if args.match_kinematics:
        if args.n_events is None:
            parser.error("--match-kinematics requires --n-events")
        suffix += "_matched"
    nutau_db = os.path.join(outdir, f"combined_nutau_5TeV{suffix}.db")
    nue_db = os.path.join(outdir, f"combined_nue_5TeV{suffix}.db")

    if args.match_kinematics:
        combine_matched(
            specs=[
                dict(db_dir="/mnt/ufs18/rs/IceCube/lownutau/sqlite/virgo_v4/nue_5TeV_train/",
                     patterns=["nue_5TeV_022855.db", "nue_5TeV_022856.db", "nue_5TeV_022857.db"],
                     output_db=nue_db, pid=11),
                dict(db_dir="/mnt/ufs18/rs/IceCube/lownutau/sqlite/virgo_v4/nutau_5TeV_train/",
                     patterns=["nutau_5TeV_022858.db", "nutau_5TeV_022859.db", "nutau_5TeV_022860.db"],
                     output_db=nutau_db, pid=16),
            ],
            n_events=args.n_events,
            seed=args.seed,
        )
        raise SystemExit(0)

    _, _, next_idx_nue = combine_dbs(
        db_dir="/mnt/ufs18/rs/IceCube/lownutau/sqlite/virgo_v4/nue_5TeV_train/",
        patterns=[
            "nue_5TeV_022855.db",
            "nue_5TeV_022856.db",
            "nue_5TeV_022857.db",
        ],
        output_db=nue_db,
        pid=11,
        n_events=args.n_events,
        seed=args.seed,
    )

    # secondary nutau dir: sample 350 of the 538 matching files, append into
    # the same combined_nutau_65TeV.db, continuing the offset.
    # sample_and_append(
    #     secondary_dir="/mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nutau_test/",
    #     output_db=nutau_db,
    #     pid=16,
    #     start_index=next_idx_nutau,
    #     n_sample=350,
    # )

    _, _, next_idx_nutau = combine_dbs(
        db_dir="/mnt/ufs18/rs/IceCube/lownutau/sqlite/virgo_v4/nutau_5TeV_train/",
        patterns=[
            "nutau_5TeV_022858.db",
            "nutau_5TeV_022859.db",
            "nutau_5TeV_022860.db",
        ],
        output_db=nutau_db,
        pid=16,
        n_events=args.n_events,
        seed=args.seed,
    )

    # secondary nue dir: sample 350 files, append into combined_nue_65TeV.db,
    # continuing the offset.
    # sample_and_append(
    #     secondary_dir="/mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nue_test/",
    #     output_db=nue_db,
    #     pid=11,
    #     start_index=next_idx_nue,
    #     n_sample=350,
    # )