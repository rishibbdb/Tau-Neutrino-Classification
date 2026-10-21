import os
import glob
import random
import sqlite3
import pandas as pd
from tqdm import tqdm

OFFSET = 1_000_000


def _assign_true_event_id(df_truth, df_reco, df_pulses, source_path):
    """Recompute a reliable event id from row position, since raw event_no
    collides across different run_id within a single source file. Returns
    (df_truth, df_reco, df_pulses) with event_no REPLACED by the true id,
    and any positionally-ambiguous groups (adjacent truth rows that already
    share event_no with no separating signal) dropped."""

    assert len(df_truth) == len(df_reco), f"{source_path}: truth/reco row count mismatch"
    assert (df_truth["event_no"].values == df_reco["event_no"].values).all(), \
        f"{source_path}: truth/reco event_no sequence mismatch -- do not trust this file"

    event_no_vals = df_truth["event_no"].values
    true_ids, ambiguous_ids = [], set()
    current_id, prev, run_start = -1, object(), 0
    for i, e in enumerate(event_no_vals):
        if e != prev:
            if i - run_start > 1:
                ambiguous_ids.add(current_id)
            current_id += 1
            run_start = i
            prev = e
        true_ids.append(current_id)
    if len(event_no_vals) - run_start > 1:
        ambiguous_ids.add(current_id)

    df_truth = df_truth.copy()
    df_reco = df_reco.copy()
    df_truth["event_no"] = true_ids
    df_reco["event_no"] = true_ids

    pulse_ids = []
    current_id, prev = -1, object()
    for e in df_pulses["event_no"].values:
        if e != prev:
            current_id += 1
            prev = e
        pulse_ids.append(current_id)
    df_pulses = df_pulses.copy()
    df_pulses["event_no"] = pulse_ids

    n_truth_events = df_truth["event_no"].nunique()
    n_pulse_events = df_pulses["event_no"].nunique()
    assert n_truth_events == n_pulse_events, \
        f"{source_path}: {n_truth_events} truth events vs {n_pulse_events} pulses runs -- mismatch"

    if ambiguous_ids:
        df_truth = df_truth[~df_truth["event_no"].isin(ambiguous_ids)]
        df_reco = df_reco[~df_reco["event_no"].isin(ambiguous_ids)]
        df_pulses = df_pulses[~df_pulses["event_no"].isin(ambiguous_ids)]

    return df_truth, df_reco, df_pulses


def combine_dbs(db_dir, patterns, output_db, pid, start_index=0, append=False, min_pe=10):
    """Combine all files matching any of `patterns` in db_dir into output_db.

    start_index: file_index to start counting from (so event_no offsets don't
                 collide with a previous call into the same output_db).
    append: if False (default), wipes output_db first. Set True to add into
            an existing output_db from a prior call.
    """
    if not append and os.path.exists(output_db):
        os.remove(output_db)
        assert not os.path.exists(output_db), f"Failed to remove stale {output_db}"

    all_files = sorted(set().union(
        *[glob.glob(os.path.join(db_dir, p)) for p in patterns]
    ))
    print(f"[{output_db}] Found {len(all_files)} matching files across patterns={patterns}")

    combined_conn = sqlite3.connect(output_db)
    valid_files, skipped_empty = [], []

    for i, full_path in enumerate(tqdm(all_files, desc=os.path.basename(output_db))):
        file_index = start_index + i
        try:
            conn = sqlite3.connect(full_path)
            df_truth = pd.read_sql_query("SELECT * FROM truth ORDER BY rowid;", conn)
            df_pulses = pd.read_sql_query("SELECT * FROM CleanedROIPulses ORDER BY rowid;", conn)
            df_reco = pd.read_sql_query("SELECT * FROM reco ORDER BY rowid;", conn)
            conn.close()

            if df_reco.empty or df_truth.empty or df_pulses.empty:
                skipped_empty.append(full_path)
                continue

            df_truth, df_reco, df_pulses = _assign_true_event_id(
                df_truth, df_reco, df_pulses, full_path
            )

            df_reco = df_reco[df_reco["brightest_om_brightest_string_PE"] > min_pe]
            keep = df_reco["event_no"].unique()
            df_truth = df_truth[df_truth["event_no"].isin(keep)]
            df_pulses = df_pulses[df_pulses["event_no"].isin(keep)]

            if df_truth.empty:
                skipped_empty.append(full_path)
                continue

            if "seed_string" in df_truth.columns:
                df_truth = df_truth.dropna(subset=["seed_string"])

            if df_truth.empty:
                skipped_empty.append(full_path)
                continue

            df_truth["pid"] = pid

            event_offset = file_index * OFFSET
            df_truth["event_no"] = df_truth["event_no"] + event_offset
            df_pulses["event_no"] = df_pulses["event_no"] + event_offset
            df_reco["event_no"] = df_reco["event_no"] + event_offset
            valid_files.append(full_path)

            df_truth.to_sql("truth", combined_conn, if_exists="append", index=False)
            df_pulses.to_sql("CleanedROIPulses", combined_conn, if_exists="append", index=False)
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
    outdir = "/mnt/scratch/baburish/doublepulse/gnn/Analysis/data/"

    nutau_db = os.path.join(outdir, "combined_nutau_5TeV.db")
    nue_db = os.path.join(outdir, "combined_nue_5TeV.db")

    _, _, next_idx_nutau = combine_dbs(
        db_dir="/mnt/ufs18/rs/IceCube/lownutau/sqlite/virgo_v3/nue_5TeV_train/",
        patterns=[
            "nue_5TeV_022855.db",
            "nue_5TeV_022856.db",
            "nue_5TeV_022857.db",
        ],
        output_db=nue_db,
        pid=11,
    )

    _, _, next_idx_nue = combine_dbs(
        db_dir="/mnt/ufs18/rs/IceCube/lownutau/sqlite/virgo_v3/nutau_5TeV_train/",
        patterns=[
            "nutau_5TeV_022858.db",
            "nutau_5TeV_022859.db",
            "nutau_5TeV_022860.db",
        ],
        output_db=nutau_db,
        pid=16,
    )