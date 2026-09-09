import os
import glob
import sqlite3
import pandas as pd
from tqdm import tqdm

OFFSET = 1_000_000


def combine_dbs(db_dir, patterns, output_db, pid):
    """Combine all files matching any of `patterns` in db_dir into output_db."""
    if os.path.exists(output_db):
        os.remove(output_db)

    all_files = sorted(set().union(
        *[glob.glob(os.path.join(db_dir, p)) for p in patterns]
    ))
    print(f"[{output_db}] Found {len(all_files)} matching files across patterns={patterns}")

    combined_conn = sqlite3.connect(output_db)
    valid_files, skipped_empty = [], []

    for file_index, full_path in enumerate(tqdm(all_files, desc=os.path.basename(output_db))):
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

    return valid_files, skipped_empty


if __name__ == "__main__":
    outdir = "/mnt/scratch/baburish/doublepulse/gnn/Analysis/data/"

    combine_dbs(
        db_dir="/mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nutau_train/",
        patterns=[
            "nutau_gemini_ftp_65TeV*",
            "nutau_gemini_65TeV*",
        ],
        output_db=os.path.join(outdir, "combined_nutau_65TeV.db"),
        pid=16,
    )

    combine_dbs(
        db_dir="/mnt/research/IceCube/lownutau/gemini_moreMC_sqlite/nue_train/",
        patterns=[
            "nue_gemini_taupede_65TeV*",
            "nue_gemini_ftp_65TeV*",
            "nue_gemini_65TeV*"
        ],
        output_db=os.path.join(outdir, "combined_nue_65TeV.db"),
        pid=11,
    )