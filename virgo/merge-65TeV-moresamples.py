import os
import glob
import random
import sqlite3
import pandas as pd
from tqdm import tqdm

OFFSET = 1_000_000


def combine_dbs(db_dir, patterns, output_db, pid, start_index=0, append=False):
    """Combine all files matching any of `patterns` in db_dir into output_db.

    start_index: file_index to start counting from (so event_no offsets don't
                 collide with a previous call into the same output_db).
    append: if False (default), wipes output_db first. Set True to add into
            an existing output_db from a prior call.
    """
    if not append and os.path.exists(output_db):
        os.remove(output_db)

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

    return valid_files, skipped_empty, start_index + len(all_files)


def sample_and_append(secondary_dir, output_db, pid, start_index, n_sample=350,
                       pattern="*65TeV*", seed=42, log_path="add_secondary.log"):
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

    nutau_db = os.path.join(outdir, "combined_nutau_65TeV.db")
    nue_db = os.path.join(outdir, "combined_nue_65TeV.db")

    _, _, next_idx_nutau = combine_dbs(
        db_dir="/mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nutau_train",
        patterns=[
            "nutau_gemini_ftp_65TeV*",
            "nutau_gemini_65TeV*",
        ],
        output_db=nutau_db,
        pid=16,
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

    _, _, next_idx_nue = combine_dbs(
        db_dir="/mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nue_train/",
        patterns=[
            "nue_gemini_taupede_65TeV*",
            "nue_gemini_ftp_65TeV*",
            "nue_gemini_65TeV*",
        ],
        output_db=nue_db,
        pid=11,
    )

    # secondary nue dir: sample 350 files, append into combined_nue_65TeV.db,
    # # continuing the offset.
    # sample_and_append(
    #     secondary_dir="/mnt/research/IceCube/lownutau/gemini/gemini_moreMC_sqlite/nue_test/",
    #     output_db=nue_db,
    #     pid=11,
    #     start_index=next_idx_nue,
    #     n_sample=350,
    # )