"""Combine ALL eligible events of the Virgo v4 65 TeV training sets into one DB per flavour.

Uses the selection and event_no offsets of merge-5TeV-moresamples.py (brightest-DOM PE > 10,
truth.seed_string not null, event_no += file_index * 1e6) and its read-only, row-selective copy
(only the eligible events of truth / reco / CleanedROIPulses are written).

Outputs: Analysis/data/combined_{nutau,nue}_65TeV_virgo.db (refuses to overwrite).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("merge5", os.path.join(HERE, "merge-5TeV-moresamples.py"))
merge5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge5)

SRC = "/mnt/ufs18/rs/IceCube/lownutau/sqlite/virgo_v4"
OUTDIR = "/mnt/scratch/baburish/doublepulse/gnn/Analysis/data"
ALL = 10 ** 9   # larger than any pool: take every eligible event

if __name__ == "__main__":
    jobs = [
        dict(db_dir=f"{SRC}/nue_65TeV_train/", patterns=["nue_65TeV_*.db"],
             output_db=f"{OUTDIR}/combined_nue_65TeV_virgo.db", pid=11),
        dict(db_dir=f"{SRC}/nutau_65TeV_train/", patterns=["nutau_65TeV_*.db"],
             output_db=f"{OUTDIR}/combined_nutau_65TeV_virgo.db", pid=16),
    ]
    for j in jobs:
        if os.path.exists(j["output_db"]):
            sys.exit(f"{j['output_db']} exists -- not overwriting")
    for j in jobs:
        merge5.combine_dbs(**j, n_events=ALL, seed=42)
