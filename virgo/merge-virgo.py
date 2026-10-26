"""Combine ALL eligible events of one Virgo v4 sample (energy x split) into one DB per flavour.

Same selection, event_no offsets and row-selective copy as merge-5TeV-moresamples.py
(brightest-DOM PE > 10, truth.seed_string not null, event_no += file_index * 1e6).

    python virgo/merge-virgo.py --energy 65TeV --split test
      -> Analysis/data/combined_{nutau,nue}_65TeV_virgo_test.db   (refuses to overwrite)

(The 65 TeV training sample was merged with merge-65TeV-virgo.py -> combined_*_65TeV_virgo.db.)
"""
import argparse
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
    p = argparse.ArgumentParser()
    p.add_argument("--energy", required=True, choices=["5TeV", "65TeV"])
    p.add_argument("--split", required=True, choices=["train", "test"])
    args = p.parse_args()

    suffix = "" if args.split == "train" else f"_{args.split}"
    jobs = [dict(db_dir=f"{SRC}/{fl}_{args.energy}_{args.split}/", patterns=[f"{fl}_{args.energy}_*.db"],
                 output_db=f"{OUTDIR}/combined_{fl}_{args.energy}_virgo{suffix}.db", pid=pid)
            for fl, pid in [("nue", 11), ("nutau", 16)]]
    for j in jobs:
        if not os.path.isdir(j["db_dir"]):
            sys.exit(f"no such directory: {j['db_dir']}")
        if os.path.exists(j["output_db"]):
            sys.exit(f"{j['output_db']} exists -- not overwriting")
    for j in jobs:
        merge5.combine_dbs(**j, n_events=ALL, seed=42)
