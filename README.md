# Tau-Neutrino-Classification: working point 1

A graph attention network (GAT) classifier that separates ν_τ (signal, pid 16) from ν_e (background, pid 11)
in IceCube cascade-like events, using DOM-level event graphs built from `CleanedROIPulses`.
There are two networks, one per energy range. Both are built with the same tools but trained separately:

| sample | network | trainer module | training data | model in this branch | val AUC |
|---|---|---|---|---|---|
| **< 65 TeV** ("5 TeV") | per-DOM pulse encoder + residual GATv1 with edge features, multi readout, brightest-DOM features, energy/direction auxiliary heads | `trainer_domenc.py` | Virgo v4 `nutau/nue_5TeV` (`data/combined_{nutau,nue}_5TeV.db`) | `trained_model5TeV/full_nodist_aux1_bs64/` | 0.669 |
| **> 65 TeV** | 3 × GATConv (no edge features) + PCA branch | `trainer_noedge.py` | Gemini 65 TeV (`data/combined_{nutau,nue}_65TeV.db`) | `trained_model/GAT_0p3dropout_weightdecay5e4_64batchsize/` | 0.777 |

Validation AUCs come from 1:1 balanced validation splits of the training samples. Quote performance
from the independent test samples instead (see [Evaluation](#evaluation)). `docs/gat_tau_classifier.tex` documents the preprocessing,
features and architecture.

## Layout

```
train.py                 training driver (both networks; --trainer selects the model module)
trainer.py               shared preprocessing, event selection, graph construction, dataset cache
trainer_domenc.py        < 65 TeV model: pulse encoder + GAT
trainer_noedge.py        > 65 TeV model: GAT + PCA branch
evaluate_test.py         score an independent test sample -> eval_test.npz, scores.csv (non-PCA runs)
infer_to_db.py           score SQLite files and append the scores to their reco table (both networks)
plot_eval.py             training history, ROC, score distributions, AUC vs true energy/direction
geometry_clean.csv       DOM geometry
virgo/                   merge scripts that build the training/test DBs
job-scripts/             SLURM jobs (ICER, general-short, 3.5 h)
notebooks/               ABCD/threshold study, feature and preprocessing checks
inspect-training-live*.ipynb   follow a running training (5 TeV / 65 TeV)
trained_model5TeV/full_nodist_aux1_bs64/                  < 65 TeV working-point weights + config + plots
trained_model/GAT_0p3dropout_weightdecay5e4_64batchsize/  > 65 TeV working-point weights + PCA scaler + log
```

The job scripts and some defaults use absolute paths under
`/mnt/scratch/baburish/doublepulse/gnn/Analysis` (the code, `data/` and `dataset_cache*` live there).
If you run from another checkout, edit `ANA=` in the job scripts.

## Environment

```bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate   # torch, torch_geometric, sklearn, pandas
```

## 1. Build the data

Every merge script applies the same selection: brightest-DOM charge > 10 PE, `truth.seed_string` not null,
and `event_no += file_index * 1e6`. The scripts write to `Analysis/data/`.

| purpose | command | output |
|---|---|---|
| < 65 TeV training | `python virgo/merge-5TeV-moresamples.py` | `combined_{nutau,nue}_5TeV.db` |
| < 65 TeV subsample for development | `python virgo/merge-5TeV-moresamples.py --n-events 40000 [--match-kinematics]` | `combined_*_5TeV_40k[_matched].db` |
| < 65 TeV test | `python virgo/merge-virgo.py --energy 5TeV --split test` | `combined_*_5TeV_virgo_test.db` |
| > 65 TeV training (Gemini, working point) | `python virgo/merge-65TeV-moresamples.py` | `combined_{nutau,nue}_65TeV.db` |
| > 65 TeV test | `python virgo/merge-virgo.py --energy 65TeV --split test` | `combined_*_65TeV_virgo_test.db` |
| > 65 TeV Virgo training (not used by the working point) | `python virgo/merge-65TeV-virgo.py` | `combined_*_65TeV_virgo.db` |

`merge-virgo.py` and `merge-65TeV-virgo.py` refuse to overwrite existing outputs.
`merge-65TeV-moresamples.py` deletes and rewrites its outputs, so check `data/` before you run it.
The Gemini 65 TeV DBs are not currently in `Analysis/data/`; rebuild them before you retrain the > 65 TeV network.

The first training run on a sample builds a dataset cache (`dataset_cache*/dataset_<md5>.pt`), keyed by every
preprocessing setting. Later runs with the same settings read the cache. The two energy ranges use separate cache directories.

## 2. Training

### < 65 TeV

The working-point configuration:

```
--trainer trainer_domenc --no_pca --min_dom_charge 10 --min_strings 1 --gat_readout multi --gat_residual
--bright_feats --extra_edges --aux_weight 1 --lr 1e-3 --batch_size 64 --epochs 150 --patience 10
```

That gives 12 node features. Pulses must be > 0.1 PE, and DOMs with < 10 PE total are removed.
Single-string events are kept. Training uses 1:1 balanced classes and a 15 % validation split, and
the LR halves when the validation loss plateaus. The auxiliary log10 E and direction regression is
for training only; the classifier score is the output.

```bash
cd job-scripts
bash submit_full_nodist_bs.sh     # cache build -> batch 64 and 512 chains (3 x 3.5 h, --resume) -> plots
# or a single run:
OPTS="--trainer trainer_domenc --no_pca --min_dom_charge 10 --min_strings 1 --gat_readout multi --gat_residual --bright_feats --extra_edges --aux_weight 1 --lr 1e-3 --max_hours 3.25"
sbatch job_full.sh my_run $OPTS --batch_size 64                 # -> Analysis/trained_model5TeV/my_run{,.log}
sbatch job_full.sh my_run $OPTS --batch_size 64 --resume        # continue from my_run/checkpoint.pt
```

`job_full.sh` refuses to overwrite an existing run directory or log. `--max_hours` stops cleanly before
the SLURM limit, and `--resume` continues from `checkpoint.pt`. A finished run directory contains
`hybrid_weights.pt` (best epoch), `run_config.json`, `eval_val.npz` and plots.

### > 65 TeV

```bash
sbatch job-scripts/job_65TeV.sh my_run65      # -> Analysis/trained_model/my_run65{,.log}
```

This uses `train.py --trainer trainer_noedge` with the PCA branch on, 13 node features, pulse charge
> 0.1 PE, ≥ 2 strings, batch 64, lr 3e-4, AdamW weight decay 5e-4, and early stopping. These are
the settings of the working-point model. Its training log is
`trained_model/GAT_0p3dropout_weightdecay5e4_64batchsize/c.log` (best epoch 81).
The fitted PCA scaler (`pca_scaler.joblib`) is saved next to the weights. Inference needs it.

## 3. Evaluation

### Validation plots (from `eval_val.npz`)

```bash
python plot_eval.py --runs trained_model5TeV/full_nodist_aux1_bs64 --labels "5 TeV WP" --out_dir plots/wp1_val
```

This writes `training_history.png`, `roc.png` and `score_distribution.png` per run, and `aux_regression.png`
for runs with auxiliary heads. It also writes a comparison ROC, AUC in true cos-zenith and log10 E bins,
and `summary.csv`. Each plot shows at most 3 runs.
The > 65 TeV legacy run has no `eval_val.npz`; use its `training_curves.png`.

### Independent test sample

**< 65 TeV** (`evaluate_test.py`). It rebuilds the test set with the run's own preprocessing,
without balancing:

```bash
ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
sbatch job-scripts/job_test_eval.sh $ANA/trained_model5TeV/full_nodist_aux1_bs64 \
    $ANA/data/combined_nutau_5TeV_virgo_test.db $ANA/data/combined_nue_5TeV_virgo_test.db $ANA/dataset_cache_5TeV_test
# -> <run>/test/eval_test.npz, <run>/test/scores.csv, <run>/test.log
python plot_eval.py --runs trained_model5TeV/full_nodist_aux1_bs64/test --eval_file eval_test.npz --out_dir plots/wp1_test
```

**> 65 TeV**. `evaluate_test.py` does not support PCA runs, so score the test DBs with `infer_to_db.py`.
Work on copies, because it adds columns to `reco`. Then compute the AUC from those columns:

```bash
cp $ANA/data/combined_{nutau,nue}_65TeV_virgo_test.db /path/to/scratch/
python infer_to_db.py --run_dir trained_model/GAT_0p3dropout_weightdecay5e4_64batchsize --trainer trainer_noedge \
    --prefix gnn65 --db /path/to/scratch/combined_nutau_65TeV_virgo_test.db /path/to/scratch/combined_nue_65TeV_virgo_test.db --no_backup
python - <<'EOF'
import sqlite3, pandas as pd
from sklearn.metrics import roc_auc_score
q = "SELECT gnn65_prob_tau AS p FROM reco WHERE gnn65_valid = 1"
d = pd.concat([pd.read_sql(q, sqlite3.connect(f"/path/to/scratch/combined_{f}_65TeV_virgo_test.db")).assign(y=y)
               for f, y in [("nutau", 1), ("nue", 0)]])
print(len(d), "events, test AUC", roc_auc_score(d.y, d.p))
EOF
```

### Scoring event files (both networks)

`infer_to_db.py` scores every event in a SQLite file's `reco` table. It uses the preprocessing and model
options stored with the run (`run_config.json`). For the legacy > 65 TeV run, pass `--trainer trainer_noedge`.
It adds `<prefix>_prob_tau`, `_logit`, `_valid` (1 = passed the run's selection) and `_n_doms`. For aux runs it also adds
`_pred_log10_energy`, `_pred_zenith` and `_pred_azimuth`. It writes a `<prefix>_model_info` table too.

```bash
python infer_to_db.py --run_dir trained_model5TeV/full_nodist_aux1_bs64 --prefix gnn5 --db file.db
python infer_to_db.py --run_dir trained_model/GAT_0p3dropout_weightdecay5e4_64batchsize --trainer trainer_noedge --prefix gnn65 --db file.db
```

Each file is copied to `--backup_dir` first, unless you pass `--no_backup`. All changes to a file go in
one transaction. Existing `<prefix>_` columns are replaced only with `--overwrite`.

### Working-point threshold (ABCD)

`notebooks/abcd_threshold.ipynb` takes a run directory (`eval_val.npz`) or a `scores.csv`. For a chosen
`prob_tau` threshold it shows the true-vs-predicted (ABCD) matrix, efficiencies, purity and a threshold scan.

## Known caveats

- In the training samples, the ν_τ and ν_e true direction and energy distributions differ. Part of the AUC
  therefore comes from kinematics. `plot_eval.py` reports AUC within true cos-zenith/log E bins
  (≈ 0.56–0.57 at < 65 TeV), which is closer to the flavour-only separation.
- The reconstructed cascade vertex in the samples is unreliable, so the < 65 TeV network does not use
  `distance_to_cascade`. The legacy > 65 TeV network (13 features) still does.
- The > 65 TeV network was trained on Gemini MC. Its performance on Virgo v4 65 TeV test files is lower
  than its Gemini validation AUC.
