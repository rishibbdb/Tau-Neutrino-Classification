#!/bin/bash
# 65 TeV network: same architecture/options as the best 5 TeV run, trained separately on the Virgo v4
# 65 TeV samples (data/combined_*_65TeV_virgo.db), own dataset cache (dataset_cache_65TeV) and outputs
# (virgo_65TeV/). The auxiliary energy target is re-standardised on the 65 TeV training set
# (--aux_norm train): the fixed 5 TeV constants would put it at ~2-7 and let it dominate the loss.
#   1. build the 65 TeV dataset cache (one job)
#   2. train in up to two chained 3.5 h segments (checkpoint/resume)
#   3. evaluation on the independent 65 TeV TEST sample, then plots (test results)
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts

RUN=virgo65_gat_arch_feats_aux1
OPTS="--trainer trainer_domenc --no_pca --min_dom_charge 10 --min_strings 1 --gat_readout multi --gat_residual --bright_feats --extra_edges --aux_weight 1 --aux_norm train --lr 1e-3 --batch_size 64 --max_hours 3.25"

C=$(sbatch --parsable job_virgo65.sh virgo65_cache $OPTS --build_only)
T1=$(sbatch --parsable --dependency=afterok:$C   job_virgo65.sh $RUN $OPTS)
T2=$(sbatch --parsable --dependency=afterany:$T1 job_virgo65.sh $RUN $OPTS --resume)
# 3. evaluate on the independent TEST sample (virgo_v4/*_65TeV_test), natural composition
ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
E=$(sbatch --parsable --dependency=afterany:$T2 job_test_eval.sh $ANA/virgo_65TeV/$RUN \
    $ANA/data/combined_nutau_65TeV_virgo_test.db $ANA/data/combined_nue_65TeV_virgo_test.db $ANA/dataset_cache_65TeV_test)
P=$(sbatch --parsable --dependency=afterok:$E job_virgo65_plot.sh)
echo "cache=$C  train=$T1 -> $T2 (resume; exits at once if finished)  test eval=$E  plots=$P"
