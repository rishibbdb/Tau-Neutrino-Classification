#!/bin/bash
# Full-sample training of the best 40k variant ("+ brightest/causal features"):
# trainer_domenc, no PCA, DOMs < 10 PE removed, multi readout + residual GAT
# layers, brightest-relative node features, fine-time/causality edge features.
# All eligible events (162k nutau / 238k nue before cuts; balanced 1:1).
# Single-string events are kept (--min_strings 1): a double-cascade imprint can sit on one string.
#   1. build the dataset cache (one job)
#   2. train in up to three chained 3.5 h segments (checkpoint/resume)
#   3. plots vs the 40k runs
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts/full_sample

RUN=full_gat_cut10_ms1_arch_feats_run2
OPTS="--trainer trainer_domenc --no_pca --min_dom_charge 10 --min_strings 1 --gat_readout multi --gat_residual --bright_feats --extra_edges"

C=$(sbatch --parsable ../job_full.sh full_cut10_ms1_cache $OPTS --build_only)
T1=$(sbatch --parsable --dependency=afterok:$C   ../job_full.sh $RUN $OPTS --max_hours 3.25)
T2=$(sbatch --parsable --dependency=afterany:$T1 ../job_full.sh $RUN $OPTS --max_hours 3.25 --resume)
T3=$(sbatch --parsable --dependency=afterany:$T2 ../job_full.sh $RUN $OPTS --max_hours 3.25 --resume)
P=$(sbatch --parsable --dependency=afterany:$T3 job_full_plot.sh)

echo "cache=$C  train=$T1 -> $T2 -> $T3 (resume segments; exit at once if finished)  plots=$P"
