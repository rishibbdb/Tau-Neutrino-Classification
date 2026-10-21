#!/bin/bash
# Full sample, node features WITHOUT distance_to_cascade (new 12-feature cache), single-string
# events kept, 10 PE DOM cut, brightest/causal GAT options, energy/direction auxiliary targets
# (weight 1), lr 1e-3. Two batch sizes: 64 and 512.
#   1. rebuild the dataset cache (one job)
#   2. per batch size: up to three chained 3.5 h training segments (checkpoint/resume)
#   3. plots for both runs vs the previous full-sample run
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts

OPTS="--trainer trainer_domenc --no_pca --min_dom_charge 10 --min_strings 1 --gat_readout multi --gat_residual --bright_feats --extra_edges --aux_weight 1 --lr 1e-3 --max_hours 3.25"

C=$(sbatch --parsable job_full.sh full_nodist_ms1_cache $OPTS --build_only)
LAST=()
for BS in 64 512; do
    RUN=full_nodist_aux1_bs$BS
    T1=$(sbatch --parsable --dependency=afterok:$C   job_full.sh $RUN $OPTS --batch_size $BS)
    T2=$(sbatch --parsable --dependency=afterany:$T1 job_full.sh $RUN $OPTS --batch_size $BS --resume)
    T3=$(sbatch --parsable --dependency=afterany:$T2 job_full.sh $RUN $OPTS --batch_size $BS --resume)
    echo "batch $BS: $T1 -> $T2 -> $T3 (resume segments exit at once if finished)"
    LAST+=($T3)
done
P=$(sbatch --parsable --dependency=afterany:${LAST[0]}:${LAST[1]} job_full_nodist_plot.sh)
echo "cache=$C  plots=$P (after ${LAST[0]},${LAST[1]})"
