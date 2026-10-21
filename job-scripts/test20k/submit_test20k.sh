#!/bin/bash
# Submit the 20k comparison: baseline and pulse-encoder runs in parallel; the
# no-PCA run waits for the pulse-encoder run so it reuses its dataset cache
# (instead of both building/writing the same cache file at once); plots run
# after all three succeed.
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts/test20k

A=$(sbatch --parsable job_test20k.sh test20k_edges_baseline --trainer trainer)
B=$(sbatch --parsable job_test20k.sh test20k_domenc         --trainer trainer_domenc)
C=$(sbatch --parsable --dependency=afterok:$B job_test20k.sh test20k_domenc_nopca --trainer trainer_domenc --no_pca)
P=$(sbatch --parsable --dependency=afterok:$A:$C job_test20k_plot.sh)

echo "baseline=$A  domenc=$B  domenc_nopca=$C (after $B)  plots=$P (after $A,$C)"
