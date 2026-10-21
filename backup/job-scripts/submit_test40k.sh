#!/bin/bash
# Submit the 40k kinematically matched comparison (both without PCA, LR reduced
# on val-loss plateau): summaries+edges baseline and per-DOM pulse encoder run
# in parallel (separate dataset caches); plots run after both succeed.
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts

A=$(sbatch --parsable job_test40k.sh test40k_edges_baseline_nopca --trainer trainer        --no_pca)
B=$(sbatch --parsable job_test40k.sh test40k_domenc_nopca         --trainer trainer_domenc --no_pca)
P=$(sbatch --parsable --dependency=afterok:$A:$B job_test40k_plot.sh)

echo "baseline_nopca=$A  domenc_nopca=$B  plots=$P (after $A,$B)"
