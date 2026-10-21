#!/bin/bash
# Submit the 40k comparison on the random (not kinematically selected) samples.
# Both runs: no PCA branch, LR reduced on val-loss plateau, and true energy +
# direction as auxiliary regression targets (multi-task, --aux_weight).
# Summaries+edges baseline and per-DOM pulse encoder run in parallel (separate
# dataset caches); plots run after both succeed.
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts/test40k

AUX="--aux_weight 0.5"
A=$(sbatch --parsable job_test40k.sh test40k_edges_baseline_nopca_aux --trainer trainer        --no_pca $AUX)
B=$(sbatch --parsable job_test40k.sh test40k_domenc_nopca_aux         --trainer trainer_domenc --no_pca $AUX)
P=$(sbatch --parsable --dependency=afterok:$A:$B job_test40k_plot.sh)

echo "baseline_nopca_aux=$A  domenc_nopca_aux=$B  plots=$P (after $A,$B)"
