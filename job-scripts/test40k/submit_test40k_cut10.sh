#!/bin/bash
# 40k random samples with the per-DOM charge cut (DOMs < 10 PE removed; events
# left with < 2 strings dropped; classes rebalanced 1:1 by the dataset).
# 1. one job builds the shared trainer_domenc dataset cache (incl. images),
# 2. GAT-only and GAT+CNN (brightest-string image) train in parallel on it,
# 3. plots after both succeed.
# All runs: no PCA branch, no auxiliary targets, LR halved on val-loss plateau.
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts/test40k

CUT="--trainer trainer_domenc --no_pca --min_dom_charge 10"
C=$(sbatch --parsable job_test40k.sh test40k_cut10_cache $CUT --build_only)
A=$(sbatch --parsable --dependency=afterok:$C job_test40k.sh test40k_gat_cut10    $CUT)
B=$(sbatch --parsable --dependency=afterok:$C job_test40k.sh test40k_gatcnn_cut10 $CUT --use_image)
P=$(sbatch --parsable --dependency=afterok:$A:$B job_test40k_cut10_plot.sh)

echo "cache_build=$C  gat=$A (after $C)  gat+cnn=$B (after $C)  plots=$P (after $A,$B)"
