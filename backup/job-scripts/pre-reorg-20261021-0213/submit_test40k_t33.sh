#!/bin/bash
# 40k random samples, 10 PE DOM cut (existing cache): best variant
# ("+ brightest/causal features") with all fine time differences expressed in
# units of the 3.3 ns detector time resolution, without / with GELU in the
# pulse encoder. Compared with test40k_gat_cut10_arch_feats (same, original scalings).
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts

OPTS="--trainer trainer_domenc --no_pca --min_dom_charge 10 --gat_readout multi --gat_residual --bright_feats --extra_edges --time_unit_ns 3.3"
A=$(sbatch --parsable job_test40k.sh test40k_gat_cut10_arch_feats_t33      $OPTS)
B=$(sbatch --parsable job_test40k.sh test40k_gat_cut10_arch_feats_t33_gelu $OPTS --pulse_act gelu)
P=$(sbatch --parsable --dependency=afterok:$A:$B job_test40k_t33_plot.sh)

echo "t33=$A  t33_gelu=$B  plots=$P (after $A,$B)"
