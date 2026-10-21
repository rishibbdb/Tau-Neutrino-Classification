#!/bin/bash
# GAT-branch improvements on the 40k random samples with the 10 PE DOM cut
# (reuses the existing dataset cache; no PCA, no aux, LR halved on plateau).
# Reference: test40k_gat_cut10 (plain GAT, job given as $1 so the plots wait for it).
#   arch       : multi readout (sum+mean+max+attention+brightest DOM) + residual GAT layers
#   arch_feats : arch + brightest-relative node features + fine-time/causality edge features
#   arch_full  : arch_feats + fully connected DOM graph per event
set -e
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis/job-scripts
REF_JOB=${1:?usage: submit_test40k_gat_ablation.sh <job id of the running test40k_gat_cut10 reference>}

BASE="--trainer trainer_domenc --no_pca --min_dom_charge 10"
ARCH="--gat_readout multi --gat_residual"
FEATS="--bright_feats --extra_edges"
A=$(sbatch --parsable job_test40k.sh test40k_gat_cut10_arch       $BASE $ARCH)
B=$(sbatch --parsable job_test40k.sh test40k_gat_cut10_arch_feats $BASE $ARCH $FEATS)
C=$(sbatch --parsable job_test40k.sh test40k_gat_cut10_arch_full  $BASE $ARCH $FEATS --full_graph)
P=$(sbatch --parsable --dependency=afterok:$REF_JOB:$A:$B:$C job_test40k_gat_ablation_plot.sh)

echo "arch=$A  arch_feats=$B  arch_full=$C  plots=$P (after $REF_JOB,$A,$B,$C)"
