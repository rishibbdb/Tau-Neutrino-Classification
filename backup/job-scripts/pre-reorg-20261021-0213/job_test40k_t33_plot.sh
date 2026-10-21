#!/bin/bash --login
#SBATCH --job-name=test40k-t33-plot
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test40k_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Plots for the 3.3 ns time-unit runs (submit_test40k_t33.sh) vs the same variant
# with the original time scalings.

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

if [ -e "$OUT/test40k_t33_comparison" ]; then
    echo "ERROR: $OUT/test40k_t33_comparison already exists -- not overwriting" >&2
    exit 1
fi

python plot_eval.py \
    --runs   $OUT/test40k_gat_cut10_arch_feats $OUT/test40k_gat_cut10_arch_feats_t33 $OUT/test40k_gat_cut10_arch_feats_t33_gelu \
    --labels "brightest/causal (original time scales)" "+ 3.3 ns time unit" "+ 3.3 ns time unit, GELU" \
    --out_dir $OUT/test40k_t33_comparison
