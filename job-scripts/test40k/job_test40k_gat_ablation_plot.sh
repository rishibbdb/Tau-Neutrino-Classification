#!/bin/bash --login
#SBATCH --job-name=test40k-gat-abl-plot
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test40k_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Comparison plots for the GAT-branch ablation (submit_test40k_gat_ablation.sh).
# plot_eval.py shows at most 3 runs per plot, so two comparisons share the reference.

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

for d in test40k_gat_ablation_a test40k_gat_ablation_b; do
    if [ -e "$OUT/$d" ]; then
        echo "ERROR: $OUT/$d already exists -- not overwriting" >&2
        exit 1
    fi
done

python plot_eval.py \
    --runs   $OUT/test40k_gat_cut10 $OUT/test40k_gat_cut10_arch $OUT/test40k_gat_cut10_arch_feats \
    --labels "GAT (reference)" "+ multi readout, residual" "+ brightest/causal features" \
    --out_dir $OUT/test40k_gat_ablation_a

python plot_eval.py \
    --runs   $OUT/test40k_gat_cut10 $OUT/test40k_gat_cut10_arch_feats $OUT/test40k_gat_cut10_arch_full \
    --labels "GAT (reference)" "+ readout, residual, features" "+ fully connected graph" \
    --out_dir $OUT/test40k_gat_ablation_b
