#!/bin/bash --login
#SBATCH --job-name=test40k-cut10-plot
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test40k_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Evaluation plots for the 40k per-DOM-charge-cut runs (submit_test40k_cut10.sh).

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

if [ -e "$OUT/test40k_cut10_comparison" ]; then
    echo "ERROR: $OUT/test40k_cut10_comparison already exists -- not overwriting" >&2
    exit 1
fi

python plot_eval.py \
    --runs   $OUT/test40k_gat_cut10 $OUT/test40k_gatcnn_cut10 \
    --labels "GAT (DOM >= 10 PE)" "GAT + CNN image (DOM >= 10 PE)" \
    --out_dir $OUT/test40k_cut10_comparison
