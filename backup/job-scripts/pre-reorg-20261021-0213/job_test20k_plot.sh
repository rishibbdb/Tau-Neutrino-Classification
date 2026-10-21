#!/bin/bash --login
#SBATCH --job-name=test20k-plot
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test20k_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Evaluation plots for the three test20k runs (submitted by submit_test20k.sh
# with a dependency on the training jobs).

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

if [ -e "$OUT/test20k_comparison" ]; then
    echo "ERROR: $OUT/test20k_comparison already exists -- not overwriting" >&2
    exit 1
fi

python plot_eval.py \
    --runs   $OUT/test20k_edges_baseline $OUT/test20k_domenc $OUT/test20k_domenc_nopca \
    --labels "Summaries + edges" "Pulse encoder" "Pulse encoder, no PCA" \
    --out_dir $OUT/test20k_comparison
