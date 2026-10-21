#!/bin/bash --login
#SBATCH --job-name=test40k-plot
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test40k_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Evaluation plots for the test40k runs (submitted by submit_test40k.sh
# with a dependency on the training jobs).

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

if [ -e "$OUT/test40k_comparison" ]; then
    echo "ERROR: $OUT/test40k_comparison already exists -- not overwriting" >&2
    exit 1
fi

python plot_eval.py \
    --runs   $OUT/test40k_edges_baseline_nopca_aux $OUT/test40k_domenc_nopca_aux \
    --labels "Summaries + edges (no PCA, aux)" "Pulse encoder (no PCA, aux)" \
    --out_dir $OUT/test40k_comparison
