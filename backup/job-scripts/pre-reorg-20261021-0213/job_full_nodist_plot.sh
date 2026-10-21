#!/bin/bash --login
#SBATCH --job-name=full-plot
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/full_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Plots for the batch-size runs of submit_full_nodist_bs.sh vs the previous full-sample run
# (full_gat_cut10_ms1_arch_feats: with distance_to_cascade, no auxiliary targets, lr 3e-4, batch 64).

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

for r in full_nodist_aux1_bs64 full_nodist_aux1_bs512; do
    if [ ! -e "$OUT/$r/eval_val.npz" ]; then
        echo "ERROR: $r has not finished (no eval_val.npz) -- submit another --resume segment" >&2
        exit 1
    fi
done
if [ -e "$OUT/full_nodist_comparison" ]; then
    echo "ERROR: $OUT/full_nodist_comparison already exists -- not overwriting" >&2
    exit 1
fi

python plot_eval.py \
    --runs   $OUT/full_nodist_aux1_bs64 $OUT/full_nodist_aux1_bs512 $OUT/full_gat_cut10_ms1_arch_feats \
    --labels "no dist, aux 1, lr 1e-3, batch 64" "no dist, aux 1, lr 1e-3, batch 512" "previous full run (dist, no aux, batch 64)" \
    --out_dir $OUT/full_nodist_comparison
