#!/bin/bash --login
#SBATCH --job-name=test20k-domenc
#SBATCH --ntasks=8       # number of CPUs
#SBATCH --gpus=a100:1   # number of GPUs
#SBATCH --mem=48G
#SBATCH --time=12:59:59
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test20k_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# 20k-events-per-flavor comparison: summaries+edges baseline vs per-DOM pulse
# encoder (with / without PCA branch), then evaluation plots.
# Refuses to run if any output dir or log already exists, so earlier runs,
# plots and logs are never overwritten.

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

RUNS=(test20k_edges_baseline test20k_domenc test20k_domenc_nopca)
for r in "${RUNS[@]}" test20k_comparison; do
    if [ -e "$OUT/$r" ] || [ -e "$OUT/$r.log" ]; then
        echo "ERROR: $OUT/$r or $OUT/$r.log already exists -- not overwriting" >&2
        exit 1
    fi
done

echo "SLURM-assigned CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

COMMON="--tau_dbs $ANA/data/combined_nutau_5TeV_20k.db --nue_dbs $ANA/data/combined_nue_5TeV_20k.db \
        --geo $ANA/geometry_clean.csv --cache_dir $ANA/dataset_cache_5TeV \
        --epochs 150 --patience 25 --batch_size 64"

set -e
python train.py --trainer trainer        $COMMON          --save_dir $OUT/test20k_edges_baseline > $OUT/test20k_edges_baseline.log 2>&1
python train.py --trainer trainer_domenc $COMMON          --save_dir $OUT/test20k_domenc         > $OUT/test20k_domenc.log 2>&1
python train.py --trainer trainer_domenc $COMMON --no_pca --save_dir $OUT/test20k_domenc_nopca   > $OUT/test20k_domenc_nopca.log 2>&1

python plot_eval.py \
    --runs   $OUT/test20k_edges_baseline $OUT/test20k_domenc $OUT/test20k_domenc_nopca \
    --labels "Summaries + edges" "Pulse encoder" "Pulse encoder, no PCA" \
    --out_dir $OUT/test20k_comparison

for r in "${RUNS[@]}"; do echo "== $r"; grep -E "Early|Best val" $OUT/$r.log; done
