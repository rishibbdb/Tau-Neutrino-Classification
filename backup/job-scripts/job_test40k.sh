#!/bin/bash --login
#SBATCH --job-name=test40k-train
#SBATCH --ntasks=8       # number of CPUs
#SBATCH --gpus=a100:1   # number of GPUs
#SBATCH --mem=48G
#SBATCH --time=03:59:00  # <= 4h so the job is eligible for general-short
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test40k_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# One training run on the 40k-events-per-flavor kinematically matched samples.
# Usage: sbatch job_test40k.sh <run_name> <extra train.py args...>
#   e.g. sbatch job_test40k.sh test40k_domenc_nopca --trainer trainer_domenc --no_pca
# Submit all runs + plots with job-scripts/submit_test40k.sh.
# Refuses to run if the run dir or its log already exists, so earlier runs,
# plots and logs are never overwritten.

RUN=$1
shift
if [ -z "$RUN" ]; then
    echo "Usage: sbatch job_test40k.sh <run_name> <extra train.py args...>" >&2
    exit 1
fi

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model5TeV
cd $ANA

if [ -e "$OUT/$RUN" ] || [ -e "$OUT/$RUN.log" ]; then
    echo "ERROR: $OUT/$RUN or $OUT/$RUN.log already exists -- not overwriting" >&2
    exit 1
fi

echo "Run: $RUN  extra args: $*"
echo "SLURM-assigned CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

python train.py "$@" \
    --tau_dbs $ANA/data/combined_nutau_5TeV_40k_matched.db --nue_dbs $ANA/data/combined_nue_5TeV_40k_matched.db \
    --geo $ANA/geometry_clean.csv --cache_dir $ANA/dataset_cache_5TeV \
    --epochs 150 --patience 25 --batch_size 64 \
    --save_dir $OUT/$RUN > $OUT/$RUN.log 2>&1
status=$?
grep -E "Early|Best val" $OUT/$RUN.log
exit $status
