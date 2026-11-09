#!/bin/bash --login
#SBATCH --job-name=train-65TeV
#SBATCH --ntasks=8       # number of CPUs
#SBATCH --gpus=a100:1   # number of GPUs
#SBATCH --mem=96G
#SBATCH --time=03:30:00  # well under 4h so the job stays in general-short
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/train65_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# 65 TeV (>65 TeV) network: the working-point model trained_model/GAT_0p3dropout_weightdecay5e4_64batchsize
# was trained with these settings (train.py defaults at the time):
#   trainer_noedge (3 x GATConv, no edge features) + PCA branch, 13 node features,
#   Gemini 65 TeV samples (data/combined_{nutau,nue}_65TeV.db from virgo/merge-65TeV-moresamples.py),
#   pulse charge > 0.1 PE, >= 2 strings, 1:1 balanced, val_frac 0.15, batch 64, lr 3e-4,
#   AdamW weight decay 5e-4, dropout 0.3, early stopping (best epoch 81, stopped at 121).
# Usage: sbatch job_65TeV.sh <run_name> <extra train.py args...>
#   refuses if trained_model/<run_name> or its log already exists

RUN=$1
shift
if [ -z "$RUN" ]; then
    echo "Usage: sbatch job_65TeV.sh <run_name> <extra train.py args...>" >&2
    exit 1
fi

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/trained_model
cd $ANA

if [[ " $* " == *" --resume "* ]]; then
    REDIRECT=">>"
else
    if [ -e "$OUT/$RUN" ] || [ -e "$OUT/$RUN.log" ]; then
        echo "ERROR: $OUT/$RUN or $OUT/$RUN.log already exists -- not overwriting" >&2
        exit 1
    fi
    REDIRECT=">"
fi

echo "Run: $RUN  extra args: $*"
echo "SLURM-assigned CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

# defaults first, then the job's own arguments (argparse keeps the LAST value)
CMD="python train.py --trainer trainer_noedge \
    --tau_dbs $ANA/data/combined_nutau_65TeV.db --nue_dbs $ANA/data/combined_nue_65TeV.db \
    --geo $ANA/geometry_clean.csv --cache_dir $ANA/dataset_cache \
    --epochs 150 --patience 40 --batch_size 64 --lr 3e-4 --max_hours 3.25 \
    --save_dir $OUT/$RUN $*"
eval "$CMD $REDIRECT $OUT/$RUN.log 2>&1"
status=$?
grep -E "Early|Best val|Time limit|already finished" $OUT/$RUN.log | tail -3
exit $status
