#!/bin/bash --login
#SBATCH --job-name=test-eval
#SBATCH --ntasks=8
#SBATCH --gpus=a100:1
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/test_eval_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Evaluate a trained run on an independent TEST sample (Virgo *_test directories) with evaluate_test.py.
# Usage: sbatch job_test_eval.sh <run_dir> <test_tau_db> <test_nue_db> <test_cache_dir>
#   writes <run_dir>/test/{eval_test.npz,scores.csv} and <run_dir>/test.log

RUN_DIR=$1; TAU=$2; NUE=$3; CACHE=$4
if [ -z "$CACHE" ]; then
    echo "Usage: sbatch job_test_eval.sh <run_dir> <test_tau_db> <test_nue_db> <test_cache_dir>" >&2
    exit 1
fi

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate
cd /mnt/scratch/baburish/doublepulse/gnn/Analysis

if [ -e "$RUN_DIR/test" ] || [ -e "$RUN_DIR/test.log" ]; then
    echo "ERROR: $RUN_DIR/test or $RUN_DIR/test.log already exists -- not overwriting" >&2
    exit 1
fi
echo "Run: $RUN_DIR  test: $TAU $NUE  cache: $CACHE"
python evaluate_test.py --run_dir $RUN_DIR --tau_dbs $TAU --nue_dbs $NUE --cache_dir $CACHE \
    --out_dir $RUN_DIR/test > $RUN_DIR/test.log 2>&1
status=$?
tail -2 $RUN_DIR/test.log
exit $status
