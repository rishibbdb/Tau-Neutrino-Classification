#!/bin/bash --login
#SBATCH --job-name=Combine_5TeV
#SBATCH --ntasks=8       # number of CPUs
#SBATCH --mem=120G
#SBATCH --time=12:59:59
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/combine_5TeV.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

# export CUDA_VISIBLE_DEVICES=0
# echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
# echo "SLURM-assigned CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
# nvidia-smi
# python /mnt/scratch/baburish/doublepulse/gnn/Analysis/train.py --tau_dbs /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/combined_nutau_65TeV.db --nue_dbs /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/combined_nue_65TeV.db --geo /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/geometry_clean.csv --save_dir /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/dom_level_train_GAT/ --cache_dir /mnt/scratch/baburish/doublepulse/gnn/Analysis/dataset_cache --patience 10

python /mnt/scratch/baburish/doublepulse/gnn/Analysis/virgo/merge-5TeV-moresamples.py | tee /mnt/scratch/baburish/doublepulse/gnn/Analysis/virgo/5tev-combine-virgo.log

