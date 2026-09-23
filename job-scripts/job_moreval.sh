#!/bin/bash --login
#SBATCH --job-name=GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac
#SBATCH --ntasks=8       # number of CPUs
#SBATCH --mem=20G
#SBATCH --time=12:59:59
#SBATCH --job-name 22644-TPN-God
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/22644slurm_GAT_0p3valfrac.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

# export CUDA_VISIBLE_DEVICES=0
# echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
# echo "SLURM-assigned CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
# nvidia-smi
# python /mnt/scratch/baburish/doublepulse/gnn/Analysis/train.py --tau_dbs /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/combined_nutau_65TeV.db --nue_dbs /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/combined_nue_65TeV.db --geo /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/geometry_clean.csv --save_dir /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/dom_level_train_GAT/ --cache_dir /mnt/scratch/baburish/doublepulse/gnn/Analysis/dataset_cache --patience 10

python /mnt/scratch/baburish/doublepulse/gnn/Analysis/train_noedge.py --tau_dbs /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/combined_nutau_65TeV.db --nue_dbs /mnt/scratch/baburish/doublepulse/gnn/Analysis/data/combined_nue_65TeV.db --geo /mnt/scratch/baburish/doublepulse/gnn/Analysis/geometry_clean.csv --save_dir /mnt/scratch/baburish/doublepulse/gnn/Analysis/trained_model/NewRun_GAT_0p3dropout_weightdecay5e4_64batchsize_0p3valfrac/ --val_frac 0.3 --cache_dir /mnt/scratch/baburish/doublepulse/gnn/Analysis/dataset_cache_0p3valdir --patience 50 | tee /mnt/scratch/baburish/doublepulse/gnn/Analysis/c1.log

