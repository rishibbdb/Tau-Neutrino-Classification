#!/bin/bash --login
#SBATCH --job-name=v65-plot
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/baburish/doublepulse/gnn/Analysis/logs/virgo65_slurm_%j.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

# Evaluation plots for the 65 TeV network (submit_virgo65.sh) on the independent TEST sample
# (evaluate_test.py output in <run>/test/); validation-set plots are kept for reference.

eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.1/setup.sh)
source /mnt/scratch/baburish/doublepulse/gnn/event-nn-env/bin/activate

ANA=/mnt/scratch/baburish/doublepulse/gnn/Analysis
OUT=$ANA/virgo_65TeV
RUN=virgo65_gat_arch_feats_aux1
cd $ANA

if [ ! -e "$OUT/$RUN/test/eval_test.npz" ]; then
    echo "ERROR: no test evaluation ($OUT/$RUN/test/eval_test.npz) -- run job_test_eval.sh" >&2
    exit 1
fi
for d in evaluation_test evaluation_validation; do
    if [ -e "$OUT/$d" ]; then
        echo "ERROR: $OUT/$d already exists -- not overwriting" >&2
        exit 1
    fi
done

python plot_eval.py --runs $OUT/$RUN/test --eval_file eval_test.npz \
    --labels "65 TeV test: GAT, brightest/causal features, aux 1" --out_dir $OUT/evaluation_test
python plot_eval.py --runs $OUT/$RUN --labels "65 TeV validation" --out_dir $OUT/evaluation_validation
