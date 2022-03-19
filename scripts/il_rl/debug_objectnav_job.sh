#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=short
#SBATCH --qos=ram-special
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/ddppo-il-rl-%j.out
#SBATCH --error=slurm_logs/ddppo-il-rl-%j.err
#SBATCH --requeue

source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1
set -x

srun python -u -m habitat_baselines.run \
--exp-config $config \
--run-type train
