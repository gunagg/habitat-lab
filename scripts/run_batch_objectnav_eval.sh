#!/bin/bash
#SBATCH --job-name=onav_eval
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=long
#SBATCH --qos=ram-special
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/eval/eval-%j.out
#SBATCH --error=slurm_logs/eval/eval-%j.err

source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

path=$1
set -x

echo "In ObjectNav Env DDP"
srun python -u -m habitat_baselines.run \
--exp-config  $path \
--run-type eval
