#!/bin/bash
#SBATCH --job-name=ddp_onav
#SBATCH --gres gpu:1
#SBATCH --nodes 2
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=long
#SBATCH --constraint=rtx_6000
#SBATCH --output=slurm_logs/ddppo-%j.out
#SBATCH --error=slurm_logs/ddppo-%j.err

source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

sensor=$1
set -x

echo "In ObjectNav IL DDP"
srun python -u -m habitat_baselines.run \
--exp-config habitat_baselines/config/objectnav/il_distrib_objectnav.yaml \
--run-type train
