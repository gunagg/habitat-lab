#!/bin/bash
#SBATCH --job-name=hab_web_eval
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --partition=debug
#SBATCH --constraint=a40
#SBATCH --output=slurm_logs/eval/eval-%j.out
#SBATCH --error=slurm_logs/eval/eval-%j.err

source /srv/flash1/gaggarwal32/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-web

cd /srv/flash1/gaggarwal32/habitat-lab-habitat-web

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

EVAL_CKPT_PATH_DIR="/srv/flash1/gaggarwal32/data/ovrl_habitat_on_web/ovrl_objectnav_rgb_fintuned_best_model.pth"
DATA_PATH="data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz"
ATTRIBUTE_NAV_COLOR_DATA_PATH="data/datasets/attributenav/hm3d/v1-color/{split}/{split}.json.gz"
ATTRIBUTE_NAV_MATERIAL_DATA_PATH="data/datasets/attributenav/hm3d/v1-material/{split}/{split}.json.gz"

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

path="/srv/flash1/gaggarwal32/habitat-lab-habitat-web/habitat_baselines/config/objectnav/ddpil_ssl_rgbd_objectnav_zson.yaml"
set -x

echo "In ObjectNav Env DDP"
srun python -u -m habitat_baselines.run \
--run-type eval \
--exp-config  $path \
EVAL.USE_CKPT_CONFIG False \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
EVAL.SPLIT "val" \
NUM_PROCESSES 20 \
NUM_ENVS 20 \
TASK_CONFIG.DATASET.DATA_PATH $ATTRIBUTE_NAV_COLOR_DATA_PATH \

