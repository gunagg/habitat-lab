#!/bin/bash
source ./sbatch_scripts/training_function.sh

set -x

## Slurm 
REPO_NAME="objectnav"
MAIN_USER="alexclegg"
REPO_PATH="/private/home/${MAIN_USER}/karmesh_code/${REPO_NAME}/habitat-lab"
PARTITION="learnfair,learnlab,devlab"
SPLIT="train"
VAL_SPLIT="val"
BASE_TASK_CONFIG_PATH="${REPO_PATH}/configs/tasks/objectnav_mp3d_il.yaml"
EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_rgbd_objectnav.yaml"
NODES=8
#WANDB_MODE="online"
ENVIRONMENT="objectnav_mp3d_40k"
VIDEO_OPTION="[]"
#NUM_STEPS=5e8
NUM_UPDATES=16000
TIME="72:00:00"
#NUM_ENV=10
TEST_EPISODE_COUNT=10000
NUM_PROCESSES=8
RUN_TRAIN_SCRIPT=false
RUN_EVAL_SCRIPT=true

EXP_NAME="objectnav_pretrained"
WEIGHTS_NAME="omnidata_DINO_02.pth"
BACKBONE="resnet50_gn"
EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
	    IL.BehaviorCloning.num_steps 64 \
	    IL.BehaviorCloning.num_mini_batch 2"
SPLIT="train"
run_training 0

#EXP_NAME="objectnav_pretrained_nccl"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#	    IL.distrib_backend NCCL"
#SPLIT="train"
#CHKP_NAME="ckpt.17.pth"
#run_training 0

#EXP_NAME="objectnav_pretrained_semantics"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL"
#EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_rgbd_objectnav.yaml"
#SPLIT="train"
#run_training 0
