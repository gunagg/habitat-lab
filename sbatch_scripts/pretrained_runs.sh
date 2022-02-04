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
#TEST_EPISODE_COUNT=4200
NUM_PROCESSES=8
RUN_TRAIN_SCRIPT=true
RUN_EVAL_SCRIPT=false

EXP_NAME="objectnav_pretrained"
WEIGHTS_NAME="omnidata_DINO_02.pth"
BACKBONE="resnet50_gn"
#EXTRA_CMDS="RL.DDPPO.pretrained_encoder True \
#            RL.DDPPO.pretrained_goal_encoder True \
#            RL.DDPPO.train_encoder True \
#            RL.DDPPO.train_goal_encoder True \
#            RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            RL.POLICY.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 4 \
#            RL.POLICY.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 128 \
#            RL.POLICY.OBS_AUGMENTATIONS color_jitter-translate_v2 \
#            RL.PPO.weight_decay 1e-6 \
#            RL.PPO.resnet_baseplanes 64 \
#            RL.DDPPO.pretraining_type groupnorm \
#            ${TASK_CONFIG_SMALL}"
EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746"
SPLIT="train"
run_training 0
