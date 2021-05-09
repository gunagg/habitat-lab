#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

echo "Starting episode generation!"
echo "Episodes v6"

sceneId=$1

python psiturk_dataset/generator/shortest_path_trajectories.py --scene $sceneId --episodes data/datasets/object_rearrangement/v6/train/train.json.gz
