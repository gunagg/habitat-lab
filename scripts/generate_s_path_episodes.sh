#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

echo "Starting episode generation!"
echo "Episodes v6"

sceneId=$1
path=$2

python psiturk_dataset/generator/shortest_path_trajectories.py --scene $sceneId --episodes $path
