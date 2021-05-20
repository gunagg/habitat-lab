#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

echo "Starting episode generation!"
echo "Episodes v6"

task=$1
sceneId=$2
path=$3

if [[ $task == "objectnav" ]]; then
    echo "ObjectNav generator"
    python psiturk_dataset/generator/objectnav_shortest_path_generator.py --episodes data/datasets/objectnav_mp3d_v2/train/train.json.gz --output-path data/episodes/s_path_objectnav
else
    python psiturk_dataset/generator/shortest_path_trajectories.py --output-path data/episodes/s_path_pick_and_place/eval --scene $sceneId --episodes $path
fi
