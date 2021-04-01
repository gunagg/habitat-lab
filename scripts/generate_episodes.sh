#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

echo "Starting episode generation!"
echo "data/scene_datasets/habitat-test-scenes/big_house.glb"
python psiturk_dataset/task/generate_object_locations.py --scenes data/scene_datasets/habitat-test-scenes/big_house.glb --task-config psiturk_dataset/task/rearrangement.yaml --number_retries_per_target 10000 --output big_house.json -n 5