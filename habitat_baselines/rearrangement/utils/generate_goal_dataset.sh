#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat

cd /srv/flash1/rramrakhya6/habitat-web/habitat-lab
echo "Generate agile goal dataset"
echo "hab sim: ${PYTHONPATH}"

scene=$1

python habitat_baselines/rearrangement/utils/generate_goal_dataset.py --episodes data/datasets/object_rearrangement/v3/train/train.json.gz --mode train_goals --scene $scene
