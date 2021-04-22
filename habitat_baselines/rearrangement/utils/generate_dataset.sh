#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Generate IL episode dataset"
echo "hab sim: ${PYTHONPATH}"

scene=$1

python habitat_baselines/rearrangement/utils/generate_dataset.py --episodes data/datasets/object_rearrangement/v4/train/train.json.gz --mode train --scene $scene
