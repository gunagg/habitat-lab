#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting objectnav training"
echo "hab sim: ${PYTHONPATH}"

python habitat_baselines/run.py --exp-config habitat_baselines/config/objectnav/il_objectnav.yaml --run-type train
