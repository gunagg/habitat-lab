#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/flash1/rramrakhya6/habitat-web/habitat-lab
echo "Starting objectnav training"
echo "hab sim: ${PYTHONPATH}"

path=$1
python habitat_baselines/run.py --exp-config $path --run-type train
