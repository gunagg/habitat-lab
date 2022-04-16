#!/bin/bash
source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-web

cd /srv/flash1/rramrakhya6/habitat-web/habitat-lab
echo "Starting eval"
echo "hab sim: ${PYTHONPATH}"

path=$1
python habitat_baselines/run.py --exp-config $path --run-type eval
