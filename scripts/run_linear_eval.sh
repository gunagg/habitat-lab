#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting linear eval"
echo "hab sim: ${PYTHONPATH}"

model=$1

python examples/places_clustering.py --data /srv/flash1/skuhar6/scaling_crl/new_scaling/temp_scaling/data/places365_standard/ --model $model
