#!/bin/bash
source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate habitat-web

cd /srv/flash1/rramrakhya6/habitat-web/habitat-lab
echo "Starting eval"
echo "hab sim: ${PYTHONPATH}"

distrib=$1

if [[ $distrib == "distrib" ]]; then
    echo "in distrib eval"
    python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/il_distrib_object_rearrangement.yaml --run-type eval
else
    python habitat_baselines/run.py --exp-config habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml --run-type eval
fi
