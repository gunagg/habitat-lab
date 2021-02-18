#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

python psiturk_dataset/parser.py --replay-path data/hit_data/visualisation/unapproved_hits/ --output-path data/hit_data/hits.json
