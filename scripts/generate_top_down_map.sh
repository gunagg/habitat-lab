#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

prefix=$1

python examples/rearrangement_replay.py --replay-episode data/hit_data/sample_hits.json.gz --output-prefix $prefix

python psiturk_dataset/upload_files_to_s3.py --file demos/ --s3-path data/hit_data/video/$prefix
python psiturk_dataset/upload_files_to_s3.py --file instructions.json --s3-path data/hit_data/instructions.json

