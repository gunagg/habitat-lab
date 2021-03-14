#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

prefix=$1

#wget https://habitat-on-web.s3.amazonaws.com/data/hit_data/instructions.json
wget https://habitat-on-web.s3.amazonaws.com/data/hit_data/unapproved_hits.zip

unzip -o unapproved_hits.zip 
python psiturk_dataset/parser.py --replay-path data/hit_data/visualisation/unapproved_hits --output-path data/hit_data/visualisation/hits.json
python examples/rearrangement_replay.py --replay-episode data/hit_data/visualisation/hits.json.gz --output-prefix $prefix --restore-state

rm unapproved_hits.zip

python psiturk_dataset/upload_files_to_s3.py --file demos/ --s3-path data/hit_data/video/$prefix
python psiturk_dataset/upload_files_to_s3.py --file instructions.json --s3-path data/hit_data/instructions.json

rm instructions.json
rm data/hit_data/visualisation/unapproved_hits/*
rm demos/*

current_dt=$(date '+%Y-%m-%d')
cp data/hit_data/visualisation/hits.json.gz data/live_hits/live_hits_${current_dt}.json.gz
