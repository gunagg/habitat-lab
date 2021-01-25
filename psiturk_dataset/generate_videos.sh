#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

wget https://habitatonweb.cloudcv.org:8000/data/hit_data/visualisation/unapproved_hits.zip
unzip unapproved_hits.zip 
python psiturk_dataset/parser.py --replay-path data/hit_data/visualisation/unapproved_hits --output-path data/hit_data/visualisation/hits.json
python examples/rearrangement_replay.py --replay-episode data/hit_data/visualisation/hits.json.gz --output-prefix demo --restore-state
zip -r unapproved_hits_demos.zip demos/
zip -r unapproved_hits_instructions.zip instructions.json
rm unapproved_hits.zip

scp -i ~/.ssh/id_rsa.pem unapproved_hits_demos.zip ubuntu@${MTURK_HOST}:/home/ubuntu/visualisation/
scp -i ~/.ssh/id_rsa.pem unapproved_hits_instructions.zip ubuntu@${MTURK_HOST}:/home/ubuntu/visualisation/

rm unapproved_hits_demos.zip
rm unapproved_hits_instructions.zip