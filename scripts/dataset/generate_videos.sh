#!/bin/bash
source /srv/share3/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat-3

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

prefix=$1
task=$2
output_path=$3
scene_dataset=$4

rm sample_unapproved_hits.zip
wget https://habitat-on-web.s3.amazonaws.com/data/unprocessed_hits/visualization/sample_unapproved_hits.zip

if [[ $task == "objectnav" ]]; then
    echo "in ObjectNav parsing"
    # Generate replays
    python psiturk_dataset/parsing/parse_objectnav_dataset.py --path sample_unapproved_hits.zip --output-path $output_path/content/ --scene-dataset $scene_dataset
    python examples/objectnav_replay.py --replay-episode $output_path/train.json.gz --step-env --output-prefix $prefix
else
    python psiturk_dataset/parsing/parser.py --replay-path data/hit_data/visualisation/unapproved_hits --output-path data/hit_data/visualisation/hits.json
    python examples/rearrangement_replay.py --replay-episode data/hit_data/visualisation/hits.json.gz --output-prefix $prefix --restore-state
fi

rm sample_unapproved_hits.zip

python psiturk_dataset/utils/upload_files_to_s3.py --file demos/ --s3-path data/hit_data/video/$prefix
python psiturk_dataset/utils/upload_files_to_s3.py --file instructions.json --s3-path data/hit_data/instructions.json

rm instructions.json
rm demos/*

# current_dt=$(date '+%Y-%m-%d')
# cp data/hit_data/visualisation/hits.json.gz data/live_hits/live_hits_${current_dt}.json.gz
