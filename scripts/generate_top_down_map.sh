#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

cd /srv/share3/rramrakhya6/habitat-lab
echo "Starting video generation"
echo "hab sim: ${PYTHONPATH}"

task=$1
path=$2

if [[ $task == "objectnav" ]]; then
    echo "in ObjectNav stats"
    # python examples/objectnav_replay.py --replay-episode data/datasets/objectnav_mp3d_v2/coverage_sample/coverage_sample.json.gz --step-env
    python examples/objectnav_replay.py --replay-episode $path --step-env --output-prefix demos --metrics
else
    python examples/rearrangement_replay.py --replay-episode $path
fi
