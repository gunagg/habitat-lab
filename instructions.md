## Object rearrangement replay

### Steps

1. Run `python examples/rearrangement_replay.py` with following parameters:
    - `--replay-episode` - path to repaly episode gzipped json file
    - `--output-prefix`  - name of the output video file
   
   Usage example:
   ```
   python examples/rearrangement_replay.py --replay-episode data/replays/{replay}.json.gz --output-prefix {prefix}
   ```
