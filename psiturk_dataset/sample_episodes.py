import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def sample_episodes(path, output_path, per_scene_limit=10):
    episode_file = open(path, "r")
    data = json.loads(episode_file.read())

    print("Number of episodes {}".format(len(data["episodes"])))

    sample_episodes = {}
    sample_episodes["instruction_vocab"] = data["instruction_vocab"]
    sample_episodes["episodes"] = []
    scene_map = {}
    for episode in data["episodes"]:
        scene_id = episode["scene_id"]

        if scene_id not in scene_map.keys():
            scene_map[scene_id] = 0
        scene_map[scene_id] += 1 

        if scene_map[scene_id] <= per_scene_limit:
            sample_episodes["episodes"].append(episode)
    
    print("Sampled episodes: {}".format(len(sample_episodes["episodes"])))
    
    write_json(sample_episodes, output_path)
    write_gzip(output_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_data/hits.json"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/sample_hits.json"
    )
    parser.add_argument(
        "--per-scene-limit", type=int, default=10
    )
    args = parser.parse_args()
    sample_episodes(args.input_path, args.output_path, args.per_scene_limit)
