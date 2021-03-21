import argparse
import gzip
import json
import random
import numpy as np

from tqdm import tqdm


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def load_json_dataset(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def get_random_int(lb, ub):
    return random.randint(lb, ub)


def generate_closer_initialization(input_path, output_path, num_steps):
    data = load_dataset(input_path)

    for ep_id, episode in tqdm(enumerate(data["episodes"])):
        first_grab_action_index = 0
        for i, step in enumerate(episode["reference_replay"]):
            if step.get("action") == "grabReleaseObject" and step["is_grab_action"] and step["action_data"]["gripped_object_id"] != -1:
                first_grab_action_index = i - 5
                break
        
        # Get a random agent position closer to object
        episode_start_index = max(0, first_grab_action_index - num_steps)
        step_index = get_random_int(episode_start_index, first_grab_action_index)
        step = episode["reference_replay"][step_index]
        episode["start_position"] = step["agent_state"]["position"]
        episode["start_rotation"] = step["agent_state"]["rotation"]
        # episode["start_index"] = step_index
    
    write_json(data, output_path)
    write_gzip(output_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/datasets/object_rearrangement/v0/train/train_pruned.json"
    )
    parser.add_argument(
        "--num-steps", type=int, default=50
    )
    args = parser.parse_args()

    generate_closer_initialization(args.input_path, args.output_path, args.num_steps)


if __name__ == "__main__":
    main()
