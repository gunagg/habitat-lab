import argparse
import gzip
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def load_dataset(path):
    with gzip.open(path, "rb") as file:
        data = json.loads(file.read(), encoding="utf-8")
    return data


def load_json_dataset(path):
    file = open(path, "r")
    data = json.loads(file.read())
    return data


def train_test_split_episodes(data):
    X = []
    y = []
    instruction_map = {}
    for episode in data["episodes"]:
        yy_id = "{}:{}".format(episode["scene_id"], episode["instruction"]["instruction_text"])
        X.append("{}".format(episode["episode_id"]))
        y.append(yy_id)
        if yy_id not in instruction_map.keys():
            instruction_map[yy_id] = 0
        instruction_map[yy_id] += 1

    single_instruction = []
    exclude_idx = []
    for key, value in instruction_map.items():
        if value == 1:
            single_instruction.append(key)
            exclude_idx.append(y.index(key))
    
    instructions = list(instruction_map.keys())
    
    labels = [instructions.index(i) for i in y]
    y = [i for i in range(len(y))]

    y_idxs = []
    labels_filtered = []
    print("Total episodes: {}, Labels: {}".format(len(y), len(labels)))
    for ii in range(len(y)):
        y_i = y[ii]
        label_i = labels[ii]
        if y_i not in exclude_idx:
            y_idxs.append(y_i)
            labels_filtered.append(label_i)
    
    print("Single instance of instructions: {}".format(len(single_instruction)))
    print("Total episodes: {}, Filtered: {}".format(len(y), len(y_idxs)))


    train_episode_idx, eval_episode_idx = train_test_split(y_idxs, test_size=0.25, stratify=labels_filtered)
    print("Train split: {}, max: {}".format(len(train_episode_idx), max(train_episode_idx)))
    print("Eval split: {}".format(len(eval_episode_idx)))

    train_episode_idx.extend(exclude_idx)

    print("Train episode indices: {}, Unique indices {}".format(len(train_episode_idx), len(set(train_episode_idx))))
    print("Eval episode indices: {}, Unique indices {}".format(len(eval_episode_idx), len(set(eval_episode_idx))))

    print("Train episode indices: {}".format(train_episode_idx[:20]))
    print("Eval episode indices: {}".format(eval_episode_idx[:20]))

    print("Overlap indices: {}".format(len(set(train_episode_idx).intersection(set(eval_episode_idx)))))

    return train_episode_idx, eval_episode_idx


def save_splits(data, indexes, path):
    split_data = {
        "episodes": [],
    }
    instructions = []
    for i in range(len(data["episodes"])):
        episode = data["episodes"][i]
        if i in indexes:
            instructions.append(episode["instruction"]["instruction_text"])
            split_data["episodes"].append(episode)
    
    split_data["instruction_vocab"] = {
        "sentences": list(set(instructions))
    }
    
    write_json(split_data, path)
    write_gzip(path, path)


def split_data(path, train_data_path, eval_data_path):
    data = load_dataset(path)
    train_idx, eval_idx = train_test_split_episodes(data)
    save_splits(data, train_idx, train_data_path)
    save_splits(data, eval_idx, eval_data_path)


def validate_data(path, train_data_path, eval_data_path):
    train_data = load_json_dataset(train_data_path)
    eval_data = load_json_dataset(eval_data_path)

    train_episode_ids = []
    train_instructions = []
    train_scene_map = {}
    for episode in train_data["episodes"]:
        train_episode_ids.append(episode["episode_id"])
        train_instructions.append(episode["instruction"]["instruction_text"])
        scene_id = episode["scene_id"]
        if scene_id not in train_scene_map.keys():
            train_scene_map[scene_id] = 0
        train_scene_map[scene_id] += 1

    eval_episode_ids = []
    eval_instructions = []
    eval_scene_map = {}
    for episode in eval_data["episodes"]:
        eval_episode_ids.append(episode["episode_id"])
        eval_instructions.append(episode["instruction"]["instruction_text"])
        scene_id = episode["scene_id"]
        if scene_id not in eval_scene_map.keys():
            eval_scene_map[scene_id] = 0
        eval_scene_map[scene_id] += 1

    print("\nOverlap episodes: {}".format(len(set(train_episode_ids).intersection(set(eval_episode_ids)))))
    print("Unique train episodes: {}".format(len(set(train_episode_ids))))
    print("Unique eval episodes: {}".format(len(set(eval_episode_ids))))

    print("\nOverlap instructions: {}".format(len(set(train_instructions).intersection(set(eval_instructions)))))
    print("Unique instructions train episodes: {}".format(len(set(train_instructions))))
    print("Unique instructions eval episodes: {}".format(len(set(eval_instructions))))

    print("\nTrain data scene map: {}".format(train_scene_map))
    print("Eval data scene map: {}".format(eval_scene_map))
    
    train_scene_data = pd.DataFrame(train_scene_map.items(), columns=["scene", "num_episodes"])
    eval_scene_data = pd.DataFrame(eval_scene_map.items(), columns=["scene", "num_episodes"])

    sns.barplot(y="scene", x="num_episodes", data=train_scene_data)
    plt.gca().set(title="Train data distribution")
    plt.savefig("analysis/train_data_distribution.jpg")
    plt.clf()

    sns.barplot(y="scene", x="num_episodes", data=eval_scene_data)
    plt.gca().set(title="Eval data distribution")
    plt.savefig("analysis/eval_data_distribution.jpg")
    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_data/hits_max_length_1500.json.gz"
    )
    parser.add_argument(
        "--train-data-path", type=str, default="data/datasets/object_rearrangement/v0/train/train.json"
    )
    parser.add_argument(
        "--eval-data-path", type=str, default="data/datasets/object_rearrangement/v0/eval/eval.json"
    )
    parser.add_argument(
        "--validate", dest='validate', action='store_true'
    )
    args = parser.parse_args()

    if not args.validate:
        split_data(args.input_path, args.train_data_path, args.eval_data_path)
    else:
        validate_data(args.input_path, args.train_data_path, args.eval_data_path)


if __name__ == "__main__":
    main()
