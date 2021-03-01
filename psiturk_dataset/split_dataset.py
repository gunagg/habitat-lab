import argparse
import gzip
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold


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

    sk_fold = StratifiedKFold(n_splits=5, shuffle=True)
    train_idxs = []
    eval_idxs = []
    for train_episode_idx, eval_episode_idx in sk_fold.split(y_idxs, labels_filtered):
        train_episode_idx = [y_idxs[i] for i in train_episode_idx.tolist()]
        eval_episode_idx = [y_idxs[i] for i in eval_episode_idx.tolist()]

        # train_episode_idx, eval_episode_idx = train_test_split(y_idxs, test_size=0.25, stratify=labels_filtered)
        print("Train split: {}, max: {}".format(len(train_episode_idx), max(train_episode_idx)))
        print("Train split overlap exclude: {}".format(set(train_episode_idx).intersection(exclude_idx)))
        print("Eval split: {}".format(len(eval_episode_idx)))

        train_episode_idx.extend(exclude_idx)

        print("Train episode indices: {}, Unique indices {}".format(len(train_episode_idx), len(set(train_episode_idx))))
        print("Eval episode indices: {}, Unique indices {}".format(len(eval_episode_idx), len(set(eval_episode_idx))))

        print("Train episode indices: {}".format(train_episode_idx[:20]))
        print("Eval episode indices: {}".format(eval_episode_idx[:20]))

        print("Overlap indices: {}".format(len(set(train_episode_idx).intersection(set(eval_episode_idx)))))

        train_idxs.append(train_episode_idx)
        eval_idxs.append(eval_episode_idx)

    return train_idxs, eval_idxs


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


def get_episode_data(data, episode_ids):
    split_data = {
        "episodes": [],
    }
    instructions = []
    for i in range(len(data["episodes"])):
        episode = data["episodes"][i]
        episode_id = episode["episode_id"]
        if episode_id in episode_ids:
            instructions.append(episode["instruction"]["instruction_text"])
            split_data["episodes"].append(episode)
    
    split_data["instruction_vocab"] = {
        "sentences": list(set(instructions))
    }
    return split_data


def get_episode_ids(data, indexes):
    episode_ids = []
    for i in range(len(data["episodes"])):
        episode = data["episodes"][i]
        if i in indexes:
            episode_ids.append(episode["episode_id"])
    
    return episode_ids


def get_episode_ids(data):
    episode_ids = []
    for i in range(len(data["episodes"])):
        episode = data["episodes"][i]
        episode_ids.append(episode["episode_id"])
    
    return episode_ids


def split_data(path, output_path):
    data = load_dataset(path)
    train_idxs, eval_idxs = train_test_split_episodes(data)

    for i, (train_idx, eval_idx) in enumerate(zip(train_idxs, eval_idxs)):
        print("Split {}: Train: {}, Eval: {}".format(i, len(train_idx), len(eval_idx)))
        train_data_path = output_path + "train_{}.json".format(i)
        eval_data_path = output_path + "eval_{}.json".format(i)
        print(train_data_path)
        save_splits(data, train_idx, train_data_path)
        save_splits(data, eval_idx, eval_data_path)


def validate_split_data(path, train_data_path, eval_data_path):
    data = load_dataset(path)
    train_idx, eval_idx = train_test_split_episodes(data)
    train_episode_ids = get_episode_ids(data, train_idx)
    eval_episodes_ids = get_episode_ids(data, eval_idx)
    validate_existing_split_overlap(train_data_path, eval_data_path, train_episode_ids, eval_episodes_ids)


def create_split_from_existing_data(path, train_data_path, eval_data_path, output_path):
    data = load_dataset(path)
    train_data = load_dataset(train_data_path)
    eval_data = load_dataset(eval_data_path)

    train_episode_ids = get_episode_ids(train_data)
    eval_episodes_ids = get_episode_ids(eval_data)

    print("Length train ep: {}".format(len(train_episode_ids)))
    print("Length eval ep: {}".format(len(eval_episodes_ids)))

    train_episodes_new = get_episode_data(data, train_episode_ids)
    eval_episodes_new = get_episode_data(data, eval_episodes_ids)

    print("\n\nLength train eps: {}".format(len(train_episodes_new['episodes'])))
    print("Length eval eps: {}".format(len(eval_episodes_new['episodes'])))

    train_output_path = output_path + "train.json"
    eval_output_path = output_path + "eval.json"
    write_json(train_episodes_new, train_output_path)
    write_json(eval_episodes_new, eval_output_path)
    write_gzip(train_output_path, train_output_path)
    write_gzip(eval_output_path, eval_output_path)


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


def validate_existing_split_overlap(train_data_path, eval_data_path, train_ep_ids, eval_ep_ids):
    train_data = load_dataset(train_data_path)
    eval_data = load_dataset(eval_data_path)

    train_episode_ids = []
    train_instructions = []
    train_scene_map = {}
    for episode in train_data["episodes"]:
        train_episode_ids.append(episode["episode_id"])

    eval_episode_ids = []
    eval_instructions = []
    eval_scene_map = {}
    for episode in eval_data["episodes"]:
        eval_episode_ids.append(episode["episode_id"])

    print("\nOverlap train episodes: {}".format(len(set(train_episode_ids).intersection(set(train_ep_ids)))))
    print("Unique train episodes existing: {}".format(len(set(train_episode_ids))))
    print("Unique train episodes new: {}".format(len(set(train_ep_ids))))

    print("\nOverlap eval episodes: {}".format(len(set(eval_episode_ids).intersection(set(eval_ep_ids)))))
    print("Unique eval episodes existing: {}".format(len(set(eval_episode_ids))))
    print("Unique eval episodes new: {}".format(len(set(eval_ep_ids))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_data/hits_max_length_1500.json.gz"
    )
    parser.add_argument(
        "--train-data-path", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--eval-data-path", type=str, default="data/datasets/object_rearrangement/v0/eval/eval.json.gz"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/hit_approvals/dataset/"
    )
    parser.add_argument(
        "--validate", dest='validate', action='store_true'
    )
    parser.add_argument(
        "--validate-before-split", dest='validate_before_split', action='store_true'
    )
    parser.add_argument(
        "--create-split", dest='create_split', action='store_true'
    )
    args = parser.parse_args()

    if args.create_split:
        create_split_from_existing_data(args.input_path, args.train_data_path, args.eval_data_path, args.output_path)
    elif args.validate_before_split:
        validate_split_data(args.input_path, args.train_data_path, args.eval_data_path)
    elif not args.validate:
        split_data(args.input_path, args.output_path)
    else:
        validate_data(args.input_path, args.train_data_path, args.eval_data_path)


if __name__ == "__main__":
    main()
