import argparse
from collections import defaultdict
import copy
from email.policy import default
import gzip
import glob
import os
import json

from tqdm import tqdm
from habitat.datasets.utils import VocabFromText
from psiturk_dataset.utils.utils import load_dataset, load_json_dataset, write_json, write_gzip, load_vocab


def get_episode_ids(path=""):
    dataset = load_dataset(path)
    episode_ids = [episode['episode_id'] for episode in dataset['episodes']]
    return set(episode_ids), dataset


def is_in_train(train_episode_ids, val_episode_ids, test_episode_ids, episode_id):
    if episode_id in train_episode_ids:
        return "train"
    if episode_id in val_episode_ids:
        return "val"
    if episode_id in test_episode_ids:
        return "test"
    return "none"


def get_episodes(dataset, episode_ids):
    new_dataset = copy.deepcopy(dataset)
    episodes = []
    for episode in dataset["episodes"]:
        if episode["episode_id"] in episode_ids:
            episodes.append(episode)
    new_dataset["episodes"] = episodes
    return new_dataset


def save_splits(dataset, episode_id_dict, split_id, output_dir):
    prefix = os.path.join(output_dir, "{}/{}/{}.json")
    for split, episode_ids in episode_id_dict.items():
        output_path = prefix.format(split_id, split, split)

        output_dir = "/".join(output_path.split("/")[:-1])
        os.makedirs(output_dir, exist_ok=True)

        new_dataset = get_episodes(dataset, episode_ids)
        print("Split: {}, Num episodes: {}".format(split, len(new_dataset["episodes"])))

        write_json(new_dataset, output_path)
        write_gzip(output_path, output_path)


def get_split_episode_ids(dataset_index, split_id):
    episode_id_dict = defaultdict(list)
    
    for episode in dataset_index:
        split = episode["splits"][split_id]
        episode_id_dict[split].append(episode["episode_id"])
    return episode_id_dict


def generate_splits(dataset_path, index_file, split_id, output_dir):
    dataset_index = load_json_dataset(index_file)
    episode_id_dict = get_split_episode_ids(dataset_index, split_id)
    print("Total episodes: {}".format(len(dataset_index)))
    print("Splits: {}".format(list(episode_id_dict.keys())))

    dataset = load_dataset(dataset_path)
    print("Full dataset length: {}".format(len(dataset["episodes"])))
    save_splits(dataset, episode_id_dict, split_id, output_dir)


def create_dataset_index_file(index_file_path):
    index_file = []
    paths = {
        "unseen_scenes-train": "data/datasets/object_rearrangement/unseen_scenes/train/train.json.gz",
        "unseen_instructions-train": "data/datasets/object_rearrangement/unseen_instructions/train/train.json.gz",
        "unseen_inits-train": "data/datasets/object_rearrangement/v4/train/train.json.gz",
        "unseen_scenes-test": "data/datasets/object_rearrangement/unseen_scenes/eval/eval.json.gz",
        "unseen_instructions-val": "data/datasets/object_rearrangement/unseen_instructions/val/val.json.gz",
        "unseen_inits-val": "data/datasets/object_rearrangement/v4/val/val.json.gz",
        "unseen_instructions-test": "data/datasets/object_rearrangement/unseen_instructions/test/test.json.gz",
        "unseen_inits-test": "data/datasets/object_rearrangement/v4/test/test.json.gz"
    }
    split_episode_ids = {}
    all_episode_ids, dataset = get_episode_ids("data/episodes/pick_and_place_v1/cleaned_dataset/clean_hits_12k.json.gz")
    print("Total episodes loaded: {}".format(len(all_episode_ids)))
    for split, path in paths.items():
        split_episode_ids[split], _ = get_episode_ids(path)
        print("Split: {}, Num episodes loaded: {}".format(split, len(split_episode_ids[split])))

    for episode_id in all_episode_ids:
        index_file.append({
            "episode_id": episode_id,
            "splits": {
                "unseen_scenes": is_in_train(split_episode_ids["unseen_scenes-train"], [], split_episode_ids["unseen_scenes-test"], episode_id),
                "unseen_instructions": is_in_train(split_episode_ids["unseen_instructions-train"], split_episode_ids["unseen_instructions-val"], split_episode_ids["unseen_instructions-test"], episode_id),
                "unseen_initializations": is_in_train(split_episode_ids["unseen_inits-train"], split_episode_ids["unseen_inits-val"], split_episode_ids["unseen_inits-test"], episode_id),
            }
        })

    write_json(index_file, index_file_path)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-file-path", type=str, default="data/datasets/pick_place/index.json"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/episodes/pick_and_place_v1/cleaned_dataset/clean_hits_12k.json.gz"
    )
    parser.add_argument(
        "--split-id", type=str, default="unseen_scenes"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/datasets/pick_place/"
    )
    parser.add_argument(
        "--generate-splits", dest="generate_splits", action="store_true"
    )
    args = parser.parse_args()

    if args.generate_splits:
        generate_splits(args.dataset, args.index_file_path, args.split_id, args.output_dir)
    else:
        create_dataset_index_file(args.index_file_path)


if __name__ == "__main__":
    main()
