import argparse
import gzip
import glob
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

def save_splits(dataset, split_episode_ids):
    pass


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
    all_episode_ids, dataset = get_episode_ids("data/episodes/pick_and_place_v1/clean_all_hits_deduped.json.gz")
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
    save_splits(dataset, split_episode_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-file-path", type=str, default="data/datasets/pick_place/index.json"
    )
    args = parser.parse_args()

    create_dataset_index_file(args.index_file_path)


if __name__ == "__main__":
    main()
