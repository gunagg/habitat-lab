import argparse
import gzip
import glob
import json

from tqdm import tqdm
from habitat.datasets.utils import VocabFromText
from psiturk_dataset.utils.utils import load_dataset, load_json_dataset, write_json, write_gzip, load_vocab


VISITED_POINT_DICT = {}
assignment_dict = {}
episode_ids = ['A1L3937MY09J3I:3Z7EFSHGNBH1CG7U84ECI5ABGYYCX5','A1ZE52NWZPN85P:3C6FJU71TSWMYFE4ZRLEVP3QQU9YUY','A2CWA5VQZ6IWMQ:3YGXWBAF72KAEEJKOTC7LUDDNIP4C8','APGX2WZ59OWDN:358010RM5GWXBPDUZL9H8XY015IVXR']


def caclulate_inflections(episode):
    inflections = 1
    reference_replay = episode["reference_replay"]
    for i in range(len(reference_replay) - 1):
        if reference_replay[i]["action"] != reference_replay[i - 1]["action"]:
            inflections += 1
    return inflections, len(reference_replay)


def calculate_inflection_weight(path):
    data = load_dataset(path)

    episodes = data["episodes"]
    inflections = 0
    total_actions = 0
    for episode in tqdm(episodes):
        num_inflections, num_actions = caclulate_inflections(episode)
        inflections += num_inflections
        total_actions += num_actions

    print("Total episodes: {}".format(len(episodes)))
    print("Inflection weight: {}".format(total_actions / inflections))

    instructions = convert_instruction_tokens(episodes)
    print("Num of distinct instructions: {}".format(len(set(instructions))))
    write_json(list(set(instructions)), "data/hit_approvals/instructions.json")


def calculate_inflection_weight_objectnav(path):
    files = glob.glob(path + "*.json.gz")
    inflections = 0
    total_actions = 0
    total_episodes = 0

    for file_path in tqdm(files):
        data = load_dataset(file_path)

        episodes = data["episodes"]
        for episode in episodes:
            num_inflections, num_actions = caclulate_inflections(episode)
            inflections += num_inflections
            total_actions += num_actions
            total_episodes += 1

    print("Total episodes: {}".format(total_episodes))
    print("Inflection weight: {}".format(total_actions / inflections))


def convert_instruction_tokens(episodes):
    vocab = load_vocab()
    instruction_vocab = VocabFromText(
        sentences=vocab["sentences"]
    )

    instructions = []
    for episode in episodes:
        instruction = instruction_vocab.token_idx_2_string(episode["instruction"]["instruction_tokens"])
        instructions.append(instruction)
    return instructions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/hit_approvals/dataset/backup/train.json.gz"
    )
    parser.add_argument(
        "--task", type=str, default="rearrangement"
    )
    args = parser.parse_args()

    if args.task == "rearrangement":
        calculate_inflection_weight(args.path)
    else:
        calculate_inflection_weight_objectnav(args.path)


if __name__ == "__main__":
    main()
