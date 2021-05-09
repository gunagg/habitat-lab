import argparse
import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import gzip

from collections import defaultdict
from psiturk_dataset.utils.utils import write_json, write_gzip, load_dataset, load_vocab
from tqdm import tqdm


def merge_episodes(path, output_path):
    files = glob.glob(path + "*.json.gz")
    vocab = load_vocab()
    instructions = vocab["sentences"]

    dataset = {
        "episodes": [],
        "instruction_vocab": {
            "sentences": instructions
        }
    }

    for file_path in files:
        if "failed" in file_path:
            continue
        print("Loading episodes: {}".format(file_path))
        data = load_dataset(file_path)
        dataset["episodes"].extend(data["episodes"])

    print("Total episodes: {}".format(len(dataset["episodes"])))
    
    write_json(dataset, output_path)
    write_gzip(output_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, default="data/hit_approvals/hits_max_length_1500.json"
    )
    parser.add_argument(
        "--output-path", type=str, default="data/episodes/sample_hits.json"
    )
    
    args = parser.parse_args()
    merge_episodes(args.input_path, args.output_path)
    
