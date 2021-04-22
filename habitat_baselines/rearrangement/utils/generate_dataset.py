import argparse
import cv2
import habitat
import json
import sys
import time
import os

from habitat import Config
from habitat import get_config as get_task_config
from habitat_baselines.rearrangement.dataset.episode_dataset import RearrangementEpisodeDataset
from habitat_baselines.objectnav.dataset.episode_dataset import ObjectNavEpisodeDataset

from time import sleep

from PIL import Image


def generate_episode_dataset(config, mode, task):
    if task == "rearrangement":
        rearrangement_dataset = RearrangementEpisodeDataset(
            config,
            content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
            mode=mode,
            use_iw=config.IL.USE_IW,
            inflection_weight_coef=config.MODEL.inflection_weight_coef
        )
    else:
        rearrangement_dataset = ObjectNavEpisodeDataset(
            config,
            content_scenes=config.TASK_CONFIG.DATASET.CONTENT_SCENES,
            mode=mode,
            use_iw=config.IL.USE_IW,
            inflection_weight_coef=config.MODEL.inflection_weight_coef
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", type=str, default="empty_house"
    )
    parser.add_argument(
        "--episodes", type=str, default="data/datasets/object_rearrangement/v0/train/train.json.gz"
    )
    parser.add_argument(
        "--mode", type=str, default="train"
    )
    parser.add_argument(
        "--task", type=str, default="rearrangement"
    )
    args = parser.parse_args()

    config = habitat.get_config("habitat_baselines/config/object_rearrangement/il_object_rearrangement.yaml")
    if args.task == "objectnav":
        config = habitat.get_config("habitat_baselines/config/objectnav/il_objectnav.yaml")

    cfg = config
    cfg.defrost()
    task_config = get_task_config(cfg.BASE_TASK_CONFIG_PATH)
    task_config.defrost()
    task_config.DATASET.DATA_PATH = args.episodes
    task_config.DATASET.CONTENT_SCENES = [args.scene]
    task_config.freeze()
    cfg.TASK_CONFIG = task_config
    cfg.freeze()

    observations = generate_episode_dataset(cfg, args.mode, args.task)

if __name__ == "__main__":
    main()
