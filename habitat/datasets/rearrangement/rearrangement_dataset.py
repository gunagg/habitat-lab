#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import sys
from typing import List, Optional

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.utils import VocabFromText
from habitat.tasks.rearrangement.rearrangement import InstructionData, RearrangementEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="RearrangementDataset-v0")
class RearrangementDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[RearrangementEpisode]
    instruction_vocab: VocabFromText

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabFromText(
           sentences=deserialized["instruction_vocab"]["sentences"]
        )

        for episode in deserialized["episodes"]:
            instruction_text = episode["instruction"]["instruction_text"]
            instruction_tokens = self.instruction_vocab.tokenize_and_index(instruction_text)
            episode["instruction"]["instruction_tokens"] = instruction_tokens
            episode = RearrangementEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            self.episodes.append(episode)
