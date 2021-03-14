import cv2
import msgpack_numpy
import os
import random
import sys
import torch
from collections import defaultdict
from typing import List, Dict

import lmdb
import magnum as mn
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import habitat
from habitat import logger
from habitat.datasets.utils import VocabDict
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrangement.rearrangement import InstructionData, RearrangementEpisode
from habitat.tasks.utils import get_habitat_sim_action
from habitat.utils.visualizations.utils import observations_to_image, images_to_video
from habitat_sim.utils import viz_utils as vut


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


class RearrangementGoalDataset(Dataset):
    """Pytorch dataset for object rearrangement task for each episode"""

    def __init__(self, config, mode="train_goals"):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        self.config = config.TASK_CONFIG
        self.dataset_path = config.GOAL_DATASET_PATH.format(split=mode)
        
        self.env = habitat.Env(config=self.config)
        self.episodes = self.env._dataset.episodes
        self.instruction_vocab = self.env._dataset.instruction_vocab

        self.resolution = self.env._sim.get_resolution()

        self.scene_ids = []
        self.scene_episode_dict = {}

        # dict for storing list of episodes for each scene
        for episode in self.episodes:
            if episode.scene_id not in self.scene_ids:
                self.scene_ids.append(episode.scene_id)
                self.scene_episode_dict[episode.scene_id] = [episode]
            else:
                self.scene_episode_dict[episode.scene_id].append(episode)

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """

            logger.info(
                "Dataset cache not found. Saving goal state observations"
            )
            logger.info(
                "Number of {} episodes: {}".format(mode, len(self.episodes))
            )

            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )

            self.count = 0
            for scene in tqdm(list(self.scene_episode_dict.keys())):
                self.count = 0
                for episode in tqdm(self.scene_episode_dict[scene]):
                    self.load_scene(scene, episode)
                    state_index_queue = []
                    try:
                        # Ignore last frame as it is only used to lookup for STOP action
                        state_index_queue.extend(range(0, len(episode.reference_replay) - 1))
                    except AttributeError as e:
                        logger.error(e)
                    self.save_frames(state_index_queue, episode)
            logger.info("Rearrangement database ready!")

        else:
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )

        self.env.close()

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / 4)
        self.lmdb_env.close()
        self.lmdb_env = None

    def save_frames(
        self, state_index_queue: List[int], episode: RearrangementEpisode
    ) -> None:
        r"""
        Writes rgb, seg, depth frames to LMDB.
        """
        next_actions = []
        prev_actions = []
        observations = {
            "rgb": [],
            "depth": [],
            "instruction": [],
            "demonstration": [],
            "gripped_object_id": [],
        }
        obs_list = []
        reference_replay = episode.reference_replay
        instruction = episode.instruction

        state_index = state_index_queue[-1]
        scene_id = episode.scene_id
    
        instruction_tokens = np.array(instruction.instruction_tokens)

        state = reference_replay[state_index]
        position = state.agent_state.position
        rotation = state.agent_state.rotation
        object_states = state.object_states
        sensor_states = state.agent_state.sensor_data

        observation = self.env.sim.get_observations_at(
            position, rotation, sensor_states, object_states
        )

        observations["depth"].append(observation["depth"])
        observations["rgb"].append(observation["rgb"])
        observations["instruction"].append(instruction_tokens)
        observations["demonstration"].append(0)
        observations["gripped_object_id"].append(-1)

        frame = observations_to_image(
            {"rgb": observation["rgb"]}, {}
        )
        obs_list.append(frame)
        
        oracle_actions = np.array(next_actions)

        scene_id = scene_id.split("/")[-1].replace(".", "_")
        sample_key = "{0}_{1:0=6d}".format(scene_id, self.count)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put((sample_key + "_obs").encode(), msgpack_numpy.packb(observations, use_bin_type=True))
        
        self.count += 1
        images_to_video(images=obs_list, output_dir="demos", video_name="dummy_{}".format(self.count))

    def cache_exists(self) -> bool:
        if os.path.exists(self.dataset_path):
            if os.listdir(self.dataset_path):
                return True
        else:
            os.makedirs(self.dataset_path)
        return False
    
    def get_vocab_dict(self) -> VocabDict:
        r"""Returns Instruction VocabDicts"""
        return self.instruction_vocab

    def load_scene(self, scene, episode) -> None:
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.SIMULATOR.objects = episode.objects
        self.config.freeze()
        self.env.sim.reconfigure(self.config.SIMULATOR)
    
    def get_scene_episode_length(self, scene: str) -> int:
        return len(self.scene_episode_dict[scene])

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])
    
        obs_idx = "{0}{1:0=6d}_obs".format(scene_id, idx)
        observations_binary = self.lmdb_cursor.get(obs_idx.encode())
        observations = msgpack_numpy.unpackb(observations_binary, raw=False)
        for k, v in observations.items():
            obs = np.array(observations[k])
            observations[k] = torch.from_numpy(obs)

        return observations
    
    def get_item(self, idx: int, scene_id: str):
        r"""Returns batches to trainer.

        batch: (rgb, depth, seg)

        """
        org_scene_id = scene_id
        scene_id = scene_id.split("/")[-1].replace(".", "_")
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )
            self.lmdb_txn = self.lmdb_env.begin()
            self.lmdb_cursor = self.lmdb_txn.cursor()
        
        height, width = int(self.resolution[0]), int(self.resolution[1])

        obs_idx = "{0}_{1:0=6d}_obs".format(scene_id, 0)
        observations_binary = self.lmdb_cursor.get(obs_idx.encode())
        observations = msgpack_numpy.unpackb(observations_binary, raw=False)
        for k, v in observations.items():
            obs = np.array(observations[k])
            observations[k] = torch.from_numpy(obs)

        return observations
