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


def collate_fn(batch):
    """Each sample in batch: (
            obs,
            prev_actions,
            oracle_actions,
            inflec_weight,
        )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        #print("pad amt: {}".format(pad_amount))
        #print(t.shape)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        #print(pad.shape)
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))
    
    observations_batch = list(transposed[1])
    next_actions_batch = list(transposed[2])
    prev_actions_batch = list(transposed[3])
    episode_not_dones_batch = list(transposed[4])
    weights_batch = list(transposed[5])
    B = len(prev_actions_batch)
    # print("nation {}".format(next_actions_batch[0].shape))
    # print("pation {}".format(prev_actions_batch[0].shape))

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    #print("max traj: {}".format(max_traj_len))
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )
        #print("starting next actions: ", next_actions_batch[0].shape)
        next_actions_batch[bid] = _pad_helper(next_actions_batch[bid], max_traj_len)
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        episode_not_dones_batch[bid] = _pad_helper(episode_not_dones_batch[bid], max_traj_len)
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )
    
    next_actions_batch = torch.stack(next_actions_batch, dim=1)
    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    episode_not_dones_batch = torch.stack(episode_not_dones_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(prev_actions_batch, dtype=torch.float)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)
    # print("done maks")
    # print(episode_not_dones_batch)

    # print(not_done_masks)
    # print("done maks")

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        episode_not_dones_batch.view(-1, 1),
        next_actions_batch,
        weights_batch,
    )



class RearrangementEpisodeDataset(Dataset):
    """Pytorch dataset for object rearrangement task for each episode"""

    def __init__(self, config, mode="train"):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        self.config = config.TASK_CONFIG
        self.dataset_path = config.DATASET_PATH.format(split=mode)
        
        self.env = habitat.Env(config=self.config)
        self.episodes = self.env._dataset.episodes
        self.instruction_vocab = self.env._dataset.instruction_vocab

        self.resolution = self.env._sim.get_resolution()

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """


            logger.info(
                "Dataset cache not found. Saving rgb, seg, depth scene images"
            )
            logger.info(
                "Number of {} episodes: {}".format(mode, len(self.episodes))
            )

            self.scene_ids = []
            self.scene_episode_dict = {}

            # dict for storing list of episodes for each scene
            for episode in self.episodes:
                if episode.scene_id not in self.scene_ids:
                    self.scene_ids.append(episode.scene_id)
                    self.scene_episode_dict[episode.scene_id] = [episode]
                else:
                    self.scene_episode_dict[episode.scene_id].append(episode)

            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(1e11),
                writemap=True,
            )

            self.count = 0
            print("HabitatSimActions.TURN_RIGHT", HabitatSimActions.TURN_RIGHT)
            print("HabitatSimActions.TURN_LEFT", HabitatSimActions.TURN_LEFT)
            print("HabitatSimActions.MOVE_FORWARD", HabitatSimActions.MOVE_FORWARD)
            print("HabitatSimActions.MOVE_BACKWARD", HabitatSimActions.MOVE_BACKWARD)
            print("HabitatSimActions.LOOK_UP", HabitatSimActions.LOOK_UP)
            print("HabitatSimActions.LOOK_DOWN", HabitatSimActions.LOOK_DOWN)
            print("HabitatSimActions.NO_OP", HabitatSimActions.NO_OP)
            print("HabitatSimActions.GRAB_RELEASE", HabitatSimActions.GRAB_RELEASE)
            print("HabitatSimActions.START", HabitatSimActions.START)
            print("HabitatSimActions.STOP", HabitatSimActions.STOP)

            for scene in tqdm(list(self.scene_episode_dict.keys())):
                for episode in tqdm(self.scene_episode_dict[scene]):
                    self.load_scene(scene, episode)
                    state_index_queue = [-1]
                    try:
                        # TODO: Consider alternative for shortest_paths
                        state_index_queue.extend(range(0, len(episode.reference_replay)))
                    except AttributeError as e:
                        logger.error(e)
                    self.save_frames(state_index_queue, episode)

            logger.info("Rearrangement database ready!")

        else:
            print("HabitatSimActions.TURN_RIGHT", HabitatSimActions.TURN_RIGHT)
            print("HabitatSimActions.TURN_LEFT", HabitatSimActions.TURN_LEFT)
            print("HabitatSimActions.MOVE_FORWARD", HabitatSimActions.MOVE_FORWARD)
            print("HabitatSimActions.MOVE_BACKWARD", HabitatSimActions.MOVE_BACKWARD)
            print("HabitatSimActions.LOOK_UP", HabitatSimActions.LOOK_UP)
            print("HabitatSimActions.LOOK_DOWN", HabitatSimActions.LOOK_DOWN)
            print("HabitatSimActions.NO_OP", HabitatSimActions.NO_OP)
            print("HabitatSimActions.GRAB_RELEASE", HabitatSimActions.GRAB_RELEASE)
            print("HabitatSimActions.START", HabitatSimActions.START)
            print("HabitatSimActions.STOP", HabitatSimActions.STOP)
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )
        
        self.env.close()

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / 5)
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
        not_dones = []
        observations = {
            "rgb": [],
            "depth": [],
            "instruction": [],
        }
        obs_list = []
        reference_replay = episode.reference_replay
        instruction = episode.instruction
        for state_index in state_index_queue:
            instruction_tokens = np.array(instruction.instruction_tokens)
            observation = self.env.sim.get_observations_at(
                episode.start_position,
                episode.start_rotation
            )

            if state_index != -1:
                state = reference_replay[state_index]
                position = state["agent_state"]["position"]
                rotation = state["agent_state"]["rotation"]
                object_states = state["object_states"]
                sensor_states = state["agent_state"]["sensor_data"]

                observation = self.env.sim.get_observations_at(
                    position, rotation, sensor_states, object_states
                )

            next_action = HabitatSimActions.STOP
            if state_index < len(reference_replay) - 1:
                next_state = reference_replay[state_index + 1]
                next_action = get_habitat_sim_action(next_state["action"])

            prev_action = HabitatSimActions.START
            if state_index != -1:
                prev_state = reference_replay[state_index]
                prev_action = get_habitat_sim_action(prev_state["action"])

            not_done = 1
            if state_index == len(reference_replay) -1:
                not_done = 0

            observations["depth"].append(observation["depth"])
            observations["rgb"].append(observation["rgb"])
            observations["instruction"].append(instruction_tokens)
            next_actions.append(next_action)
            prev_actions.append(prev_action)
            not_dones.append(not_done)

            # if state_index == len(reference_replay) - 150:
            #     break

            frame = observations_to_image(
                {"rgb": observation["rgb"]}, {}
            )
            obs_list.append(frame)
        
        oracle_actions = np.array(next_actions)
        inflection_weights = np.concatenate(([1], oracle_actions[1:] != oracle_actions[:-1]))

        sample_key = "{0:0=6d}".format(self.count)
        with self.lmdb_env.begin(write=True) as txn:
            txn.put((sample_key + "_obs").encode(), msgpack_numpy.packb(observations, use_bin_type=True))
            txn.put((sample_key + "_next_action").encode(), np.array(next_actions).tobytes())
            txn.put((sample_key + "_prev_action").encode(), np.array(prev_actions).tobytes())
            txn.put((sample_key + "_not_done").encode(), np.array(not_dones).tobytes())
            txn.put((sample_key + "_weights").encode(), inflection_weights.tobytes())
        
        self.count += 1
        # images_to_video(images=obs_list, output_dir="demos", video_name="dummy")

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

        obs_idx = "{0:0=6d}_obs".format(idx)
        observations_binary = self.lmdb_cursor.get(obs_idx.encode())
        observations = msgpack_numpy.unpackb(observations_binary, raw=False)
        for k, v in observations.items():
            obs = np.array(observations[k])
            observations[k] = torch.from_numpy(obs)

        next_action_idx = "{0:0=6d}_next_action".format(idx)
        next_action_binary = self.lmdb_cursor.get(next_action_idx.encode())
        next_action = np.frombuffer(next_action_binary, dtype="int")
        next_action = torch.from_numpy(np.copy(next_action))

        prev_action_idx = "{0:0=6d}_prev_action".format(idx)
        prev_action_binary = self.lmdb_cursor.get(prev_action_idx.encode())
        prev_action = np.frombuffer(prev_action_binary, dtype="int")
        prev_action = torch.from_numpy(np.copy(prev_action))

        not_done_idx = "{0:0=6d}_not_done".format(idx)
        not_done_binary = self.lmdb_cursor.get(not_done_idx.encode())
        not_done = np.frombuffer(not_done_binary, dtype="int")
        not_done = torch.from_numpy(np.copy(not_done))

        weight_idx = "{0:0=6d}_weights".format(idx)
        weight_binary = self.lmdb_cursor.get(weight_idx.encode())
        weight = np.frombuffer(weight_binary, dtype="int")
        weight = torch.from_numpy(np.copy(weight))

        return idx, observations, next_action, prev_action, not_done, weight
