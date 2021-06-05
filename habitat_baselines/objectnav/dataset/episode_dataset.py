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
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image
from habitat_sim.utils import viz_utils as vut
from habitat_baselines.objectnav.models.rednet import load_rednet


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
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))
    
    observations_batch = list(transposed[1])
    next_actions_batch = list(transposed[2])
    prev_actions_batch = list(transposed[3])
    weights_batch = list(transposed[4])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )
        next_actions_batch[bid] = _pad_helper(next_actions_batch[bid], max_traj_len)
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
    
    next_actions_batch = torch.stack(next_actions_batch, dim=1)
    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(next_actions_batch, dtype=torch.float)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch,
        not_done_masks,
        next_actions_batch,
        weights_batch,
    )


class ObjectNavEpisodeDataset(Dataset):
    """Pytorch dataset for object rearrangement task for each episode"""

    def __init__(self, config, content_scenes=["*"], mode="train", use_iw=False, split_name="split_1", inflection_weight_coef=1.0):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            mode: 'train'/'val'
        """
        scene_split_name = split_name

        self.config = config.TASK_CONFIG
        self.dataset_path = config.DATASET_PATH.format(split=mode, scene_split=scene_split_name)
        logger.info("datasetpath: {}".format(self.dataset_path))

        self.config.defrost()
        self.config.DATASET.CONTENT_SCENES = content_scenes
        self.config.freeze()

        self.resolution = [self.config.SIMULATOR.RGB_SENSOR.WIDTH, self.config.SIMULATOR.RGB_SENSOR.HEIGHT]
        self.possible_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        self.count = 0
        self.success_threshold = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        self.total_success = 0
        self.inflection_weight_coef = inflection_weight_coef

        if use_iw:
            self.inflec_weight = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weight = torch.tensor([1.0, 1.0])

        if not self.cache_exists():
            """
            for each scene > load scene in memory > save frames for each
            episode corresponding to that scene
            """
            self.device = (
                torch.device("cuda", 0)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt="data/rednet-models/rednet_semmap_mp3d_tuned.pth",
                resize=True # since we train on half-vision
            )

            self.env = habitat.Env(config=self.config)
            self.episodes = self.env._dataset.episodes

            logger.info(
                "Dataset cache not found/ignored. Saving goal state observations"
            )
            logger.info(
                "Number of {} episodes: {}".format(mode, len(self.episodes))
            )
            logger.info("datasetpath: {}".format(self.dataset_path))

            self.lmdb_env = lmdb.open(
                self.dataset_path,
                map_size=int(2e12),
                writemap=True,
            )

            for episode in tqdm(self.episodes):
                observations = self.env.reset()
                episode = self.env.current_episode
                state_index_queue = []
                try:
                    # Ignore last frame as it is only used to lookup for STOP action
                    state_index_queue.extend(range(1, len(episode.reference_replay) - 1))
                except AttributeError as e:
                    logger.error(e)
                self.save_frames(state_index_queue, episode, observations)
            
            logger.info("Total success: {}".format(self.total_success / len(self.episodes)))
            logger.info("Objectnav dataset ready!")
            self.env.close()
        else:
            logger.info("Dataset cache found.")
            self.lmdb_env = lmdb.open(
                self.dataset_path,
                readonly=True,
                lock=False,
            )

        self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / 4)
        self.lmdb_env.close()
        self.lmdb_env = None
    
    def get_pred_semantic_obs(self, observations):
        obs_rgb = torch.Tensor(observations["rgb"]).unsqueeze(0).to(self.device)
        obs_depth = torch.Tensor(observations["depth"]).unsqueeze(0).to(self.device)
        semantic_obs = self.semantic_predictor(obs_rgb, obs_depth)
        return semantic_obs.squeeze(0).cpu().numpy()

    def save_frames(
        self, state_index_queue: List[int], episode: RearrangementEpisode, observation: Dict
    ) -> None:
        r"""
        Writes rgb, seg, depth frames to LMDB.
        """
        next_actions = []
        prev_actions = []
        obs_list = []
        observations = defaultdict(list)

        observations["depth"].append(observation["depth"])
        observations["rgb"].append(observation["rgb"])
        observations["gps"].append(observation["gps"])
        observations["compass"].append(observation["compass"])
        observations["objectgoal"].append(observation["objectgoal"])
        observations["semantic"].append(observation["semantic"])

        next_action = self.possible_actions.index(episode.reference_replay[1].action)
        next_actions.append(next_action)
        prev_action = self.possible_actions.index(episode.reference_replay[0].action)
        prev_actions.append(prev_action)
        
        reference_replay = episode.reference_replay
        success = 0
        info = {}
        if len(reference_replay) > 1000:
            return
        for state_index in state_index_queue:
            state = reference_replay[state_index]

            action = self.possible_actions.index(state.action)
            if state_index > 0:
                observation = self.env.step(action=action)
                info = self.env.get_metrics()
                success = info["distance_to_goal"] < self.success_threshold

            next_state = reference_replay[state_index + 1]
            next_action = self.possible_actions.index(next_state.action)

            prev_state = reference_replay[state_index]
            prev_action = self.possible_actions.index(prev_state.action)

            observations["depth"].append(observation["depth"])
            observations["rgb"].append(observation["rgb"])
            observations["semantic"].append(observation["semantic"])
            observations["gps"].append(observation["gps"])
            observations["compass"].append(observation["compass"])
            observations["objectgoal"].append(observation["objectgoal"])
            next_actions.append(next_action)
            prev_actions.append(prev_action)

        oracle_actions = np.array(next_actions)
        inflection_weights = np.concatenate(([1], oracle_actions[1:] != oracle_actions[:-1]))
        inflection_weights = self.inflec_weight[torch.from_numpy(inflection_weights)].numpy()

        if not success:
            return

        sample_key = "{0:0=6d}".format(self.count)
        logger.info("Episode: {}, Success: {}, Dist: {}, Len: {}".format(self.count, success, info["distance_to_goal"], len(next_actions)))
        self.total_success += success
        with self.lmdb_env.begin(write=True) as txn:
            txn.put((sample_key + "_obs").encode(), msgpack_numpy.packb(observations, use_bin_type=True))
            txn.put((sample_key + "_next_action").encode(), np.array(next_actions).tobytes())
            txn.put((sample_key + "_prev_action").encode(), np.array(prev_actions).tobytes())
            txn.put((sample_key + "_weights").encode(), inflection_weights.tobytes())
        
        self.count += 1
        # images_to_video(images=obs_list, output_dir="demos", video_name="dummy_{}".format(self.count))

    def cache_exists(self) -> bool:
        if os.path.exists(self.dataset_path):
            if os.listdir(self.dataset_path):
                return True
        else:
            os.makedirs(self.dataset_path)
        return False

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
                map_size=int(2e12),
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

        # sem_obs_idx = "{0:0=6d}_sobs".format(idx)
        # sem_observations_binary = self.lmdb_cursor.get(sem_obs_idx.encode())
        # sem_observations = msgpack_numpy.unpackb(sem_observations_binary, raw=False)
        # for k, v in sem_observations.items():
        #     obs = np.array(sem_observations[k])
        #     observations[k] = torch.from_numpy(obs)
        
        # pred_sem_obs_idx = "{0:0=6d}_pobs".format(idx)
        # pred_sem_observations_binary = self.lmdb_cursor.get(pred_sem_obs_idx.encode())
        # pred_sem_observations = msgpack_numpy.unpackb(pred_sem_observations_binary, raw=False)
        # for k, v in pred_sem_observations.items():
        #     obs = np.array(pred_sem_observations[k])
        #     observations[k] = torch.from_numpy(obs)

        next_action_idx = "{0:0=6d}_next_action".format(idx)
        next_action_binary = self.lmdb_cursor.get(next_action_idx.encode())
        next_action = np.frombuffer(next_action_binary, dtype="int")
        next_action = torch.from_numpy(np.copy(next_action))

        prev_action_idx = "{0:0=6d}_prev_action".format(idx)
        prev_action_binary = self.lmdb_cursor.get(prev_action_idx.encode())
        prev_action = np.frombuffer(prev_action_binary, dtype="int")
        prev_action = torch.from_numpy(np.copy(prev_action))

        weight_idx = "{0:0=6d}_weights".format(idx)
        weight_binary = self.lmdb_cursor.get(weight_idx.encode())
        weight = np.frombuffer(weight_binary, dtype="float32")
        weight = torch.from_numpy(np.copy(weight))
        weight = torch.where(weight != 1.0, self.inflection_weight_coef, 1.0)

        return idx, observations, next_action, prev_action, weight
