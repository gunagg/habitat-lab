import gzip
import json
import random
from collections import defaultdict, deque
from logging import info

import numpy as np
import torch
import tqdm
from gym import Space
from habitat import make_dataset, logger
from habitat.config.default import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from habitat_baselines.utils.env_utils import construct_envs, construct_ddp_envs


class TeacherRecollectionDataset(torch.utils.data.IterableDataset):
    def __init__(self, config: Config, is_ddp=False):
        super().__init__()
        self.config = config
        self._preload = deque()

        assert (
            config.IL.BehaviorCloning.preload_size >= config.IL.BehaviorCloning.batch_size
        ), "preload size must be greater than batch size."
        self.envs = None
        self._env_observations = None

        if config.IL.USE_IW:
            self.inflec_weights = torch.tensor(
                [1.0, config.IL.inflection_weight_coef]
            )
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])
        if self.config.IL.BehaviorCloning.preload_trajectories_file:
            with gzip.open(
                config.IL.BehaviorCloning.trajectories_file, "rt"
            ) as f:
                self.trajectories = json.load(f)
        else:
            self.trajectories = self.collect_dataset()
        self.is_ddp = is_ddp

        self.initialize_sims()

    def initialize_sims(self):
        config = self.config.clone()
        config.defrost()
        config.TASK_CONFIG.MEASUREMENTS = []
        config.freeze()

        if self.is_ddp:
            self.envs = construct_ddp_envs(
                config,
                get_env_class(config.ENV_NAME),
                world_rank=self.config.world_rank,
                world_size=self.config.world_size,
            )
        else:
            self.envs = construct_envs(
                config,
                get_env_class(config.ENV_NAME),
            )
        self.length = sum(self.envs.number_of_episodes)
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], self.obs_transforms
        )

        self.env_step = [0 for _ in range(self.envs.num_envs)]
        self._env_observations = [[] for _ in range(self.envs.num_envs)]

        observations = self.envs.reset()
        for i, ep in enumerate(self.envs.current_episodes()):
            path_step = self.trajectories[str(ep.episode_id)][0]
            self._env_observations[i].append(
                (
                    observations[i],
                    path_step[0],  # prev_action
                    path_step[2],  # oracle_action
                )
            )

    @property
    def batch_size(self):
        return self.config.IL.BehaviorCloning.batch_size

    @property
    def observation_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        assert self._observation_space is not None
        return self._observation_space

    @property
    def action_space(self) -> Space:
        assert self.envs is not None, "Simulator must first be loaded."
        return self.envs.action_spaces[0]

    def close_sims(self):
        self.envs.close()
        del self.envs
        del self._env_observations
        self.envs = None
        self._env_observations = None

    def collect_dataset(self):
        r"""Uses the ground truth trajectories to create a teacher forcing
        datset for a given split. Loads both guide and follower episodes.
        """
        trajectories = defaultdict(list)
        gt_dataset = make_dataset(
                    self.config.TASK_CONFIG.DATASET.TYPE, config=self.config.TASK_CONFIG.DATASET
                )
        t = (
            tqdm.tqdm(gt_dataset.episodes, "GT Collection")
            if self.config.use_pbar
            else gt_dataset.episodes
        )
        self.total_episodes = 0
        possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        for episode in t:
            episode_id = episode.episode_id
            # Ignore first step
            trajectory = episode.reference_replay[1:]
            if (
                self.config.IL.BehaviorCloning.max_traj_len != -1
                and len(trajectory)
                > self.config.IL.BehaviorCloning.max_traj_len
            ):
                continue

            for i, step in enumerate(trajectory):
                action = possible_actions.index(step.action)
                prev_action = (
                    trajectories[episode_id][i - 1][1]
                    if i
                    else HabitatSimActions.STOP
                )

                # [prev_action, action, oracle_action]
                trajectories[episode_id].append([prev_action, action, action])
            self.total_episodes += 1

        with gzip.open(
            self.config.IL.BehaviorCloning.trajectories_file, "wt"
        ) as f:
            f.write(json.dumps(trajectories))
        return trajectories

    def _load_next(self):
        """
        Episode length is currently not considered. We were previously batching episodes
        together with similar lengths. Not sure if we need to bring that back.
        """

        if len(self._preload):
            # idx = random.randint(0, len(self._preload) - 1)
            return self._preload.popleft()

        while (
            len(self._preload) < self.config.IL.BehaviorCloning.preload_size
        ):
            current_episodes = self.envs.current_episodes()
            prev_eps = current_episodes

            # get the next action for each env
            actions = [
                self.trajectories[str(ep.episode_id)][self.env_step[i]][1]
                for i, ep in enumerate(current_episodes)
            ]

            outputs = self.envs.step(actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            current_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                self.env_step[i] += 1
                if dones[i]:
                    assert len(self._env_observations[i]) == len(
                        self.trajectories[str(prev_eps[i].episode_id)]
                    ), "Collected episode does not match the step count of trajectory"
                    self._preload.append(
                        (
                            [o[0] for o in self._env_observations[i]],
                            [o[1] for o in self._env_observations[i]],
                            [o[2] for o in self._env_observations[i]],
                        )
                    )
                    self._env_observations[i] = []
                    self.env_step[i] = 0

                path_step = self.trajectories[
                    str(current_episodes[i].episode_id)
                ][self.env_step[i]]
                self._env_observations[i].append(
                    (
                        observations[i],
                        path_step[0],  # prev_action
                        path_step[2],  # oracle_action
                    )
                )
                assert (
                    len(self._env_observations[i])
                    <= self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
                ), "Trajectories should be no more than the maximum episode steps."
        
        # logger.info("collect {}".format(len(self._preload)))

        return self._preload.popleft()
        #return self._preload[-1] #.popleft()

    def __next__(self):
        """Takes about 1s to once self._load_next() has finished with a batch
        size of 5. For this reason, we probably don't need to use extra workers.
        """
        x = self._load_next()
        obs, prev_actions, oracle_actions = x

        # transpose obs
        obs_t = defaultdict(list)
        for k in obs[0]:
            for i in range(len(obs)):
                obs_t[k].append(obs[i][k])

            obs_t[k] = np.array(obs_t[k])

        for k, v in obs_t.items():
            obs_t[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            0,
            obs_t,
            oracle_actions,
            prev_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            assert (
                worker_info.num_workers == 1
            ), "multiple workers not supported."

        return self