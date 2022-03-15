#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from collections import deque, defaultdict
from typing import Optional, Type

import numbers
import numpy as np

import habitat
from habitat import Config, Dataset, logger
from habitat_baselines.common.baseline_registry import baseline_registry


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="RearrangeRLEnv")
class RearrangeRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self._rl_config.SLACK_REWARD

        reward += current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if (
            self._rl_config.get("END_ON_SUCCESS", True)
            and self._episode_success()
        ):
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="StackedNavRLEnv")
class StackedNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        self._observation_queue = deque()
        self._stack_n_frames = 5
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._observation_queue = [observations]
        observations = self.stack_observations()
        logger.info("\n\n stack end\n\n")
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations
    
    def stack_observations(self):
        logger.info("\n\n len obs queue: {} \n\n".format(len(self._observation_queue)))
        stacked_obs = defaultdict(list)

        obs = self._observation_queue[0]
        sensor_names = sorted(
            obs.keys(),
            key=lambda name: 1
            if isinstance(obs[name], numbers.Number)
            else np.prod(obs[name].shape),
            reverse=True,
        )
        for sensor_name in sensor_names:
            for i, obs in enumerate(self._observation_queue):
                sensor = obs[sensor_name]
                stacked_obs[sensor_name].append(sensor)
        logger.info("stacked transform done")
        logger.info("stacking started")
        observations = {}
        for sensor_name in sensor_names:
            obs = stacked_obs[sensor_name]
            logger.info("sensor: {}, len:{}, stack: {}".format(sensor_name, len(obs), self._stack_n_frames))
            if len(obs) != self._stack_n_frames:
                n_pad = self._stack_n_frames - len(obs)
                pad_obs = np.zeros_like(obs[0])
                logger.info("pad_obs:{}".format(pad_obs.shape))
                padding = [pad_obs] * n_pad
                logger.info("pad len:{}, {}".format(len(padding), n_pad))
                obs = padding + obs
                logger.info("after padding:{}".format(len(obs)))
            logger.info("after padding:{}".format(len(obs)))
            if len(obs[0].shape) == 0:
                logger.info("stacking:{}".format(np.stack(obs, axis=-1).shape))
                observations[sensor_name] = np.stack(obs, axis=-1)
            else:
                logger.info("concat:{}".format(np.concatenate(obs, axis=-1).shape))
                observations[sensor_name] = np.concatenate(obs, axis=-1)
                logger.info("stacked {}: {}".format(sensor_name, observations[sensor_name].shape))
        logger.info("return")
        return observations

    def step(self, *args, **kwargs):
        logger.info("\n\n step \n\n")
        self._previous_action = kwargs["action"]
        step_data = super().step(*args, **kwargs)
        logger.info("\n\n step done \n\n")
        if len(self._observation_queue) == self._stack_n_frames:
            self._observation_queue.popleft()
        self._observation_queue.append(step_data[0])
        logger.info("\n\n stack start \n\n")
        stacked_obs = self.stack_observations()
        logger.info("\n\n stack end, {} \n\n".format(type(step_data)))
        step_data = list(step_data)
        step_data[0] = stacked_obs
        logger.info("stepping done")
        return tuple(step_data)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
