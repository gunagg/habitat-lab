#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import sys
import random
import torch

from habitat import logger
from habitat_baselines.common.tensor_dict import TensorDict


class StackRolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
        n_stack_frames: int = 1,
        n_samples: int = 8,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()
        self.n_stack_frames = n_stack_frames
        self.n_samples = n_samples

        for sensor in observation_space.spaces:
            dtype = observation_space.spaces[sensor].dtype
            if dtype == np.uint32:
                dtype = np.int
            elif dtype == np.float64:
                dtype = np.float32
            
            obs_shape = list(observation_space.spaces[sensor].shape)
            logger.info("ob shape: {}, {}".format(sensor, obs_shape))
            if len(obs_shape) > 0:
                obs_shape[-1] = obs_shape[-1] * n_stack_frames
                obs_shape = tuple(obs_shape)
            else:
                obs_shape = (n_stack_frames,)

            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *obs_shape,
                    ),
                    dtype=dtype
                )
            )

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,) * n_stack_frames
            else:
                action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *(action_shape)
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, n_stack_frames, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        actions=None,
        next_masks=None,
        rewards=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self, rnn_hidden_states = None):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]
        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    def get_next_actions(self):
        next_action_observations = self.buffers["observations"]["demonstration"][self.current_rollout_step_idx]
        actions = next_action_observations.clone()
        return actions

    def recurrent_generator(self, num_mini_batch) -> TensorDict:
        num_environments = self.buffers["masks"].size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        # TODO: Enable batch shuffling for IL generator
        # for inds in torch.randperm(num_environments).chunk(num_mini_batch):
        for inds in torch.arange(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            yield batch.map(lambda v: v.flatten(0, 1))
    

    def stacked_generator(self, num_mini_batch) -> TensorDict:
        num_environments = self.buffers["actions"].size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        # TODO: Enable batch shuffling for IL generator
        # for inds in torch.randperm(num_environments).chunk(num_mini_batch):
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            sampled_idxs = torch.randperm(self.current_rollout_step_idx)[:self.n_samples]
            print(type(self.buffers))
            batch = self.buffers[sampled_idxs][:, inds]

            batch["observations"]["rgb"] = []
            for i in range(sampled_idxs.shape[0]):
                idx = sampled_idxs[i].item()
                lb = max(idx - self.n_stack_frames, 0)

                zero_indices = (self.buffers["masks"][lb:idx][:, inds] == 0).nonzero(as_tuple=True)
                masks = torch.zeros((self.n_stack_frames, len(inds)))

                for k in range(zero_indices[0].shape[0]):
                    r_idx = zero_indices[0][k]
                    c_idx = zero_indices[1][k] 
                    masks[r_idx, c_idx:] = 1
                
                rgb_obs = self.buffers["observations"]["rgb"][lb:idx, inds].permute(0, 1, 4, 2, 3)

                if rgb_obs.shape[0] != self.n_stack_frames:
                    pad_size = (self.n_stack_frames - rgb_obs.shape[0], ) + tuple(rgb_obs.shape[1:])
                    pad = torch.zeros(pad_size)
                    rgb_obs = torch.cat([pad, rgb_obs], dim=0)
                else:
                    # zero pad the observations from previous episode
                    rgb_obs = rgb_obs * masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                rgb_obs = torch.hstack([rgb_obs[i] for i in range(rgb_obs.shape[0])]).unsqueeze(0)

                batch["observations"]["rgb"].append(rgb_obs.permute(0, 1, 3, 4, 2))
            batch["observations"]["rgb"] = torch.stack(batch["observations"]["rgb"], dim=0).squeeze(1)
            print("obs shape: {}".format(batch["observations"]["rgb"].shape))
            print("mask shape: {}".format(batch["masks"].shape))
            print("goal: {}".format(batch["observations"]["objectgoal"].shape))

            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            yield batch.map(lambda v: v.flatten(0, 1))

