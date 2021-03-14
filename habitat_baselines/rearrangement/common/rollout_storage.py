#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import numpy as np
import torch

from habitat_baselines.rearrangement.common.dictlist import DictList


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        goal_state_dataset=None,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        self.running_scenes = [[]] * (num_steps + 1)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        if goal_state_dataset is not None:
            self.ground_truth_buffer = goal_state_dataset
            self.ground_truth_buffer_size = len(goal_state_dataset)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        scene_ids,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.running_scenes[self.step + 1] = scene_ids


        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.running_scenes[0] = self.running_scenes[self.step]
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch, discr_batch_size, discr_rho):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            discr_observations_batch = defaultdict(list)
            discr_targets_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )

                adv_targ.append(advantages[: self.step, ind])
                discriminator_observations, discriminator_targets = self.sample_discriminator_observations(
                    self.step, discr_batch_size, ind, discr_rho, self.actions.device
                )
                for sensor in self.observations:
                    discr_observations_batch[sensor].append(discriminator_observations[sensor])
                discr_targets_batch.append(discriminator_targets)

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )
                discr_observations_batch[sensor] = torch.stack(discr_observations_batch[sensor], 1)

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)
            discr_targets_batch = torch.stack(discr_targets_batch, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            for sensor in discr_observations_batch:
                discr_T, discr_N = discr_observations_batch[sensor].shape[0], discr_observations_batch[sensor].shape[1]
                discr_observations_batch[sensor] = self._flatten_helper(discr_T, discr_N, discr_observations_batch[sensor])

            discr_targets_batch = self._flatten_helper(T, N, discr_targets_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                discr_observations_batch,
                discr_targets_batch
            )
    
    def sample_discriminator_observations(self, num_steps, batch_size, env_index, discr_rho, device):
        half_batch_size = batch_size // 2
        discr_experience_batch_size = int(half_batch_size / discr_rho)

        scene_step_count_map = defaultdict(int)
        for scenes in self.running_scenes[1:]:
            scene_step_count_map[scenes[env_index]] += 1

        ground_truth_sample_obs = defaultdict(list)
        scenes = scene_step_count_map.keys()
        total_ground_truth_samples = 0
        for scene in scenes:
            percent_steps_in_rollout = num_steps / scene_step_count_map[scene]
            scene_batch_size = int(half_batch_size * percent_steps_in_rollout)

            # If on last scene pick all remaining gt samples from last scene
            if total_ground_truth_samples + scene_batch_size < half_batch_size and scene == scenes[-1]:
                scene_batch_size += (half_batch_size - (total_ground_truth_samples + scene_batch_size))

            scene_ground_truth_buffer_size = self.ground_truth_buffer.get_scene_episode_length(scene)
            indices = np.random.randint(scene_ground_truth_buffer_size, size=scene_batch_size)
            for idx in indices:
                observations = self.ground_truth_buffer.get_item(idx, scene)
                for sensor in self.observations.keys():
                    ground_truth_sample_obs[sensor].append(observations[sensor])
            total_ground_truth_samples += scene_batch_size

        for sensor in self.observations.keys():
            ground_truth_sample_obs[sensor] = torch.stack(ground_truth_sample_obs[sensor]).squeeze(1)

        experience_sample_obs = defaultdict(list)
        if num_steps >= discr_experience_batch_size:
            indices = np.random.randint(num_steps, size=discr_experience_batch_size)
            for sensor in self.observations.keys():
                experience_sample_obs[sensor] = self.observations[sensor][indices, env_index]

        discr_observations = {}
        for sensor in self.observations.keys():
            discr_observations[sensor] = torch.cat((ground_truth_sample_obs[sensor], experience_sample_obs[sensor]))

        discr_targets = torch.ones(half_batch_size, device=device)
        discr_targets = torch.cat([discr_targets, -1 * discr_targets], 0)
        return discr_observations, discr_targets

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
