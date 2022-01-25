#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet


class DiscriminatorHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, reward_type, eps=1e-12):
        super().__init__()
        self.net = net
        self.reward_type = reward_type
        self.eps = eps

        self.discriminator = DiscriminatorHead(self.net.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *x):
        out, hidden_states = self.net(*x)
        logits = self.discriminator(out)
        return logits, hidden_states

    def compute_rewards(self, observations, rnn_hidden_states, prev_actions, masks):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        preds = self.discriminator(features)
        rewards = self.sigmoid(preds)

        if self.reward_type == "gail":
            rewards = (rewards + self.eps).log()
        elif self.reward_type == "airl":
            rewards = (rewards + self.eps).log() - (1 - rewards + self.eps).log()
        return rewards, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


@baseline_registry.register_discriminator
class ObjectNavDiscriminator(Discriminator):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        discriminator_config: Config = None,
        reward_type: str = "gail",
        **kwargs
    ):
        if discriminator_config is not None:
            discrete_actions = (
                discriminator_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                discriminator_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
            ),
            reward_type
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        print(cls)
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            discriminator_config=config.RL.POLICY,
            reward_type=config.IL.GAIL.reward_type
        )
