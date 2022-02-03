#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple

import abc
import numpy as np
import torch

from gym import spaces
from torch import nn as nn

from habitat.config import Config
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, Policy

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
        embed_actions: bool = False,
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
            ObjectNavDiscriminatorNet(
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
                embed_actions=embed_actions,
            ),
            reward_type
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
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
            reward_type=config.IL.GAIL.reward_type,
            embed_actions=config.IL.GAIL.embed_actions,
        )

class ObjectNavDiscriminatorNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        embed_actions: bool = False,
    ):
        super().__init__()

        self.discrete_actions = discrete_actions
        self.embed_actions = embed_actions
        rnn_input_size = 0

        if self.embed_actions:
            if discrete_actions:
                self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
            else:
                self.prev_action_embedding = nn.Linear(action_space.n, 32)

            self._n_prev_action = 32
            rnn_input_size += self._n_prev_action

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space if not force_blind_policy else spaces.Dict({}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self.n_input = rnn_input_size + self._hidden_size

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                visual_feats = self.visual_encoder(observations)

            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.embed_actions:
            if self.discrete_actions:
                prev_actions = prev_actions.squeeze(-1)
                start_token = torch.zeros_like(prev_actions)
                prev_actions = self.prev_action_embedding(
                    torch.where(masks.view(-1), prev_actions + 1, start_token)
                )
            else:
                prev_actions = self.prev_action_embedding(
                    masks * prev_actions.float()
                )

            x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks
        )

        return out, rnn_hidden_states
