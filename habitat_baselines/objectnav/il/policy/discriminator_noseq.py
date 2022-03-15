#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple

import abc
import math
import numpy as np
import torch

from gym import spaces
from torch import nn as nn

from habitat import logger
from habitat.config import Config
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net

class DiscriminatorHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class DiscriminatorNoSeq(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, reward_type, eps=1e-12):
        super().__init__()
        self.net = net
        self.reward_type = reward_type
        self.eps = eps

        self.discriminator = DiscriminatorHead(self.net.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *x):
        out = self.net(*x)
        logits = self.discriminator(out)
        return (logits, )

    def compute_rewards(self, observations, prev_actions, masks):
        features = self.net(
            observations, prev_actions, masks
        )
        preds = self.discriminator(features)
        rewards = self.sigmoid(preds)

        if self.reward_type == "gail":
            rewards = (rewards + self.eps).log()
        elif self.reward_type == "airl":
            rewards = (rewards + self.eps).log() - (1 - rewards + self.eps).log()
        return (rewards,)

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


@baseline_registry.register_discriminator_noseq
class ObjectNavDiscriminator(DiscriminatorNoSeq):
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
        no_memory: bool = False,
    ):
        super().__init__()

        self.discrete_actions = discrete_actions
        self.embed_actions = embed_actions
        self.no_memory = no_memory
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

        self.n_input = rnn_input_size + hidden_size
        self.fc = nn.Linear(self.n_input, self._hidden_size)
        self.dropout = nn.Dropout(0.5)

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
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
        # logger.info("out shape: {} -- {}".format(out.shape, self.n_input))

        out = self.dropout(self.fc(out))
        return out


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        dropout_prob: float = 0.0,
        n_input_channels = 15,
    ):
        super().__init__()
        self.n_input_channels = n_input_channels

        if "rgb" in observation_space.spaces:
            self._frame_size = tuple(observation_space.spaces["rgb"].shape[:2])
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[:2]
        else:
            self._n_input_rgb = 0

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self.n_input_channels
            )
            print("running mean in: {}".format(self.n_input_channels))
        else:
            self.running_mean_and_var = nn.Sequential()
        print("num in: {}".format(self.n_input_channels))

        if not self.is_blind:
            input_channels = self.n_input_channels
            self.backbone = make_backbone(input_channels, baseplanes, ngroups, dropout_prob=dropout_prob)

            final_spatial = np.array([math.ceil(
                d * self.backbone.final_spatial_compress
            ) for d in spatial_size])
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / np.prod(final_spatial))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial[0],
                final_spatial[1],
            )

    @property
    def is_blind(self):
        return self.n_input_channels == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB
            print("rgb", rgb_observations.shape)
            cnn_input.append(rgb_observations)

        x = torch.cat(cnn_input, dim=1)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x