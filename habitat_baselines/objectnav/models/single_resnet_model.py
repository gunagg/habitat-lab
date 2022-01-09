import math
import sys
from typing import Dict, Iterable, Tuple

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from gym.spaces import Discrete, Dict, Box
from habitat import Config, logger
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoalSensor,
    task_cat2mpcat40
)
from habitat_baselines.objectnav.models.encoders.resnet_encoders import (
    ResnetEncoder
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.utils.common import CategoricalNet, CustomFixedCategorical
from habitat_baselines.common.baseline_registry import baseline_registry


SEMANTIC_EMBEDDING_SIZE = 4
class SingleResnetSeqNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions, device):
        super().__init__()
        self.model_config = model_config

        # Init the visual encoder        
        if model_config.USE_SEMANTICS:
            sem_embedding_size = model_config.SEMANTIC_ENCODER.embedding_size
            self.embed_goal_seg = hasattr(model_config, "embed_goal_seg") and model_config.embed_goal_seg 
            if self.embed_goal_seg:
                sem_embedding_size = 1
            else:
                logger.info("Not setting up goal seg")

            self.semantic_embedder = nn.Embedding(40 + 2, sem_embedding_size)

        self.visual_encoder = ResnetEncoder(
            observation_space,
            output_size=model_config.VISUAL_ENCODER.output_size,
            backbone=model_config.VISUAL_ENCODER.backbone,
            trainable=model_config.VISUAL_ENCODER.train_encoder,
            sem_embedding_size=model_config.SEMANTIC_ENCODER.embedding_size,
        )
        logger.info("Setting up visual encoder")

        # Init the RNN state decoder
        rnn_input_size = (
            model_config.VISUAL_ENCODER.output_size
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

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
            logger.info("\n\nSetting up Object Goal sensor")

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.embed_sge = model_config.embed_sge
        if self.embed_sge:
            self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40, device=device)
            rnn_input_size += 1

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=model_config.STATE_ENCODER.num_recurrent_layers,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in observations and "objectgoal" in observations:
            goal_semantic = observations["semantic"].contiguous()
            obj_semantic = observations["semantic"].contiguous().flatten(start_dim=1)
            
            if len(observations["objectgoal"].size()) == 3:
                observations["objectgoal"] = observations["objectgoal"].contiguous().view(
                    -1, observations["objectgoal"].size(2)
                )
            idx = self.task_cat2mpcat40[
                observations["objectgoal"].long()
            ]
            idx = idx.to(obj_semantic.device)
            if len(idx.size()) == 3:
                idx = idx.squeeze(1)

            goal_visible_pixels = (obj_semantic == idx).sum(dim=1) # Sum over all since we're not batched
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1))
            
            # logger.info("goal sem shape : {}, object gaol shape: {}".format(goal_semantic.shape, observations["objectgoal"].shape))
            goal_sem_seg_mask = (goal_semantic == observations["objectgoal"].contiguous().unsqueeze(-1)).float()
            # logger.info("goal visible shape: {}".format(goal_sem_seg_mask.shape))
            return goal_visible_area.unsqueeze(-1), goal_sem_seg_mask.unsqueeze(-1)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]
        x = []
        if len(rgb_obs.size()) == 5:
            observations["rgb"] = rgb_obs.contiguous().view(
                -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
            )
        
        if len(depth_obs.size()) == 5:
            observations["depth"] = depth_obs.contiguous().view(
                -1, depth_obs.size(2), depth_obs.size(3), depth_obs.size(4)
            )
        
        semantic_obs = observations["semantic"]
        if len(semantic_obs.size()) == 4:
            observations["semantic"] = semantic_obs.contiguous().view(
                -1, semantic_obs.size(2), semantic_obs.size(3)
            )
        
        if self.model_config.USE_SEMANTICS:
            sge_embedding, goal_sem_seg_mask = self._extract_sge(observations)
            x.append(sge_embedding)
            # Use goal segmentation instead of full segmentation mask
            # for all object categories
            if self.embed_goal_seg:
                observations["semantic"] = goal_sem_seg_mask
            else:
                categories = observations["semantic"].long() + 1
                observations["semantic"] = self.semantic_embedder(categories)

        visual_embedding = self.visual_encoder(observations)

        x.append(visual_embedding)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))
        
        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class SingleResNetSeqModel(nn.Module):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config, device
    ):
        super().__init__()
        self.net = SingleResnetSeqNet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
            device=device,
        )
        self.action_distribution = CategoricalNet(
            self.net.output_size, action_space.n
        )
        self.train()
    
    def forward(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:

        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        return distribution.logits, rnn_hidden_states