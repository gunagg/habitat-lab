import math
import sys
from typing import Dict, Iterable, Tuple

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
from habitat_baselines.rearrangement.models.encoders.instruction import InstructionEncoder
from habitat_baselines.rearrangement.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
    ResnetSemSeqEncoder,
)
from habitat_baselines.rearrangement.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet
from habitat_baselines.objectnav.models.rednet import load_rednet
from habitat_baselines.utils.common import CategoricalNet, CustomFixedCategorical
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin

from habitat.utils.visualizations.utils import observations_to_image, images_to_video


SEMANTIC_EMBEDDING_SIZE = 4
class SemSegSeqNet(Net):
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

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, model_config.DEPTH_ENCODER.output_size
            )
        elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
                trainable=model_config.DEPTH_ENCODER.trainable,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
            "ResnetRGBEncoder",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
            self.rgb_encoder = SimpleRGBCNN(
                observation_space, model_config.RGB_ENCODER.output_size
            )
        elif model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, model_config.RGB_ENCODER.output_size, device
            )
        elif model_config.RGB_ENCODER.cnn_type == "ResnetRGBEncoder":
            self.rgb_encoder = ResnetRGBEncoder(
                observation_space,
                output_size=model_config.RGB_ENCODER.output_size,
                backbone=model_config.RGB_ENCODER.backbone,
                trainable=model_config.RGB_ENCODER.train_encoder,
            )

        sem_seg_output_size = 0
        self.semantic_predictor = None
        if model_config.USE_SEMANTICS:
            # self.semantic_predictor = load_rednet(
            #     device,
            #     ckpt=model_config.SEMANTIC_ENCODER.rednet_ckpt,
            #     resize=True # since we train on half-vision
            # )
            # self.semantic_predictor.eval()
            # # Disable gradients
            # for param in self.semantic_predictor.parameters():
            #     param.requires_grad_(False)

            sem_embedding_size = model_config.SEMANTIC_ENCODER.embedding_size
            # self.semantic_embedder = nn.Embedding(40 + 2, sem_embedding_size)

            rgb_shape = observation_space.spaces["rgb"].shape
            spaces = {
                "semantic": Box(
                    low=0,
                    high=255,
                    shape=(rgb_shape[0], rgb_shape[1], sem_embedding_size),
                    dtype=np.uint8,
                ),
            }
            sem_obs_space = Dict(spaces)
            self.sem_seg_encoder = ResnetSemSeqEncoder(
                sem_obs_space,
                output_size=model_config.SEMANTIC_ENCODER.output_size,
                backbone=model_config.SEMANTIC_ENCODER.backbone,
                trainable=model_config.SEMANTIC_ENCODER.train_encoder,
                semantic_embedding_size=sem_embedding_size
            )
            sem_seg_output_size = model_config.SEMANTIC_ENCODER.output_size
            logger.info("Setting up Sem Seg model")

        # Init the RNN state decoder
        rnn_input_size = (
            model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
            + sem_seg_output_size
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

    def get_semantic_observations(self, observations_batch):
        with torch.no_grad():
            semantic_observations = self.semantic_predictor(observations_batch["rgb"], observations_batch["depth"])
            return semantic_observations

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"].flatten(start_dim=1)
            
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

            return goal_visible_area.unsqueeze(-1)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """

        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]

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

        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = [depth_embedding, rgb_embedding]

        if self.model_config.USE_SEMANTICS:
            if self.model_config.embed_sge:
                sge_embedding = self._extract_sge(observations)
                x.append(sge_embedding)
            sem_seg_embedding = self.sem_seg_encoder(observations)
            x.append(sem_seg_embedding)

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


class SemSegSeqModel(nn.Module):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config, device
    ):
        super().__init__()
        self.net = SemSegSeqNet(
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

class SemSegSeqPolicy(nn.Module):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config, device
    ):
        super().__init__()
        self.actor_critic = SemSegSeqModel(
            observation_space=observation_space,
            model_config=model_config,
            action_space=action_space,
            device=device,
        )
        self.device = device
    
    def forward(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> CustomFixedCategorical:

        logits, rnn_hidden_states = self.actor_critic(
            observations, rnn_hidden_states, prev_actions, masks
        )

        return logits, rnn_hidden_states


class DDPSemSegSeqModel(DecentralizedDistributedMixin, SemSegSeqPolicy):
    pass