import math
import sys
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.rearrangement.il.models.encoders.instruction import InstructionEncoder
from habitat_baselines.rearrangement.il.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder
)
from habitat_baselines.rearrangement.il.models.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.utils.common import CategoricalNet, CustomFixedCategorical


class Seq2SeqNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(model_config.INSTRUCTION_ENCODER)

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
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
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

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )

        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

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

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        x = torch.cat([instruction_embedding, depth_embedding, rgb_embedding], dim=1)

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


class Seq2SeqModel(nn.Module):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__()
        self.net = Seq2SeqNet(
            observation_space=observation_space,
            model_config=model_config,
            num_actions=action_space.n,
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