#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lib2to3.pgen2.token import OP
from optparse import Option
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.il_rollout_storage import ILRolloutStorage

EPS_PPO = 1e-5


class GAIL(nn.Module):
    def __init__(
        self,
        discriminator: nn.Module,
        num_envs: int,
        num_mini_batch: int,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_normalized_advantage: Optional[float] = True
    ) -> None:

        super().__init__()

        self.discriminator = discriminator

        self.num_mini_batch = num_mini_batch
        self.use_normalized_advantage = use_normalized_advantage

        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, discriminator.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(discriminator.parameters()).device

    def forward(self, *x):
        raise NotImplementedError
    
    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1]
            - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, agent_rollouts: RolloutStorage, expert_rollouts: ILRolloutStorage, num_update) -> Tuple[float, float, float]:
        advantages = self.get_advantages(agent_rollouts)
        avg_discr_loss = 0.0
        avg_agent_loss = 0.0
        avg_expert_loss = 0.0
        avg_discr_accuracy = 0.0
        avg_agent_accuracy = 0.0
        avg_expert_accuracy = 0.0

        profiling_wrapper.range_push("GAIL update epoch")
        agent_sampler = agent_rollouts.recurrent_generator(
            advantages, self.num_mini_batch
        )
        expert_sampler = expert_rollouts.recurrent_generator(
            self.num_mini_batch
        )
        bce_loss = torch.nn.BCEWithLogitsLoss()
        expert_hidden_states = []
        agent_hidden_states = []

        for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
            (
                expert_logits,
                expert_batch_hidden_states,
            ) = self.discriminator(
                expert_batch["observations"],
                expert_batch["recurrent_hidden_states"],
                expert_batch["prev_actions"],
                expert_batch["masks"],
            )

            targets = torch.ones(expert_logits.shape).to(self.device)
            expert_loss = bce_loss(expert_logits, targets)
            expert_preds = (torch.sigmoid(expert_logits) > 0.5).long()
            expert_accuracy = torch.sum(expert_preds == targets) / targets.shape[0]

            (
                agent_logits,
                agent_batch_hidden_states,
            ) = self.discriminator(
                agent_batch["observations"],
                agent_batch["discr_recurrent_hidden_states"],
                agent_batch["prev_actions"],
                agent_batch["masks"],
            )

            targets = torch.zeros(agent_logits.shape).to(self.device)
            agent_loss = bce_loss(agent_logits, targets)
            agent_preds = (torch.sigmoid(agent_logits) > 0.5).long()
            agent_accuracy = torch.sum(agent_preds == targets) / targets.shape[0]

            total_loss = expert_loss + agent_loss
            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            avg_discr_loss += total_loss.item()
            avg_agent_loss += agent_loss.item()
            avg_expert_loss += expert_loss.item()
            avg_discr_accuracy += (agent_accuracy + expert_accuracy) / 2
            avg_agent_accuracy += agent_accuracy
            avg_expert_accuracy += expert_accuracy

            expert_hidden_states.append(expert_batch_hidden_states)
            agent_hidden_states.append(agent_batch_hidden_states)

        profiling_wrapper.range_pop()

        expert_hidden_states = torch.cat(expert_hidden_states, dim=0)
        agent_hidden_states = torch.cat(agent_hidden_states, dim=0)

        avg_discr_loss /= self.num_mini_batch
        avg_agent_loss /= self.num_mini_batch
        avg_expert_loss /= self.num_mini_batch
        avg_discr_accuracy /= self.num_mini_batch
        avg_agent_accuracy /= self.num_mini_batch
        avg_expert_accuracy /= self.num_mini_batch

        return (
            avg_discr_loss,
            avg_agent_loss,
            avg_expert_loss,
            avg_discr_accuracy,
            avg_agent_accuracy,
            avg_expert_accuracy,
            expert_hidden_states,
            agent_hidden_states,
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
