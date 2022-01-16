#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage


class GAIL(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_envs: int,
        num_mini_batch: int,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:

        super().__init__()

        self.model = model

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(model.parameters()).device

    def forward(self, *x):
        raise NotImplementedError

    def update(self, agent_rollouts: RolloutStorage, expert_rollouts: RolloutStorage) -> Tuple[float, float, float]:
        total_loss_epoch = 0.0

        profiling_wrapper.range_push("GAIL update epoch")
        agent_sampler = agent_rollouts.recurrent_generator(
            self.num_mini_batch
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
            ) = self.model(
                expert_batch["observations"],
                expert_batch["recurrent_hidden_states"],
                expert_batch["prev_actions"],
                expert_batch["masks"],
            )

            targets = torch.ones(expert_logits.shape).to(self.device)
            expert_loss = bce_loss(expert_logits, targets)

            (
                agent_logits,
                agent_batch_hidden_states,
            ) = self.model(
                agent_batch["observations"],
                agent_batch["discr_recurrent_hidden_states"],
                agent_batch["prev_actions"],
                agent_batch["masks"],
            )

            targets = torch.zeros(agent_logits.shape).to(self.device)
            agent_loss = bce_loss(agent_logits, targets)

            total_loss = expert_loss + agent_loss
            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            total_loss_epoch += total_loss.item()
            expert_hidden_states.append(expert_batch_hidden_states)
            agent_hidden_states.append(agent_batch_hidden_states)

        profiling_wrapper.range_pop()

        expert_hidden_states = torch.cat(expert_hidden_states, dim=1)
        agent_hidden_states = torch.cat(agent_hidden_states, dim=1)

        total_loss_epoch /= self.num_mini_batch

        return total_loss_epoch

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
