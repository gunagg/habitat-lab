#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rearrangement.reward_modeling.models.discriminator import DiscriminatorModel

EPS_PPO = 1e-5

class WeightClipper:

    def __init__(self, max_column_norm):
        self._max_column_norm = max_column_norm

    def __call__(self, module):
        if hasattr(module, 'weight'):
            max_column_norm = torch.tensor(self._max_column_norm, device=module.weight.device)
        if isinstance(module, torch.nn.Linear):
            # (out, in)
            module.weight.data /= torch.max(
                max_column_norm,
                torch.norm(module.weight.data, 2, 1)[:, None])
        elif isinstance(module, torch.nn.Conv2d):
            # (in, out, kw, kh)
            n_out = module.weight.shape[1]
            module.weight.data /= torch.max(
                max_column_norm,
                torch.norm(module.weight.data.permute(1, 0, 2, 3).contiguous().view(n_out, -1), 2, 1)
                [None, :, None, None])
        elif isinstance(module, torch.nn.Embedding):
            # (#embeddings, dim)
            module.weight.data /= torch.max(
                max_column_norm,
                torch.norm(module.weight.data, 2, 1)[:, None])

class PPOAgile(nn.Module):
    def __init__(
        self,
        actor_critic: Policy,
        discriminator: DiscriminatorModel,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
        discr_lr: Optional[float] = None,
        discr_rho: float = 1.0,
        discr_batch_size: int = 128,
        discr_thr: float = 0.0,
    ) -> None:

        super().__init__()

        self.actor_critic = actor_critic
        self.discriminator = discriminator

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )
        self.discriminator_optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, discriminator.parameters())),
            lr=discr_lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage
        self.discr_rho = discr_rho
        self.discr_thr = discr_thr
        self.discr_batch_size = discr_batch_size

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        discr_loss_epoch = 0.0
        discr_accuracy_epoch = 0.0
        num_discr_updates = 0.0

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch, self.discr_batch_size, self.discr_rho
            )

            for sample in data_generator:
                (
                    obs_batch,
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
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Check if we can reject (1 - p) negative samples for discriminator update
                n_envs = recurrent_hidden_states_batch.size(1)
                discr_loss = 0
                half_discr_batch =  (self.discr_batch_size) // 2
                discr_experience_batch_size = int(half_discr_batch / self.discr_rho)
                if rollouts.step >= discr_experience_batch_size:
                    discr_logits = self.discriminator(discr_observations_batch)
                    half_discr_batch =  (n_envs * self.discr_batch_size) // 2

                    # take `half_discr_batch` most negative ones
                    experience_order = np.argsort(discr_logits[half_discr_batch:].data.cpu().numpy(), axis=0)
                    discr_logits_filtered = discr_logits[
                        list(range(half_discr_batch))
                        + list(half_discr_batch + experience_order[:half_discr_batch])]
                    
                    discr_logits_filtered = discr_logits_filtered.contiguous().view(-1)
                    discr_loss = F.softplus(-discr_logits_filtered * discr_targets_batch).mean()
                    discr_accuracy = ((discr_logits_filtered > self.discr_thr) == (discr_targets_batch > 0)).float().mean()
                    num_discr_updates += 1

                self.optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    + discr_loss
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.discriminator_optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                discr_loss_epoch += discr_loss.item()
                discr_accuracy_epoch += discr_accuracy

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        discr_loss_epoch /= num_updates
        discr_accuracy_epoch /= num_discr_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, discr_loss_epoch, discr_accuracy_epoch

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )
        # nn.utils.clip_grad_norm_(
        #     self.discriminator.parameters(), self.max_grad_norm
        # )
        self.discriminator.apply(WeightClipper(0))

    def after_step(self) -> None:
        pass
