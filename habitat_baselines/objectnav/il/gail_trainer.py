#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import log
import copy
import os
import sys
import time
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Union

import torch
from torch.optim.lr_scheduler import LambdaLR

from gym import Space
from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image, append_text_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.il_rollout_storage import ILRolloutStorage
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import (
    batch_obs,
    linear_decay,
)
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.objectnav.models.rednet import load_rednet
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.objectnav.il.algos.gail import GAIL
from habitat_baselines.objectnav.il.policy.discriminator import Discriminator 


@baseline_registry.register_trainer(name="gail")
class GAILTrainer(BaseRLTrainer):
    r"""Trainer class for behavior cloning.
    """
    supported_tasks = ["ObjectNav-v1"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)


        self.policy_action_space = self.agent_envs.action_spaces[0]
        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        observation_space = self.expert_envs.observation_spaces[0]
        self.obs_space = observation_space
        self.obs_transforms = get_active_obs_transforms(self.config)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.actor_critic.to(self.device)

        self.semantic_predictor = None
        if self.config.MODEL.USE_PRED_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=self.config.MODEL.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True, # since we train on half-vision
                num_classes=self.config.MODEL.SEMANTIC_ENCODER.num_classes
            )
            self.semantic_predictor.eval()

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    
    def _setup_discriminator(self, il_cfg):
        discriminator = baseline_registry.get_discriminator(il_cfg.DISCRIMINATOR.name)
        self.discriminator = discriminator.from_config(
            self.config, self.obs_space, self.policy_action_space
        )
        self.discriminator.to(self.device)

        self.gail = GAIL(
            discriminator=self.discriminator,
            num_envs=self.expert_envs.num_envs,
            num_mini_batch=il_cfg.num_mini_batch,
            lr=il_cfg.lr,
            eps=il_cfg.eps,
            max_grad_norm=il_cfg.max_grad_norm,
            use_normalized_advantage=il_cfg.use_normalized_advantage,
        )

    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "room_visitation_map", "exploration_metrics"}

    @profiling_wrapper.RangeContext("_collect_expert_rollout_step")
    def _collect_expert_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats, buffer_index=0
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        # fetch actions and environment state from replay buffer
        next_actions = rollouts.get_next_actions()
        actions = next_actions.long().unsqueeze(-1)
        step_data = [a.item() for a in next_actions.long().to(device="cpu")]

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        profiling_wrapper.range_pop()  # compute actions

        outputs = self.expert_envs.step(step_data)
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.bool,
            device=current_episode_reward.device,
        )
        done_masks = torch.logical_not(masks)

        current_episode_reward += rewards
        running_episode_stats["reward"] += done_masks * current_episode_reward  # type: ignore
        running_episode_stats["count"] += done_masks  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += done_masks * v  # type: ignore

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            actions,
            masks,
        )
        rollouts.advance_rollout(buffer_index)

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.expert_envs.num_envs


    @profiling_wrapper.RangeContext("_collect_agent_rollout_step")
    def _collect_agent_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats, buffer_index=0
    ):
        pth_time = 0.0
        env_time = 0.0
        num_envs = self.agent_envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        with torch.no_grad():
            step_batch = rollouts.buffers[
                rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        step_data = [a.item() for a in actions.to(device="cpu")]
        profiling_wrapper.range_pop()  # compute actions

        outputs = self.agent_envs.step(step_data)
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.bool,
            device=current_episode_reward.device,
        )
        done_masks = torch.logical_not(masks)

        current_episode_reward += rewards
        running_episode_stats["reward"] += done_masks * current_episode_reward  # type: ignore
        running_episode_stats["count"] += done_masks  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += done_masks * v  # type: ignore

        current_episode_reward *= masks

        rollouts.insert(
            next_observations=batch,
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            rewards=rewards,
            next_masks=masks,
            value_preds=values,
            buffer_index=buffer_index
        )
        rollouts.advance_rollout(buffer_index)

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.agent_envs.num_envs

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self, agent_rollouts, expert_rollouts, running_episode_stats):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()

        agent_hidden_states = torch.zeros(agent_rollouts.discr_recurrent_hidden_states[0].shape)
        expert_hidden_states = torch.zeros(agent_rollouts.discr_recurrent_hidden_states[0].shape)

        discr_loss = 0
        agent_loss = 0
        expert_loss = 0

        # Discriminator update
        self.gail.train()
        (
            discr_loss,
            agent_loss,
            expert_loss,
            expert_hidden_states,
            agent_hidden_states,
        ) = self.gail.update(agent_rollouts, expert_rollouts)

        # Overwrite environment rewards with discriminator preds
        self.gail.eval()
        with torch.no_grad():
            eps = 1e-20
            discr_recurrent_hidden_states = agent_rollouts.discr_recurrent_hidden_states[0].to(agent_rollouts.discr_recurrent_hidden_states.device)
            for i in range(agent_rollouts.current_rollout_step_idx):
                step_batch = agent_rollouts.buffers[i]
                reward, discr_recurrent_hidden_states = self.discriminator.compute_rewards(
                    step_batch["observations"],
                    discr_recurrent_hidden_states,
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )
                agent_rollouts.buffers["rewards"][i] = (reward + eps).log()
                running_episode_stats["pred_reward"] += agent_rollouts.buffers["rewards"][i].detach().cpu() * agent_rollouts.buffers["masks"][i].detach().cpu()

        # Agent update
        with torch.no_grad():
            step_batch = agent_rollouts.buffers[
                agent_rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        agent_rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            agent_rollouts
        )

        agent_rollouts.after_update(agent_hidden_states)
        expert_rollouts.after_update(expert_hidden_states)

        pth_time = time.time() - t_update_model

        return (
            pth_time,
            value_loss,
            action_loss,
            dist_entropy,
            discr_loss,
            agent_loss,
            expert_loss,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self.expert_envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        self.agent_envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        gail_cfg = self.config.IL.GAIL
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(self.config.RL.PPO)
        self._setup_discriminator(self.config.IL.GAIL)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
        logger.info(
            "discriminator number of parameters: {}".format(
                sum(param.numel() for param in self.discriminator.parameters())
            )
        )

        expert_rollouts = ILRolloutStorage(
            gail_cfg.num_steps,
            self.expert_envs.num_envs,
            self.obs_space,
            self.expert_envs.action_spaces[0],
            self.config.MODEL.STATE_ENCODER.hidden_size,
            self.config.MODEL.STATE_ENCODER.num_recurrent_layers,
        )
        expert_rollouts.to(self.device)

        agent_rollouts = RolloutStorage(
            gail_cfg.num_steps,
            self.agent_envs.num_envs,
            self.obs_space,
            self.agent_envs.action_spaces[0],
            self.config.MODEL.STATE_ENCODER.hidden_size,
            self.config.MODEL.STATE_ENCODER.num_recurrent_layers,
        )
        agent_rollouts.to(self.device)

        expert_observations = self.expert_envs.reset()
        expert_obs_batch = batch_obs(expert_observations, device=self.device)
        expert_obs_batch = apply_obs_transforms_batch(expert_obs_batch, self.obs_transforms)

        agent_observations = self.agent_envs.reset()
        agent_obs_batch = batch_obs(agent_observations, device=self.device)
        agent_obs_batch = apply_obs_transforms_batch(agent_obs_batch, self.obs_transforms)

        current_episode_reward = torch.zeros(self.agent_envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.agent_envs.num_envs, 1),
            reward=torch.zeros(self.agent_envs.num_envs, 1),
            pred_reward=torch.zeros(self.agent_envs.num_envs, 1),
        )
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=gail_cfg.reward_window_size)
        )

        expert_current_episode_reward = torch.zeros(self.agent_envs.num_envs, 1)
        expert_running_episode_stats = dict(
            count=torch.zeros(self.agent_envs.num_envs, 1),
            reward=torch.zeros(self.agent_envs.num_envs, 1),
        )
        expert_window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=gail_cfg.reward_window_size)
        )

        t_start = time.time()
        agent_env_time = 0
        agent_pth_time = 0
        agent_count_steps = 0
        expert_env_time = 0
        expert_count_steps = 0
        expert_pth_time = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),  # type: ignore
        )
        self.possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")
                
                self.current_update = update

                if gail_cfg.use_linear_lr_decay and update > 0:
                    lr_scheduler.step()  # type: ignore

                if gail_cfg.use_linear_clip_decay and update > 0:
                    self.agent.clip_param = gail_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                profiling_wrapper.range_push("expert rollouts loop")
                for _step in range(gail_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_expert_rollout_step(
                        expert_rollouts, expert_current_episode_reward, expert_running_episode_stats
                    )
                    expert_pth_time += delta_pth_time
                    expert_env_time += delta_env_time
                    expert_count_steps += delta_steps
                profiling_wrapper.range_pop()  # rollouts loop

                self.agent.eval()
                profiling_wrapper.range_push("agent rollouts loop")
                for _step in range(gail_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_agent_rollout_step(
                        agent_rollouts, current_episode_reward, running_episode_stats
                    )
                    agent_pth_time += delta_pth_time
                    agent_env_time += delta_env_time
                    agent_count_steps += delta_steps
                profiling_wrapper.range_pop()  # rollouts loop

                (
                    update_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                    discr_loss,
                    agent_loss,
                    expert_loss,
                ) = self._update_agent(agent_rollouts, expert_rollouts, running_episode_stats)
                agent_pth_time += update_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                for k, v in expert_running_episode_stats.items():
                    expert_window_episode_stats[k].append(v.clone())

                expert_deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in expert_window_episode_stats.items()
                }
                expert_deltas["count"] = max(expert_deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], agent_count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, agent_count_steps)
                

                expert_metrics = {
                    k: v / expert_deltas["count"]
                    for k, v in expert_deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(expert_metrics) > 0:
                    writer.add_scalars("expert_metrics", expert_metrics, expert_count_steps)

                losses = [value_loss, action_loss, discr_loss, agent_loss, expert_loss]
                writer.add_scalars(
                    "policy losses",
                    {k: l for l, k in zip(losses, ["value", "action"])},
                    agent_count_steps,
                )

                writer.add_scalars(
                    "discriminator losses",
                    {k: l for l, k in zip(losses, ["discrimator", "agent_discr", "expert_discr"])},
                    agent_count_steps,
                )

                # log stats
                if update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}".format(
                            update, (agent_count_steps + expert_count_steps) / (time.time() - t_start)
                        )
                    )
                    logger.info(
                        "update: {}\tvalue loss: {:.3f}\taction loss: {:.3f}".format(
                            update, value_loss, action_loss
                        )
                    )
                    logger.info(
                        "update: {}\tdiscr loss: {:.3f}\texpert discr loss: {:.3f}\tagent discr loss: {:.3f}".format(
                            update, discr_loss, expert_loss, agent_loss
                        )
                    )

                    logger.info(
                        "update: {}\texpert-env-time: {:.3f}s\texpert-pth-time: {:.3f}s\t"
                        "expert-frames: {}".format(
                            update, expert_env_time, expert_pth_time, expert_count_steps
                        )
                    )

                    logger.info(
                        "update: {}\tagent-env-time: {:.3f}s\tagent-pth-time: {:.3f}s\t"
                        "agent-frames: {}".format(
                            update, agent_env_time, agent_pth_time, agent_count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                    logger.info(
                        "Average expert window size: {}  {}".format(
                            len(expert_window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / expert_deltas["count"])
                                for k, v in expert_deltas.items()
                                if k != "count"
                            ),
                        )
                    )

   
                if update == self.config.MODEL.SWITCH_TO_PRED_SEMANTICS_UPDATE - 1:
                    self.save_checkpoint(
                        f"ckpt_gt_best.{count_checkpoints}.pth",
                        dict(
                            step=agent_count_steps,
                            lr_scheduler=lr_scheduler.state_dict(),
                            optimizer=self.agent.optimizer.state_dict(),
                        ),
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=agent_count_steps)
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.expert_envs.close()
            self.agent_envs.close()
