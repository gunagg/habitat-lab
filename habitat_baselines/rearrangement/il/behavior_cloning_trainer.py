#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import contextlib
import os
import json
import random
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
import sys
import torch
import tqdm
import torch.nn.functional as F

from torch import Tensor
from torch import distributed as distrib
from numpy import ndarray
from torch.utils.data import DataLoader

from habitat import Config, logger
from habitat.core.env import Env, RLEnv
from habitat.core.vector_env import VectorEnv
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rearrangement.common.aux_losses import AuxLosses
from habitat_baselines.rearrangement.dataset.dataset import RearrangementDataset
from habitat_baselines.rearrangement.dataset.episode_dataset import RearrangementEpisodeDataset, collate_fn
from habitat_baselines.rearrangement.il.models.model import RearrangementLstmCnnAttentionModel
from habitat_baselines.rearrangement.il.models.seq_2_seq_model import Seq2SeqNet, Seq2SeqModel
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
)
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import get_habitat_sim_action, get_habitat_sim_action_str
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum, get_distance
from PIL import Image


@baseline_registry.register_trainer(name="rearrangement-behavior-cloning")
class RearrangementBCTrainer(BaseILTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["RearrangementTask-v0"]

    def __init__(self, config=None):
        super().__init__(config)

        self.obs_transforms = []

        if config is not None:
            logger.info(f"config: {config}")

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # self.device = (
        #     torch.device("cpu")
        # )

    def _make_results_dir(self):
        r"""Makes directory for saving eqa-cnn-pretrain eval results."""
        for s_type in ["rgb", "seg", "depth"]:
            dir_name = self.config.RESULTS_DIR.format(split="val", type=s_type)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

    def _save_results(
        self,
        gt_rgb: torch.Tensor,
        pred_rgb: torch.Tensor,
        gt_seg: torch.Tensor,
        pred_seg: torch.Tensor,
        gt_depth: torch.Tensor,
        pred_depth: torch.Tensor,
        path: str,
    ) -> None:
        r"""For saving EQA-CNN-Pretrain reconstruction results.

        Args:
            gt_rgb: rgb ground truth
            preg_rgb: autoencoder output rgb reconstruction
            gt_seg: segmentation ground truth
            pred_seg: segmentation output
            gt_depth: depth map ground truth
            pred_depth: depth map output
            path: to write file
        """

        save_rgb_results(gt_rgb[0], pred_rgb[0], path)
        save_seg_results(gt_seg[0], pred_seg[0], path)
        save_depth_results(gt_depth[0], pred_depth[0], path)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
    
    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: Union[VectorEnv, RLEnv, Env],
        test_recurrent_hidden_states: Tensor,
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        batch: Dict[str, Tensor],
        rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
        action_indices: List[int],
        reference_replays: List[Any],
        action_match_count: List[Any]
    ) -> Tuple[
        Union[VectorEnv, RLEnv, Env],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, Tensor],
        List[List[Any]],
        List[int],
        List[Any],
        List[Any]
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]

            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]
            action_indices = [action_indices[i] for i in state_index]
            reference_replays = [reference_replays[i] for i in state_index]
            action_match_count = [action_match_count[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
            action_indices,
            reference_replays,
            action_match_count
        )
    

    def _setup_model(self, observation_space, action_space, model_config):
        model_config.defrost()
        model_config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        model_config.freeze()

        model = Seq2SeqModel(observation_space, action_space, model_config)
        return model


    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        config = self.config
        if config.IL.distrib_training:
            self.local_rank, tcp_store = init_distrib_slurm(
                self.config.IL.distrib_backend
            )

            self.world_rank = distrib.get_rank()
            self.world_size = distrib.get_world_size()

            self.config.defrost()
            self.config.TORCH_GPU_ID = self.local_rank
            self.config.SIMULATOR_GPU_ID = self.local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                self.world_rank * self.config.NUM_PROCESSES
            )
            self.config.freeze()

            if torch.cuda.is_available():
                self.device = torch.device("cuda", self.local_rank)
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        
        rearrangement_dataset = RearrangementEpisodeDataset(
            config,
            use_iw=config.IL.USE_IW,
            inflection_weight_coef=config.MODEL.inflection_weight_coef
        )
        batch_size = config.IL.BehaviorCloning.batch_size

        train_loader = DataLoader(
            rearrangement_dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=True,
        )

        logger.info(
            "[ train_loader has {} samples ]".format(
                len(rearrangement_dataset)
            )
        )

        action_space = self.envs.action_spaces[0]

        self.model = self._setup_model(
            self.envs.observation_spaces[0],
            action_space,
            config.MODEL
        )
        self.model = torch.nn.DataParallel(self.model, dim=1)
        self.model.to(self.device)

        if config.IL.distrib_training:
            # Distributed data parallel setup
            if torch.cuda.is_available():
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.device], output_device=self.device
                )
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(config.IL.BehaviorCloning.lr),
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=config.IL.BehaviorCloning.lr,
            steps_per_epoch=len(train_loader), epochs=config.IL.BehaviorCloning.max_epochs
        )
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        rnn_hidden_states = torch.zeros(
            config.MODEL.STATE_ENCODER.num_recurrent_layers,
            batch_size,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        epoch, t = 1, 0
        softmax = torch.nn.Softmax(dim=1)
        AuxLosses.activate()
        with (
            TensorboardWriter(
                config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
        ) as writer:
            while epoch <= config.IL.BehaviorCloning.max_epochs:
                start_time = time.time()
                avg_loss = 0.0

                for batch in train_loader:
                    torch.cuda.empty_cache()
                    t += 1

                    (
                     observations_batch,
                     gt_prev_action,
                     episode_not_done,
                     gt_next_action,
                     inflec_weights
                    ) = batch

                    optim.zero_grad()
                    AuxLosses.clear()

                    num_samples = gt_prev_action.shape[0]
                    timestep_batch_size = config.IL.BehaviorCloning.timestep_batch_size
                    num_steps = num_samples // timestep_batch_size + (num_samples % timestep_batch_size != 0)
                    batch_loss = 0
                    for i in range(num_steps):
                        start_idx = i * timestep_batch_size
                        end_idx = start_idx + timestep_batch_size
                        observations_batch_sample = {
                            k: v[start_idx:end_idx].to(device=self.device, non_blocking=True)
                            for k, v in observations_batch.items()
                        }

                        gt_next_action_sample = gt_next_action[start_idx:end_idx].long().to(self.device)
                        gt_prev_action_sample = gt_prev_action[start_idx:end_idx].long().to(self.device)
                        episode_not_dones_sample = episode_not_done[start_idx:end_idx].long().to(self.device)
                        inflec_weights_sample = inflec_weights[start_idx:end_idx].long().to(self.device)

                        logits, rnn_hidden_states = self.model(
                            observations_batch_sample,
                            rnn_hidden_states,
                            gt_prev_action_sample,
                            episode_not_dones_sample
                        )

                        T, N = gt_next_action_sample.shape
                        logits = logits.view(T, N, -1)
                        pred_actions = torch.argmax(logits, dim=2)

                        action_loss = cross_entropy_loss(logits.permute(0, 2, 1), gt_next_action_sample)
                        denom = inflec_weights_sample.sum(0)
                        denom[denom == 0.0] = 1
                        action_loss = ((inflec_weights_sample * action_loss).sum(0) / denom).mean()
                        loss = (action_loss / num_steps)
                        loss.backward()
                        batch_loss += loss.item()
                        rnn_hidden_states = rnn_hidden_states.detach()

                    avg_loss += batch_loss

                    if t % config.LOG_INTERVAL == 0:
                        logger.info(
                            "[ Epoch: {}; iter: {}; loss: {:.3f} ]".format(
                                epoch, t, batch_loss
                            )
                        )
                        writer.add_scalar("train_loss", loss, t)

                    optim.step()
                    scheduler.step()
                    rnn_hidden_states = torch.zeros(
                                            config.MODEL.STATE_ENCODER.num_recurrent_layers,
                                            batch_size,
                                            config.MODEL.STATE_ENCODER.hidden_size,
                                            device=self.device,
                                        )

                end_time = time.time()
                time_taken = "{:.1f}".format((end_time - start_time) / 60)
                avg_loss = avg_loss / len(train_loader)

                logger.info(
                    "[ Epoch {} completed. Time taken: {} minutes. ]".format(
                        epoch, time_taken
                    )
                )
                logger.info("[ Average loss: {:.3f} ]".format(avg_loss))
                writer.add_scalar("avg_train_loss", avg_loss, epoch)

                print("-----------------------------------------")

                if epoch % config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        self.model.module.state_dict(), "model_{}.ckpt".format(epoch)
                    )

                epoch += 1
        self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        config = self.config

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            if config.SHOW_TOP_DOWN_MAP:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        batch_size = config.IL.BehaviorCloning.batch_size

        rearrangement_dataset = RearrangementEpisodeDataset(
            config,
            use_iw=config.IL.USE_IW,
            inflection_weight_coef=config.MODEL.inflection_weight_coef
        )

        logger.info(
            "[ val_loader has {} samples ]".format(
                self.envs.count_episodes()
            )
        )

        action_space = self.envs.action_spaces[0]

        self.model = self._setup_model(
            self.envs.observation_spaces[0],
            action_space,
            config.MODEL
        )
        # self.model = torch.nn.DataParallel(self.model, dim=1)

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt_dict, strict=True)
        # self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        prev_actions = torch.zeros(
            self.envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        rnn_hidden_states = torch.zeros(
            config.MODEL.STATE_ENCODER.num_recurrent_layers,
            self.envs.num_envs,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        action_indices = [1] * self.envs.num_envs
        action_match_count = [(0, 0)] * self.envs.num_envs
        possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            reference_replays = [episode.reference_replay for episode in current_episodes]

            with torch.no_grad():
                (
                    logits,
                    rnn_hidden_states
                ) = self.model(
                    batch,
                    rnn_hidden_states,
                    prev_actions,
                    not_done_masks
                )

                actions = torch.argmax(logits, dim=1)
                prev_actions.copy_(actions.unsqueeze(1))

            # print(get_habitat_sim_action_str(actions[0].item()))
            for i in range(self.envs.num_envs):
                gt_action = get_habitat_sim_action(reference_replays[i][action_indices[i]].action)
                pred_action = actions[i].item()
                num_action_match = action_match_count[i][0] + (gt_action == pred_action)
                action_match_count[i] = (num_action_match, action_match_count[i][1] + 1)

            action_names = [possible_actions[a.item()] for a in actions.to(device="cpu")]
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]
            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            reference_replays = [episode.reference_replay for episode in current_episodes]
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                action_indices[i] = min(action_indices[i] + 1, len(reference_replays[i]) - 1)
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    avg_action_match = action_match_count[i][0] / action_match_count[i][1]
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["match_gt_trajectory"] = (avg_action_match == 1)
                    episode_stats["avg_match_actions"] = avg_action_match
                    episode_stats["ref_replay_len"] = len(current_episodes[i].reference_replay)
                    episode_stats["pred_actions"] = action_match_count[i][1]
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    next_episodes = self.envs.current_episodes()
                    action_match_count[i] = (0, 0)
                    action_indices[i] = 1

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []
                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {"rgb": batch["rgb"][i]}, infos[i]
                    )
                    frame = append_text_to_image(frame, "Action: {}".format(action_names[i]))
                    frame = append_text_to_image(frame, "Instruction: {}".format(next_episodes[i].instruction.instruction_text))
                    rgb_frames[i].append(frame)


            (
                self.envs,
                rnn_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                action_indices,
                reference_replays,
                action_match_count
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                rnn_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                action_indices,
                reference_replays,
                action_match_count
            )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)
        print(aggregated_stats)
        print(num_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_success_mean = aggregated_stats["success"] / num_episodes
        episode_spl_mean = aggregated_stats["spl"] / num_episodes
        episode_grab_success_mean = aggregated_stats["grab_success"] / num_episodes

        episode_agent_object_distance_mean = aggregated_stats["agent_object_distance"] / num_episodes
        episode_agent_receptacle_distance_mean = aggregated_stats["agent_receptacle_distance"] / num_episodes
        match_gt_trajectory_mean = aggregated_stats["match_gt_trajectory"] / num_episodes
        avg_match_actions_mean = aggregated_stats["avg_match_actions"] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        logger.info(f"Average episode success: {episode_success_mean:.6f}")
        logger.info(f"Average episode SPL: {episode_spl_mean:.6f}")
        logger.info(f"Average episode grab success: {episode_grab_success_mean:.6f}")
        logger.info(f"Average episode agent object distance: {episode_agent_object_distance_mean:.6f}")
        logger.info(f"Average episode agent receptacle distance: {episode_agent_receptacle_distance_mean:.6f}")
        logger.info(f"Average episode exact GT match: {match_gt_trajectory_mean:.6f}")
        logger.info(f"Average episode percentage action match: {avg_match_actions_mean:.6f}")

        self.envs.close()

        print ("environments closed")

    def get_agent_object_distance(self, agent_state, object_states, batch, obs_rgb):
        eval_agent_state = self.envs.get_agent_pose()[0]
        eval_object_states = self.envs.get_current_object_states()[0]
        if "semantic" in agent_state["sensor_data"]:
            del agent_state["sensor_data"]["semantic"]
        if "depth" in eval_agent_state["sensor_data"]:
            del eval_agent_state["sensor_data"]["depth"]
        if "depth" in agent_state["sensor_data"]:
            del agent_state["sensor_data"]["depth"]

        agent_l2_distance = get_distance(agent_state["position"], eval_agent_state["position"], "l2")
        agent_quat_distance = get_distance(agent_state["rotation"], eval_agent_state["rotation"], "quat")
        sensor_quat_distance = get_distance(agent_state["sensor_data"]["rgb"]["rotation"], eval_agent_state["sensor_data"]["rgb"]["rotation"], "quat")

        object_dist_map = {}
        for idx in range(len(object_states)):
            object_id = object_states[idx]["object_id"]
            eval_idx = -1
            for idx2 in range(len(eval_object_states)):
                eval_object_id = eval_object_states[idx2]["object_id"]
                handle = eval_object_states[idx2]["object_handle"].split("/")[-1].split(".")[0]
                if object_id == eval_object_id:
                    eval_idx = idx2
                    break
            trans_dist = -1
            rot_dist = -1
            if eval_idx != -1:
                trans_dist = get_distance(object_states[idx]["translation"], eval_object_states[eval_idx]["translation"], "l2")
                rot_dist = get_distance(object_states[idx]["rotation"], eval_object_states[eval_idx]["rotation"], "quat")
            object_dist_map[handle] = {
                "trans_dist": trans_dist,
                "rot_dist": rot_dist
            }

        # pred_img = "../psiturk-habitat-sim/data/training_results/preds/task_3/img_{}.png".format(count)
        # img = Image.fromarray(batch["rgb"].squeeze(0).numpy())
        # img.save(pred_img)

        # gt_img = "../psiturk-habitat-sim/data/training_results/gt/task_3/img_{}.png".format(count)
        # img = Image.fromarray(obs_rgb[count][0].numpy().astype(np.uint8))
        # img.save(gt_img)
        return agent_l2_distance, agent_quat_distance, sensor_quat_distance, object_dist_map

