#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
import sys
import torch
import tqdm

from torch import Tensor
from numpy import ndarray
from torch.utils.data import DataLoader

from habitat import Config, logger
from habitat.core.env import Env, RLEnv
from habitat.core.vector_env import VectorEnv
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rearrangement.common.aux_losses import AuxLosses
from habitat_baselines.rearrangement.data.dataset import RearrangementDataset
from habitat_baselines.rearrangement.data.episode_dataset import RearrangementEpisodeDataset
from habitat_baselines.rearrangement.il.models.model import RearrangementLstmCnnAttentionModel
from habitat_baselines.rearrangement.il.models.seq_2_seq_model import Seq2SeqNet, Seq2SeqModel
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import get_habitat_sim_action, get_habitat_sim_action_str
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

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if config is not None:
            logger.info(f"config: {config}")

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
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        batch: Dict[str, Tensor],
        rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
    ) -> Tuple[
        Union[VectorEnv, RLEnv, Env],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, Tensor],
        List[List[Any]],
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
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

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        
        rearrangement_dataset = RearrangementEpisodeDataset(config)
        batch_size = config.IL.BehaviorCloning.batch_size

        train_loader = DataLoader(
            rearrangement_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        logger.info(
            "[ train_loader has {} samples ]".format(
                len(rearrangement_dataset)
            )
        )

        instruction_vocab_dict = rearrangement_dataset.get_vocab_dict()
        action_space = self.envs.action_spaces[0]

        self.model = self._setup_model(
            self.envs.observation_spaces[0],
            action_space,
            config.MODEL
        )
        self.model.to(self.device)
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = torch.load(self.config.EVAL_CKPT_PATH_DIR, map_location=self.device)
        self.model.load_state_dict(ckpt_dict)


        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(config.IL.BehaviorCloning.lr),
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.9)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        test_rnn_hidden_states = torch.zeros(
            1,
            self.envs.num_envs,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        epoch, t = 1, 0
        softmax = torch.nn.Softmax(dim=1)
        AuxLosses.activate()
        with TensorboardWriter(
            config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while epoch <= config.IL.BehaviorCloning.max_epochs:
                start_time = time.time()
                avg_loss = 0.0

                for batch in train_loader:
                    t += 1

                    (
                     idx,
                     obs_rgb,
                     obs_depth,
                     obs_seg,
                     obs_instruction_tokens,
                     gt_next_action,
                     gt_prev_action,
                     episode_done,
                     inflec_weights
                    ) = batch

                    optim.zero_grad()
                    AuxLosses.clear()

                    obs_rgb = obs_rgb.to(self.device)
                    obs_depth = obs_depth.to(self.device)
                    obs_seg = obs_seg.to(self.device)
                    obs_instruction_tokens = obs_instruction_tokens.to(self.device)
                    gt_next_action = gt_next_action.long().to(self.device)
                    gt_prev_action = gt_prev_action.long().to(self.device)
                    episode_not_dones = episode_done.long().to(self.device)
                    inflec_weights = inflec_weights.long().to(self.device)

                    observations = {
                        "rgb": obs_rgb,
                        "depth": obs_depth,
                        "instruction": obs_instruction_tokens
                    }

                    distribution, rnn_hidden_states = self.model(observations, test_rnn_hidden_states, gt_prev_action, episode_not_dones)

                    logits = distribution.logits
                    action_loss = cross_entropy_loss(logits, gt_next_action.squeeze(0))

                    action_loss = ((inflec_weights * action_loss).sum(1) / inflec_weights.sum(1)).mean()
                    loss = action_loss
                    avg_loss += loss.item()

                    if t % config.LOG_INTERVAL == 0:
                        logger.info(
                            "[ Epoch: {}; iter: {}; loss: {:.3f} ]".format(
                                epoch, t, loss.item()
                            )
                        )
                        writer.add_scalar("train_total_loss", loss, t)
                        writer.add_scalar("train_action_loss", action_loss.item(), t)

                    loss.backward()
                    optim.step()
                    scheduler.step()

                end_time = time.time()
                time_taken = "{:.1f}".format((end_time - start_time) / 60)
                avg_loss = avg_loss / len(train_loader)

                logger.info(
                    "[ Epoch {} completed. Time taken: {} minutes. ]".format(
                        epoch, time_taken
                    )
                )
                logger.info("[ Average loss: {:.3f} ]".format(avg_loss))

                print("-----------------------------------------")

                if epoch % config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        self.model.state_dict(), "epoch_{}.ckpt".format(epoch)
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
            config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        
        rearrangement_dataset = RearrangementEpisodeDataset(config)
        batch_size = config.IL.BehaviorCloning.batch_size

        train_loader = DataLoader(
            rearrangement_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        logger.info(
            "[ val_loader has {} samples ]".format(
                len(rearrangement_dataset)
            )
        )

        instruction_vocab_dict = rearrangement_dataset.get_vocab_dict()
        action_space = self.envs.action_spaces[0]

        model_kwargs = {
            "num_actions": action_space.n,
            "instruction_vocab": instruction_vocab_dict.word2idx_dict,
            "freeze_encoder": config.IL.BehaviorCloning.freeze_encoder,
        }

        self.model = self._setup_model(
            self.envs.observation_spaces[0],
            action_space,
            config.MODEL
        )

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt_dict)
        self.model.eval()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        prev_actions = torch.zeros(
            1, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.ones(
            1, 1, device=self.device
        )
        rnn_hidden_states = torch.zeros(
            1,
            self.envs.num_envs,
            config.MODEL.STATE_ENCODER.hidden_size,
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
        softmax = torch.nn.Softmax(dim=1)
        count = 0
        prev_actions[0][0] = HabitatSimActions.START

        current_episodes = self.envs.current_episodes()
        reference_replay = current_episodes[0].reference_replay
        gt_batch = next(iter(train_loader))
        mismatch_actions = 0
        mismatch_ids = []
        gt_actions = []
        pred_actions = []

        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0 and count < len(reference_replay)
        ):
            current_episodes = self.envs.current_episodes()

            inpute_data = gt_batch
            episode_done = gt_batch[7]
            gt_next_action = gt_batch[5]
            gt_prev_action = gt_batch[6]

            not_done_masks[0][0] = episode_done[0][count]

            with torch.no_grad():
                (
                    distribution,
                    rnn_hidden_states
                ) = self.model(
                    batch,
                    rnn_hidden_states,
                    prev_actions,
                    not_done_masks
                )

                rnn_hidden_states = rnn_hidden_states.detach()

                logits = distribution.logits
                actions = torch.argmax(softmax(logits), dim=1)
                prev_actions.copy_(actions.unsqueeze(1))

            gt_action = HabitatSimActions.STOP
            if count != len(reference_replay) - 1:
                gt_action = get_habitat_sim_action(reference_replay[count + 1]["action"])

            gt_actions.append(get_habitat_sim_action_str(gt_action))
            pred_actions.append(get_habitat_sim_action_str(actions[0].item()))
            if not (gt_action == actions[0].item()):

                print("prev action: ", prev_actions, gt_prev_action[0][count])
                print("actual action: ", gt_action, gt_next_action[0][count])

                print("gt_action: {}, pred action: {}".format(gt_action, actions))
                print("match: {}".format(gt_action == actions[0].item()))
                print("\n")
                mismatch_actions += 1
                mismatch_ids.append(count)
            else:
                print("gt_action: {}, pred action: {}".format(gt_action, actions[0].item()))

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]
            # step_data = [gt_action]

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
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
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
                        {"rgb": batch["rgb"].squeeze(0)}, infos[i]
                    )
                    rgb_frames[i].append(frame)
            (
                self.envs,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )
            count += 1
        print("\n\n\n")
        print("Total actions: {}".format(len(reference_replay)))
        print("Mismatch actions: {}".format(mismatch_actions))
        print("Mismatch index: {}".format(mismatch_ids))

        with open("predictions.csv", "w") as f:
            f.write("ground_truth,prediction,match\n")
            for i in range(len(gt_actions)):
                f.write("{},{},{}\n".format(gt_actions[i], pred_actions[i], (gt_actions[i] == pred_actions[i])))
        self.envs.close()

