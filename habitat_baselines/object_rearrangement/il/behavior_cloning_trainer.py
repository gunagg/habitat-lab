#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
import sys
import torch
import tqdm
from torch.utils.data import DataLoader

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.object_rearrangement.data.dataset import RearrangementDataset
from habitat_baselines.object_rearrangement.il.models.model import MultitaskCNN


@baseline_registry.register_trainer(name="rearrangement-behavior-cloning")
class RearrangementBCTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["RearrangementTask-v0"]

    def __init__(self, config=None):
        super().__init__(config)

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

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        config = self.config
        
        rearrangement_dataset = RearrangementDataset(config)

        train_loader = DataLoader(
            rearrangement_dataset,
            batch_size=config.IL.BehaviorCloning.batch_size,
            shuffle=False,
        )

        logger.info(
            "[ train_loader has {} samples ]".format(
                len(rearrangement_dataset)
            )
        )

        model = MultitaskCNN(num_actions=10)
        model.train().to(self.device)

        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.IL.BehaviorCloning.lr),
        )

        action_loss = torch.nn.CrossEntropyLoss()

        epoch, t = 1, 0
        with TensorboardWriter(
            config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while epoch <= config.IL.BehaviorCloning.max_epochs:
                start_time = time.time()
                avg_loss = 0.0

                for batch in train_loader:
                    t += 1

                    idx, obs_rgb, obs_depth, obs_seg, obs_instruction_tokens, gt_action = batch

                    optim.zero_grad()

                    obs_rgb = obs_rgb.to(self.device)
                    obs_depth = obs_depth.to(self.device)
                    obs_seg = obs_seg.to(self.device)
                    obs_instruction_tokens = obs_instruction_tokens.to(self.device)
                    gt_action = gt_action.long().to(self.device)

                    pred_action = model(obs_rgb)

                    loss = action_loss(pred_action, gt_action)

                    avg_loss += loss.item()

                    if t % config.LOG_INTERVAL == 0:
                        logger.info(
                            "[ Epoch: {}; iter: {}; loss: {:.3f} ]".format(
                                epoch, t, loss.item()
                            )
                        )

                        writer.add_scalar("total_loss", loss, t)

                    loss.backward()
                    optim.step()

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
                        model.state_dict(), "epoch_{}.ckpt".format(epoch)
                    )

                epoch += 1

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

        rearrangement_dataset = RearrangementDataset(config, mode="val")

        eval_loader = DataLoader(
            rearrangement_dataset,
            batch_size=config.IL.BehaviorCloning.batch_size,
            shuffle=False,
        )

        logger.info(
            "[ eval_loader has {} samples ]".format(
                len(rearrangement_dataset)
            )
        )

        model = MultitaskCNN()

        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

        model.to(self.device).eval()

        action_loss = torch.nn.CrossEntropyLoss()

        t = 0
        avg_loss = 0.0

        with torch.no_grad():
            for batch in eval_loader:
                t += 1

                idx, obs_rgb, obs_depth, obs_seg, obs_instruction_tokens, gt_action = batch
                obs_rgb = obs_rgb.to(self.device)
                obs_depth = obs_depth.to(self.device)
                obs_seg = obs_seg.to(self.device)
                obs_instruction_tokens = obs_instruction_tokens.to(self.device)
                gt_action = gt_action.long().to(self.device)

                pred_action = model(obs_rgb)

                loss = action_loss(pred_action, gt_action)

                avg_loss += loss.item()

                if t % config.LOG_INTERVAL == 0:
                    logger.info(
                        "[ Iter: {}; loss: {:.3f} ]".format(t, loss.item()),
                    )

                if (
                    config.EVAL_SAVE_RESULTS
                    and t % config.EVAL_SAVE_RESULTS_INTERVAL == 0
                ):

                    result_id = "ckpt_{}_{}".format(
                        checkpoint_index, idx[0].item()
                    )
                    result_path = os.path.join(
                        self.config.RESULTS_DIR, result_id
                    )

                    self._save_results(
                        gt_rgb,
                        pred_rgb,
                        gt_seg,
                        pred_seg,
                        gt_depth,
                        pred_depth,
                        result_path,
                    )

        avg_loss /= len(eval_loader)

        writer.add_scalar("avg val total loss", avg_loss, checkpoint_index)

        logger.info("[ Average loss: {:.3f} ]".format(avg_loss))
