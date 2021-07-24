#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.base_trainer import BaseRLTrainer, BaseTrainer
from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
    EQACNNPretrainTrainer,
)
from habitat_baselines.il.trainers.vqa_trainer import VQATrainer
from habitat_baselines.rl.ddppo import DDPPOTrainer  # noqa: F401
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer, RolloutStorage
from habitat_baselines.rearrangement.il.behavior_cloning_trainer import RearrangementBCTrainer
from habitat_baselines.rearrangement.il.behavior_cloning_distrib_trainer import RearrangementBCDistribTrainer
from habitat_baselines.rearrangement.il.behavior_cloning_env_trainer import RearrangementBCEnvTrainer
from habitat_baselines.rearrangement.rl.ppo_trainer import RearrangementPPOTrainer
from habitat_baselines.rearrangement.rl.ddppo_trainer import RearrangementDDPPOTrainer
from habitat_baselines.rearrangement.reward_modeling.ppo_agile_trainer import RearrangementPPOAgileTrainer
from habitat_baselines.rearrangement.reward_modeling.ddppo_agile_trainer import RearrangementDDPPOAgileTrainer
from habitat_baselines.objectnav.il.behavior_cloning_trainer import ObjectNavBCTrainer
from habitat_baselines.objectnav.il.behavior_cloning_distrib_trainer import ObjectNavDistribBCTrainer
from habitat_baselines.objectnav.il.behavior_cloning_env_trainer import ObjectNavBCEnvTrainer
from habitat_baselines.objectnav.il.behavior_cloning_env_ddp_trainer import ObjectNavBCEnvDDPTrainer
from habitat_baselines.objectnav.il.recollect_trainer import RecollectTrainer
from habitat_baselines.objectnav.il.recollect_ddp_trainer import RecollectDDPTrainer


__all__ = [
    "BaseTrainer",
    "BaseRLTrainer",
    "BaseILTrainer",
    "PPOTrainer",
    "RolloutStorage",
    "EQACNNPretrainTrainer",
    "VQATrainer",
    "RearrangementBCTrainer",
    "RearrangementBCDistribTrainer",
    "RearrangementPPOTrainer",
    "RearrangementDDPPOTrainer",
    "RearrangementPPOAgileTrainer",
    "RearrangementDDPPOAgileTrainer",
    "RearrangementBCEnvTrainer",
    "ObjectNavBCTrainer",
    "ObjectNavDistribBCTrainer",
    "ObjectNavBCEnvTrainer",
    "ObjectNavBCEnvDDPTrainer",
    "RecollectTrainer",
    "RecollectDDPTrainer",
]
