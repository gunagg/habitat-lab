import json
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.objectnav.models.seq_2_seq_model import Seq2SeqModel
from habitat_baselines.objectnav.models.sem_seg_model import SemSegSeqModel
from habitat_baselines.objectnav.models.single_resnet_model import SingleResNetSeqModel

class BaseObjectNavTrainer(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0
    
    def _setup_policy(self, observation_space, action_space, model_config, device):
        model_config.defrost()
        model_config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        model_config.freeze()

        model = None
        if hasattr(model_config, "VISUAL_ENCODER"):
            model = SingleResNetSeqModel(observation_space, action_space, model_config, device)
        elif model_config.USE_SEMANTICS:
            model = SemSegSeqModel(observation_space, action_space, model_config, device)
        else:
            model = Seq2SeqModel(observation_space, action_space, model_config)
        return model   

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        num_episodes: int,
    ) -> None:
        self.policy = self._setup_policy(
            observation_space=observation_space,
            action_space=action_space,
            model_config=self.config.MODEL,
            device=self.device,
        )
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.IL.BehaviorCloning.lr
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.IL.BehaviorCloning.lr,
            steps_per_epoch=num_episodes, epochs=config.IL.BehaviorCloning.max_epochs
        )
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    def save_checkpoint(self, file_name) -> None:
        r"""Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        rnn_hidden_states = torch.zeros(
            self.policy.net.num_recurrent_layers,
            N,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        
        self.optimizer.zero_grad()
        num_samples = corrected_actions.shape[0]
        timestep_batch_size = self.config.IL.BehaviorCloning.timestep_batch_size
        num_steps = num_samples // timestep_batch_size + (num_samples % timestep_batch_size != 0)
        batch_loss = 0
        for i in range(num_steps):
            slice_start_time = time.time()
            start_idx = i * timestep_batch_size
            end_idx = start_idx + timestep_batch_size
            observations_batch_sample = {
                k: v[start_idx:end_idx].to(device=self.device)
                for k, v in observations.items()
            }

            gt_next_action_sample = corrected_actions[start_idx:end_idx].long().to(self.device)
            gt_prev_action_sample = prev_actions[start_idx:end_idx].long().to(self.device)
            episode_not_dones_sample = not_done_masks[start_idx:end_idx].long().to(self.device)
            inflec_weights_sample = weights[start_idx:end_idx].long().to(self.device)

            logits, rnn_hidden_states = self.policy(
                observations_batch_sample,
                rnn_hidden_states,
                gt_prev_action_sample,
                episode_not_dones_sample
            )

            T, N = gt_next_action_sample.shape
            logits = logits.view(T, N, -1)

            action_loss = self.cross_entropy_loss(logits.permute(0, 2, 1), gt_next_action_sample)
            denom = inflec_weights_sample.sum(0)
            denom[denom == 0.0] = 1
            action_loss = ((inflec_weights_sample * action_loss).sum(0) / denom).mean()
            loss = (action_loss / num_steps)
            loss.backward()
            batch_loss += loss.item()
            rnn_hidden_states = rnn_hidden_states.detach()


        if step_grad:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return batch_loss #loss.item()

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
        current_episode_reward=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[:, state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]
            current_episode_reward = current_episode_reward[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
            current_episode_reward,
        )
    
