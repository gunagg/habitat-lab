import os
import random
import time
import warnings
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import torch
import tqdm
from torch import distributed as distrib

from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from habitat_baselines.objectnav.il.base_il_trainer import BaseObjectNavTrainer
from habitat_baselines.objectnav.dataset.recollection_dataset import (
    TeacherRecollectionDataset,
)
from habitat_baselines.objectnav.il.recollect_trainer import (
    RecollectTrainer,
)
from habitat_baselines.objectnav.dataset.episode_dataset import collate_fn
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


@baseline_registry.register_trainer(name="recollect_ddp_trainer")
class RecollectDDPTrainer(RecollectTrainer):
    r"""A Teacher Forcing trainer that re-collects episodes from simulation
    rather than saving them all to disk. Included as starter code for the
    RxR-Habitat Challenge but can also train R2R agents.
    """
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
    
    def _make_results_dir(self, split="val"):
        r"""Makes directory for saving eqa-cnn-pretrain eval results."""
        for s_type in ["rgb", "seg", "depth", "top_down_map"]:
            dir_name = self.config.RESULTS_DIR.format(split=split, type=s_type)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(
            os.path.dirname(
                self.config.IL.BehaviorCloning.trajectories_file
            ),
            exist_ok=True,
        )
        # if self.config.EVAL_SAVE_RESULTS:
        #     self._make_results_dir()

    def save_checkpoint(self, epoch: int, step_id: int) -> None:
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "step_id": step_id,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{epoch}.pth"),
        )

    def _save_results(
        self,
        observations,
        infos,
        path: str,
        env_idx: int,
        split: str,
        episode_id: str
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
        rgb_frame = observations_to_image(
                        {}, infos[env_idx]
                    )
        rgb_path = os.path.join(path.format(split=split, type="rgb"), "frame_{}".format(episode_id))
        save_frame(rgb_frame, rgb_path)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "goal_vis_pixels", "rearrangement_reward", "coverage"}

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

        # Distributed data parallel setup
        if torch.cuda.is_available():
            self.policy = torch.nn.parallel.DistributedDataParallel(
                self.policy, device_ids=[self.device], output_device=self.device,
                find_unused_parameters = True
            )
        else:
            self.policy = torch.nn.parallel.DistributedDataParallel(
                self.policy, find_unused_parameters = True
            )

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
            self.policy.module.net.num_recurrent_layers,
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

            if i != num_steps - 1:
                with self.policy.no_sync():
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
            else:
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

        # Sync loss
        # stats = torch.tensor(
        #     [batch_loss],
        #     device=self.device,
        # )
        # distrib.all_reduce(stats)
        # batch_loss = stats[0].item()

        if step_grad:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return batch_loss

    def train(self) -> None:

        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.IL.distrib_backend
        )
        add_signal_handlers()

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.world_rank = self.world_rank
        self.config.world_size = self.world_size
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.use_pbar = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        dataset = TeacherRecollectionDataset(self.config, is_ddp=True)
        diter = iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=1,
            )
        )
        num_episodes = self.config.IL.BehaviorCloning.num_batches_per_epoch * dataset.batch_size

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=dataset.observation_space,
            action_space=dataset.action_space,
            num_episodes=num_episodes,
        )

        if self.config.IL.BehaviorCloning.effective_batch_size > 0:
            assert (
                self.config.IL.BehaviorCloning.effective_batch_size
                % self.config.IL.BehaviorCloning.batch_size
                == 0
            ), (
                "Gradient accumulation: effective_batch_size"
                " should be a multiple of batch_size."
            )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:

            #  batches_per_epoch = dataset.length // dataset.batch_size
            batches_per_epoch = self.config.IL.BehaviorCloning.num_batches_per_epoch

            for epoch in range(self.start_epoch, self.config.IL.BehaviorCloning.max_epochs):
                epoch_time = time.time()
                epoch_str = f"{epoch + 1}/{self.config.IL.BehaviorCloning.max_epochs}"

                t = (
                    tqdm.trange(
                        batches_per_epoch, leave=False, dynamic_ncols=True
                    )
                    if self.config.use_pbar
                    else range(batches_per_epoch)
                )
                avg_loss = 0

                for batch_idx in t:
                    batch_time = time.time()
                    batch_str = f"{batch_idx + 1}/{batches_per_epoch}"

                    (
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch,
                        weights_batch,
                    ) = next(diter)

                    observations_batch = {
                        k: v.to(device=self.device, non_blocking=True)
                        for k, v in observations_batch.items()
                    }

                    prev_actions_batch = prev_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    not_done_masks = not_done_masks.to(
                        device=self.device, non_blocking=True
                    )
                    corrected_actions_batch = corrected_actions_batch.to(
                        device=self.device, non_blocking=True
                    )
                    weights_batch = weights_batch.to(
                        device=self.device, non_blocking=True
                    )

                    # gradient accumulation
                    if (
                        self.config.IL.BehaviorCloning.effective_batch_size
                        > 0
                    ):
                        loss_accumulation_scalar = (
                            self.config.IL.BehaviorCloning.effective_batch_size
                            // self.config.IL.BehaviorCloning.batch_size
                        )
                        step_grad = bool(
                            self.step_id % loss_accumulation_scalar
                        )
                    else:
                        loss_accumulation_scalar = 1
                        step_grad = True

                    loss = self._update_agent(
                        observations_batch,
                        prev_actions_batch,
                        not_done_masks,
                        corrected_actions_batch,
                        weights_batch,
                        step_grad=step_grad,
                        loss_accumulation_scalar=loss_accumulation_scalar,
                    )

                    if self.config.use_pbar and self.world_rank == 0:
                        t.set_postfix(
                            {
                                "Epoch": epoch_str,
                                "Loss": round(loss, 4),
                            }
                        )
                    elif self.step_id % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            f"[Epoch: {epoch_str}] [Batch: {batch_str}]"
                            + f" [BatchTime: {round(time.time() - batch_time, 2)}s]"
                            + f" [EpochTime: {round(time.time() - epoch_time)}s]"
                            + f" [Loss: {round(loss, 4)}]"
                        )
                    if self.world_rank == 0:
                        writer.add_scalar("loss", loss, self.step_id)
                    self.step_id += 1  # noqa: SIM113
                    avg_loss += loss

                avg_loss = avg_loss / batches_per_epoch
                stats = torch.tensor(
                    [avg_loss],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                avg_loss = stats[0].item()
                if self.world_rank == 0:
                    writer.add_scalar("avg_loss", avg_loss / self.world_size, epoch)
                    logger.info(
                        f"[Epoch: {epoch_str}] [Avg loss: {avg_loss / self.world_size}]"
                    )
                    if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(epoch, epoch)

            dataset.close_sims()
