import os
import time
import warnings
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from habitat_baselines.objectnav.il.base_il_trainer import BaseObjectNavTrainer
from habitat_baselines.objectnav.dataset.recollection_dataset import (
    TeacherRecollectionDataset,
)
from habitat_baselines.objectnav.dataset.episode_dataset import collate_fn
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
)


@baseline_registry.register_trainer(name="recollect_trainer")
class RecollectTrainer(BaseObjectNavTrainer):
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

    def train(self) -> None:
        self.config.defrost()
        self.config.use_pbar = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.freeze()

        dataset = TeacherRecollectionDataset(self.config)
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

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=dataset.observation_space,
            action_space=dataset.action_space,
            num_episodes=dataset.length,
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

            batches_per_epoch = dataset.length // dataset.batch_size

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
                    # logger.info(prev_actions_batch)
                    # logger.info(corrected_actions_batch)
                    # logger.info(not_done_masks)

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

                    if self.config.use_pbar:
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
                    writer.add_scalar("loss", loss, self.step_id)
                    self.step_id += 1  # noqa: SIM113

                if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(epoch, epoch)

            dataset.close_sims()

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
        config.TASK_CONFIG.DATASET.TYPE = "ObjectNav-v1"
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            if config.SHOW_TOP_DOWN_MAP:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        batch_size = config.IL.BehaviorCloning.batch_size

        logger.info(
            "[ val_loader has {} samples ]".format(
                self.envs.count_episodes()
            )
        )

        action_space = self.envs.action_spaces[0]

        self.policy = self._setup_policy(
            self.envs.observation_spaces[0],
            action_space,
            config.MODEL,
            self.device
        )

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(ckpt_dict["state_dict"], strict=True)
        self.policy.to(self.device)
        self.policy.eval()

        self.semantic_predictor = None
        if config.MODEL.USE_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt=config.MODEL.SEMANTIC_ENCODER.rednet_ckpt,
                resize=True # since we train on half-vision
            )
            self.semantic_predictor.eval()

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

        self._make_results_dir(config.EVAL.SPLIT)

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
        possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        episode_count = 0
        success_episode_data = []
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                if self.semantic_predictor is not None:
                    batch["pred_semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])

                if config.MODEL.USE_PRED_SEMANTICS:
                    batch["semantic"] = batch["pred_semantic"].clone()

                (
                    logits,
                    rnn_hidden_states
                ) = self.policy(
                    batch,
                    rnn_hidden_states,
                    prev_actions,
                    not_done_masks
                )

                actions = torch.argmax(logits, dim=1)
                prev_actions.copy_(actions.unsqueeze(1))

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
                    next_episodes = self.envs.current_episodes()
                    episode_count += 1

                    if episode_stats["success"]:
                        success_episode_data.append({
                            "episode_id": current_episodes[i].episode_id,
                            "episode_count": episode_count,
                            "metrics": self._extract_scalars_from_info(infos[i])
                        })

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

                        # self._save_results(
                        #     batch,
                        #     infos,
                        #     config.RESULTS_DIR,
                        #     i,
                        #     config.EVAL.SPLIT,
                        #     current_episodes[i].episode_id,
                        # )

                        rgb_frames[i] = []
                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    # frame = append_text_to_image(frame, "Find: {}".format(current_episodes[i].object_category))
                    # frame = append_text_to_image(frame, "Action: {}".format(action_names[i]))
                    rgb_frames[i].append(frame)


            (
                self.envs,
                rnn_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
                current_episode_reward,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                rnn_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
                current_episode_reward,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        output_dir = config.EVAL_RESUTLS_DIR
        with open(os.path.join(output_dir, "{}_success_episodes.json".format(config.EVAL.SPLIT)), "w") as f:
            f.write(json.dumps(success_episode_data))

        self.envs.close()

        print ("environments closed")

