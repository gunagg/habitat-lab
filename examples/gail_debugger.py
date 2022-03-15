import argparse
import copy
import cv2
import habitat
import json
import sys
import time
import os
import torch
import numpy as np
import torch.nn.functional as F

from habitat import Config
from habitat.core.simulator import Observations
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image, images_to_video, append_text_to_image
from habitat_sim.utils.common import quat_to_coeffs, quat_from_magnum, quat_from_angle_axis

from habitat_baselines.config.default import get_config
from habitat_baselines.objectnav.il.algos.gail import GAIL

from threading import Thread
from time import sleep

from PIL import Image

from habitat_baselines.common.baseline_registry import baseline_registry

config = habitat.get_config("configs/tasks/objectnav_mp3d_gail.yaml")


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def get_habitat_sim_action(data):
    if data.action == "turnRight":
        return HabitatSimActions.TURN_RIGHT
    elif data.action == "turnLeft":
        return HabitatSimActions.TURN_LEFT
    elif data.action == "moveForward":
        return HabitatSimActions.MOVE_FORWARD
    elif data.action == "lookUp":
        return HabitatSimActions.LOOK_UP
    elif data.action == "lookDown":
        return HabitatSimActions.LOOK_DOWN
    return HabitatSimActions.STOP


def log_action_data(data, i):
    if data.action != "stepPhysics":
        print("Action: {} - {}".format(data.action, i))
    else:
        print("Action {} - {}".format(data.action, i))


def get_goal_visible_area(observations, task_cat2mpcat40, episode):
    obj_semantic = observations["predicted_sem_obs"].flatten(start_dim=1)
    task_cat2mpcat40 = torch.tensor(task_cat2mpcat40)
    idx = task_cat2mpcat40[
        torch.Tensor(observations["objectgoal"]).long()
    ]
    idx = idx.to(obj_semantic.device)

    goal_visible_pixels = (obj_semantic == idx).sum(dim=1) # Sum over all since we're not batched
    goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1))

    obj_semantic = observations["semantic"].flatten(start_dim=1)
    idx = task_cat2mpcat40[
        torch.Tensor(observations["objectgoal"]).long()
    ]
    idx = idx.to(obj_semantic.device)

    goal_visible_pixels = (obj_semantic == idx).sum(dim=1) # Sum over all since we're not batched
    goal_visible_area_gt = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1))
    return goal_visible_area, goal_visible_area_gt


def setup_model(model_config, observation_space, action_space, device, checkpoint_path):
    # model_config = get_config("habitat_baselines/config/objectnav/gail_objectnav.yaml")
    discriminator = baseline_registry.get_discriminator(model_config.IL.GAIL.DISCRIMINATOR.name)
    discriminator = discriminator.from_config(
        model_config, observation_space, action_space
    )

    gail = GAIL(
        discriminator=discriminator,
        num_envs=1,
        num_mini_batch=model_config.IL.GAIL.num_mini_batch,
        lr=model_config.IL.GAIL.lr,
        eps=model_config.IL.GAIL.eps,
        max_grad_norm=model_config.IL.GAIL.max_grad_norm,
        use_normalized_advantage=model_config.IL.GAIL.use_normalized_advantage,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    gail.load_state_dict(ckpt["reward_model_state_dict"], strict=True)
    print("Loading pretrained weights from : {}".format(checkpoint_path))

    gail.to(device)
    return gail


def parse_gt_semantic(sim, semantic_obs):
    # obtain mapping from instance id to semantic label id
    scene = sim.semantic_annotations()

    instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
    mapping = np.array([ instance_id_to_label_id[i] for i in range(len(instance_id_to_label_id)) ])

    # ! MP3D object id to category ID mapping
    # Should raise a warning, but just driving ahead for now
    if mapping.size > 0:
        semantic_obs = np.take(mapping, semantic_obs)
    else:
        semantic_obs = semantic_obs.astype(int)
    return semantic_obs.astype(int)


def get_goal_semantic(sem_obs, object_oal, task_cat2mpcat40, episode):
    obj_semantic = sem_obs
    task_cat2mpcat40 = torch.tensor(task_cat2mpcat40)
    idx = task_cat2mpcat40[
        torch.Tensor(object_oal).long()
    ]
    idx = idx.to(obj_semantic.device)

    goal_visible_pixels = (obj_semantic == idx) # Sum over all since we're not batched
    return goal_visible_pixels.long()


def get_coverage(info):
    top_down_map = info["map"]
    visted_points = np.where(top_down_map <= 9, 0, 1)
    coverage = np.sum(visted_points) / get_navigable_area(info)
    return coverage


def get_navigable_area(info):
    top_down_map = info["map"]
    navigable_area = np.where(((top_down_map == 1) | (top_down_map >= 10)), 1, 0)
    return np.sum(navigable_area)


def get_visible_area(info):
    fog_of_war_mask = info["fog_of_war_mask"]
    visible_area = fog_of_war_mask.sum() / get_navigable_area(info)
    if visible_area > 1.0:
        visible_area = 1.0
    return visible_area


def save_top_down_map(info):
    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], 512
    )
    save_image(top_down_map, "top_down_map.png")


def run_reference_replay(
    cfg, log_action=False, num_episodes=None, output_prefix=None, compute_reward=False, checkpoint_path=""
):
    instructions = []
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    with habitat.Env(cfg) as env:
        obs_list = []
        success = 0
        spl = 0
        avg_ep_length = 0

        num_episodes = 0
        print("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            observation_list = []
            top_down_obs_list = []
            obs = env.reset()

            model_config = get_config("habitat_baselines/config/objectnav/gail_objectnav.yaml")
            model_config.defrost()
            model_config.SENSORS = ["RGB_SENSOR"]
            model_config.NUM_PROCESSES = 1
            model_config.freeze()
            
            gail = setup_model(model_config, env.observation_space, env.action_space, device, checkpoint_path)
            rnn_hidden_states = torch.zeros((1, model_config.RL.DDPPO.num_recurrent_layers, model_config.IL.GAIL.hidden_size))

            print('Scene has physiscs {}'.format(cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS))
            physics_simulation_library = env._sim.get_physics_simulation_library()
            print("Physics simulation library: {}".format(physics_simulation_library))
            print("Episode length: {}, Episode index: {}".format(len(env.current_episode.reference_replay), ep_id))
            print("Scene Id : {}, Episode Id: {}".format(env.current_episode.scene_id, env.current_episode.episode_id))
            i = 0
            step_index = 1
            total_reward = 0.0
            episode = env.current_episode
            ep_success = 0
            prev_action = 0

            for data in env.current_episode.reference_replay[step_index:]:
                if log_action:
                    log_action_data(data, i)
                
                if env.episode_over:
                    break
                # action = possible_actions.index(data.action)
                # action = possible_actions.index("MOVE_FORWARD")
                action = possible_actions.index("TURN_LEFT")
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)

                batch = {
                    "rgb": torch.from_numpy(observations["rgb"]).unsqueeze(0).to(device),
                    "objectgoal": torch.from_numpy(observations["objectgoal"]).to(device)
                }
                print(batch["rgb"].shape, rnn_hidden_states.shape)

                reward, rnn_hidden_states = gail.discriminator.compute_rewards(
                    batch,
                    rnn_hidden_states,
                    torch.tensor([prev_action]).unsqueeze(0).to(device),
                    torch.tensor([1]).unsqueeze(0).bool().to(device)
                )
                prev_action = action
                print("step: {}, action: {}, reward: {}".format(i, action_name, reward.exp().detach()))
                rnn_hidden_states = rnn_hidden_states.detach()

                info = env.get_metrics()
                print("Step: {}, action: {}".format(i, action_name))
                
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                if info["success"]:
                    ep_success = 1

                observation_list.append(frame)
                i+=1

            make_videos([observation_list], output_prefix, ep_id)
            print("Total reward for trajectory: {} - {}".format(total_reward, ep_success))
            
            success += ep_success
            spl += info["spl"]

            num_episodes += 1
            avg_ep_length += len(episode.reference_replay)


        print("Total episode success: {}".format(success))
        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(success/num_episodes, success, num_episodes))
        print("Average episode length: {}, {}, {}".format(avg_ep_length/num_episodes, avg_ep_length, num_episodes))

        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="demo"
    )
    parser.add_argument(
        "--log-action", dest='log_action', action='store_true'
    )
    parser.add_argument(
        "--compute-reward", dest='compute_reward', action='store_true'
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default=""
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.episodes
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.freeze()

    observations = run_reference_replay(
        cfg, args.log_action,
        num_episodes=1, output_prefix=args.output_prefix, compute_reward=args.compute_reward,
        checkpoint_path=args.checkpoint_path
    )

if __name__ == "__main__":
    main()
