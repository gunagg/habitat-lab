import argparse
import cv2
import habitat
import json
import sys
import time
import os
import torch
import numpy as np

from habitat import Config
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import make_video_cv2, observations_to_image, images_to_video, append_text_to_image
from habitat_sim.utils.common import quat_to_coeffs, quat_from_magnum, quat_from_angle_axis

from threading import Thread
from time import sleep

from PIL import Image

from habitat_baselines.objectnav.models.rednet import load_rednet
from habitat_baselines.objectnav.models.sem_seg_model import SemSegSeqModel
from psiturk_dataset.utils.utils import write_json

config = habitat.get_config("configs/tasks/objectnav_mp3d_video.yaml")

task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    5,  # ('table', 4, 1)
    6,  # ('picture', 5, 2)
    7,  # ('cabinet', 6, 3)
    8,  # ('cushion', 7, 4)
    10,  # ('sofa', 9, 5),
    11,  # ('bed', 10, 6)
    13,  # ('chest_of_drawers', 12, 7),
    14,  # ('plant', 13, 8)
    15,  # ('sink', 14, 9)
    18,  # ('toilet', 17, 10),
    19,  # ('stool', 18, 11),
    20,  # ('towel', 19, 12)
    22,  # ('tv_monitor', 21, 13)
    23,  # ('shower', 22, 14)
    25,  # ('bathtub', 24, 15)
    26,  # ('counter', 25, 16),
    27,  # ('fireplace', 26, 17),
    33,  # ('gym_equipment', 32, 18),
    34,  # ('seating', 33, 19),
    38,  # ('clothes', 37, 20),
    43,  # ('foodstuff', 42, 21),
    44,  # ('stationery', 43, 22),
    45,  # ('fruit', 44, 23),
    46,  # ('plaything', 45, 24),
    47,  # ('hand_tool', 46, 25),
    48,  # ('game_equipment', 47, 26),
    49,  # ('kitchenware', 48, 27)
]


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def make_videos(observations_list, output_prefix, ep_id):
    #print(observations_list[0][0].keys(), type(observations_list[0][0]))
    prefix = output_prefix + "_{}".format(ep_id)
    # make_video_cv2(observations_list[0], prefix=prefix, open_vid=False)
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
    print(obj_semantic.shape, idx.shape)
    idx = idx.to(obj_semantic.device)

    goal_visible_pixels = (obj_semantic == idx).sum(dim=1) # Sum over all since we're not batched
    print(goal_visible_pixels.shape)
    print("\n\n")
    goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1))

    #obj_semantic = torch.Tensor(observations["semantic"].astype(np.uint8)).unsqueeze(0).flatten(start_dim=1)
    obj_semantic = observations["semantic"].flatten(start_dim=1)
    idx = task_cat2mpcat40[
        torch.Tensor(observations["objectgoal"]).long()
    ]
    # obj_goal = episode.goals[0]
    # idx = np.array([obj_goal.object_id], dtype=np.int64)
    # idx = torch.tensor(idx).long()
    idx = idx.to(obj_semantic.device)
    print(obj_semantic.shape, idx.shape)

    goal_visible_pixels = (obj_semantic == idx).sum(dim=1) # Sum over all since we're not batched
    print(goal_visible_pixels.shape)
    goal_visible_area_gt = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1))
    print(obj_semantic)
    print(idx, task_cat2mpcat40[
        torch.Tensor(observations["objectgoal"]).long()
    ])
    return goal_visible_area, goal_visible_area_gt

def setup_model(observation_space, action_space, device):
    model_config = habitat.get_config("habitat_baselines/config/objectnav/il_objectnav_sem_seg.yaml").MODEL
    model = SemSegSeqModel(observation_space, action_space, model_config, device)
    state_dict = torch.load("data/new_checkpoints/objectnav/sem_resnet18_challenge/seed_1/model_10.ckpt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


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


def get_agent_pose(sim):
    agent_translation = sim._default_agent.body.object.translation
    agent_rotation = sim._default_agent.body.object.rotation
    sensor_data = {}
    for sensor_key, v in sim._default_agent._sensors.items():
        rotation = quat_from_magnum(v.object.rotation)
        rotation = quat_to_coeffs(rotation).tolist()
        translation = v.object.translation
        sensor_data[sensor_key] = {
            "rotation": rotation,
            "translation": np.array(translation).tolist()
        }
    
    return {
        "position": np.array(agent_translation).tolist(),
        "rotation": quat_to_coeffs(quat_from_magnum(agent_rotation)).tolist(),
        "sensor_data": sensor_data
    }


def create_episode_trajectory(trajectory, episode):
    goal_positions = []
    for goal in episode.goals:
        for view_point in goal.view_points:
            goal_positions.append(view_point.agent_state.position)
    data = {
        "trajectory": trajectory,
        "object_goal": episode.object_category,
        "goal_locations": goal_positions,
        "scene_id": episode.scene_id.split("/")[-1]
    }
    return data


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
    return visible_area


def save_top_down_map(info):
    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], 512
    )
    save_image(top_down_map, "top_down_map.png")


def run_reference_replay(
    cfg, step_env=False, log_action=False, num_episodes=None, output_prefix=None, task_cat2mpcat40=None, sem_seg=False
):
    instructions = []
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    semantic_predictor = None
    if sem_seg:
        semantic_predictor = load_rednet(
                device,
                ckpt="data/rednet-models/rednet_semmap_mp3d_tuned.pth",
                # ckpt="data/rednet-models/rednet_semmap_mp3d_40.pth",
                resize=True # since we train on half-vision
            )
    
    with habitat.Env(cfg) as env:
        obs_list = []
        success = 0
        spl = 0
        total_coverage = 0
        visible_area = 0

        num_episodes = 0
        print("Total episodes: {}".format(len(env.episodes)))
        fails = []
        # scenes = []
        # for ep_id in range(len(env.episodes)):
        #     episode = env.episodes[ep_id]
        #     if ep_id in [1, 30, 43] or episode.episode_id in ["AOMFEAWQHU3D8:34S9DKFK75S93PUV2Q9SHUBWT7RYNA", "A34YDGVZKRJ0LZ:39U1BHVTDNU6IZ2RA12E0ZLB9E83TG", "A2JQPSIVUCW92T:39GAF6DQWT3PLOS1SSOADOUZ8GRV1F", "A3KC26Z78FBOJT:3NC5L260MQPLLJDCYFHH7Y4LD55FO6"]:
        #         episode = env.episodes[ep_id]
        #         scenes.append(episode.scene_id)
        # print("\n\n\n")
        # print(set(scenes))
        # sys.exit(1)
        for ep_id in range(len(env.episodes)):
            observation_list = []
            obs = env.reset()

            print('Scene has physiscs {}'.format(cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS))
            physics_simulation_library = env._sim.get_physics_simulation_library()
            print("Physics simulation library: {}".format(physics_simulation_library))
            print("Episode length: {}, Episode index: {}".format(len(env.current_episode.reference_replay), ep_id))
            print("Scene Id : {}, Episode Id: {}".format(env.current_episode.scene_id, env.current_episode.episode_id))
            i = 0
            data = {
                "episodeId": env.current_episode.episode_id,
                "sceneId": env.current_episode.scene_id,
                "video": "{}_{}.mp4".format(output_prefix, ep_id),
                "task": "Find and go to {}".format(env.current_episode.object_category),
                "episodeLength": len(env.current_episode.reference_replay)
            }
            instructions.append(data)
            step_index = 1
            grab_count = 0
            total_reward = 0.0
            episode = env.current_episode
            ep_success = 0
            replay_data = []

            if len(episode.reference_replay) > 2500:
                continue
            print("Object category: {}".format(env.current_episode.object_category))

            for data in env.current_episode.reference_replay[step_index:]:
                if log_action:
                    log_action_data(data, i)
                action = possible_actions.index(data.action)
                action_name = env.task.get_action_name(
                    action
                )

                observations = env.step(action=action)

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                if semantic_predictor is not None:
                    sem_obs = semantic_predictor(torch.Tensor(observations["rgb"]).unsqueeze(0).to(device), torch.Tensor(observations["depth"]).unsqueeze(0).to(device))
                    # sem_obs_2 = semantic_predictor_2(torch.Tensor(observations["rgb"]).unsqueeze(0).to(device), torch.Tensor(observations["depth"]).unsqueeze(0).to(device))
                    sem_obs_2 = parse_gt_semantic(env._sim, observations["semantic"])
                    sem_obs_2 = torch.Tensor(sem_obs_2).unsqueeze(0).long().to(device)
                    observations["predicted_sem_obs"] = sem_obs
                    observations["semantic"] = sem_obs_2
                    sem_obs_goal = get_goal_semantic(sem_obs, observations["objectgoal"], task_cat2mpcat40, episode)
                    sem_obs_gt_goal = get_goal_semantic(sem_obs_2, observations["objectgoal"], task_cat2mpcat40, episode)
                    frame = observations_to_image({"rgb": observations["rgb"], "semantic": sem_obs_goal, "gt_semantic": sem_obs_gt_goal}, info)                    

                frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))
                replay_data.append(get_agent_pose(env._sim))

                if info["success"]:
                    ep_success = 1

                observation_list.append(frame)
                if action_name == "STOP":
                    break
                i+=1
            # make_videos([observation_list], output_prefix, ep_id)
            print("Total reward for trajectory: {} - {}".format(total_reward, ep_success))
            if len(episode.reference_replay) <= 500:
                success += ep_success
                spl += info["spl"]

            # visible_area += get_visible_area(info["top_down_map"])
            # total_coverage += get_coverage(info["top_down_map"])
            num_episodes += 1

            if sem_seg:
                goal_visible_area, goal_visible_area_gt = get_goal_visible_area(observations, task_cat2mpcat40, episode)
                print("Goal visible area: {}, GT: {}".format(goal_visible_area, goal_visible_area_gt))

            # if ep_id >= 0: # in [0, 1, 9]:
            #     trajectory = create_episode_trajectory(replay_data, episode)
            #     print(trajectory.keys())
            #     write_json(trajectory, "data/visualizations/teaser/trajectories/{}_{}_custom_human_trajectory.json".format(trajectory["scene_id"].split(".")[0], ep_id))
            #     save_top_down_map(info)

            if ep_success == 0:
                fails.append({
                    "episodeId": instructions[-1]["episodeId"],
                    "distanceToGoal": info["distance_to_goal"]
                })

        print("Total episode success: {}".format(success))
        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Success: {}, {}, {}".format(success/num_episodes, success, num_episodes))
        print("Coverage: {}, {}, {}".format(total_coverage/num_episodes, total_coverage, num_episodes))
        print("Visible area: {}, {}, {}".format(visible_area/num_episodes, visible_area, num_episodes))
        print("Failed episodes: {}".format(fails))
        inst_file = open("instructions.json", "w")
        inst_file.write(json.dumps(instructions))
        return obs_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-episode", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="demo"
    )
    parser.add_argument(
        "--step-env", dest='step_env', action='store_true'
    )
    parser.add_argument(
        "--log-action", dest='log_action', action='store_true'
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.replay_episode
    # cfg.DATASET.CONTENT_SCENES = ["D7G3Y4RVNrH"]
    # cfg.DATASET.CONTENT_SCENES = ['i5noydFURQK']
    cfg.freeze()

    observations = run_reference_replay(
        cfg, args.step_env, args.log_action,
        num_episodes=1, output_prefix=args.output_prefix, task_cat2mpcat40=task_cat2mpcat40
    )

if __name__ == "__main__":
    main()
