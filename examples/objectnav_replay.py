import argparse
import cv2
import habitat
import json
import sys
import time
import os

from habitat import Config
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
from habitat.utils.visualizations.utils import make_video_cv2, observations_to_image, images_to_video, append_text_to_image

from threading import Thread
from time import sleep

from PIL import Image

config = habitat.get_config("configs/tasks/objectnav_mp3d_video.yaml")


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


def run_reference_replay(cfg, step_env=False, log_action=False, num_episodes=None, output_prefix=None):
    instructions = []
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        obs_list = []
        success = 0
        spl = 0

        if num_episodes is None:
            num_episodes = len(env.episodes)

        print("Total episodes: {}".format(len(env.episodes)))
        fails = []
        for ep_id in range(len(env.episodes)):
            observation_list = []
            print("before reset")
            obs = env.reset()
            print("after reset")

            print('Scene has physiscs {}'.format(cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS))
            physics_simulation_library = env._sim.get_physics_simulation_library()
            print("Physics simulation library: {}".format(physics_simulation_library))
            print("Episode length: {}, Episode index: {}".format(len(env.current_episode.reference_replay), ep_id))
            print("Scene Id : {}".format(env.current_episode.scene_id))
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
            for data in env.current_episode.reference_replay[step_index:]:
                if log_action:
                    log_action_data(data, i)
                action = possible_actions.index(data.action)
                action_name = env.task.get_action_name(
                    action
                )

                if step_env:
                    observations = env.step(action=action)

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, info)
                depth_frame = observations_to_image({"depth": observations["depth"]}, {})

                # frame = append_text_to_image(frame, "Find and go to {}".format(episode.object_category))

                if info["success"]:
                    ep_success = 1

                observation_list.append(frame)

                i+=1
            make_videos([observation_list], output_prefix, ep_id)
            print("Total reward for trajectory: {} - {}".format(total_reward, grab_count))
            success += ep_success
            spl += info["spl"]
            if ep_success == 0:
                fails.append({
                    "episodeId": instructions[-1]["episodeId"],
                    "distanceToGoal": info["distance_to_goal"]
                })

        print("Total episode success: {}".format(success))
        print("SPL: {}, {}, {}".format(spl/num_episodes, spl, num_episodes))
        print("Failed episodes: {}".format(fails))
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
    cfg.freeze()

    observations = run_reference_replay(
        cfg, args.step_env, args.log_action,
        num_episodes=1, output_prefix=args.output_prefix
    )

if __name__ == "__main__":
    main()
