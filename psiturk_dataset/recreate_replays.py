import argparse
import cv2
import gzip
import habitat
import json
import sys
import time

from habitat import Config
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
from habitat.utils.visualizations.utils import make_video_cv2

from threading import Thread
from time import sleep

config = habitat.get_config("configs/tasks/object_rearrangement.yaml")


def make_videos(observations_list, output_prefix):
    for idx in range(len(observations_list)):
        prefix = output_prefix + "_{}".format(idx)
        make_video_cv2(observations_list[idx], prefix=prefix)


def write_gzip(input_path, output_path):
    with open(input_path, "rb") as input_file:
        with gzip.open(output_path + ".gz", "wb") as output_file:
            output_file.writelines(input_file)


def save_episode_replay(episodes, output_path):
    episode_data = {
        "episodes": []
    }
    for episode, reference_replay in episodes:
        data = {}
        data["episode_id"] = episode.episode_id
        data["scene_id"] = episode.scene_id.split("/")[-1]
        data["start_position"] = episode.start_position
        data["start_rotation"] = episode.start_rotation
        data["objects"] = episode.objects
        data["instruction"] = {
            "instruction_text": episode.instruction.instruction_text
        }
        data["goals"] = episode.goals
        data["reference_replay"] = reference_replay
        episode_data["episodes"].append(data)
        
    with open(output_path, "w") as f:
        f.write(json.dumps(episode_data))
    write_gzip(output_path, output_path)


def get_habitat_sim_action(data):
    if data["action"] == "turnRight":
        return HabitatSimActions.TURN_RIGHT
    elif data["action"] == "turnLeft":
        return HabitatSimActions.TURN_LEFT
    elif data["action"] == "moveForward":
        return HabitatSimActions.MOVE_FORWARD
    elif data["action"] == "moveBackward":
        return HabitatSimActions.MOVE_BACKWARD
    elif data["action"] == "lookUp":
        return HabitatSimActions.LOOK_UP
    elif data["action"] == "lookDown":
        return HabitatSimActions.LOOK_DOWN
    elif data["action"] == "grabReleaseObject":
        return HabitatSimActions.GRAB_RELEASE
    elif data["action"] == "stepPhysics":
        return HabitatSimActions.NO_OP
    return HabitatSimActions.STOP


def rebuild_episode_replay(cfg, output_path, num_episodes=None):
    instructions = []
    with habitat.Env(cfg) as env:
        obs_list = []

        if num_episodes is None:
            num_episodes = len(env.episodes)

        episodes = []
        print("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            observation_list = []

            env._sim.update_cross_hair()
            obs = env.reset()
            observation_list.append(obs)

            print('Scene has physiscs {}'.format(cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS))
            physics_simulation_library = env._sim.get_physics_simulation_library()
            print("Physics simulation library: {}".format(physics_simulation_library))
            print("Episode length: {}".format(len(env.current_episode.reference_replay)))
            i = 0
            data = {
                "episodeId": env.current_episode.episode_id,
                "video": "demo_{}.mp4".format(ep_id),
                "task": env.current_episode.instruction.instruction_text
            }
            instructions.append(data)
            reference_replay = []
            episode = env.current_episode
            for data in env.current_episode.reference_replay:
                action = get_habitat_sim_action(data)
                observations = env.step(action=action, replay_data=data)
                observation_list.append(observations)

                agent_state = env._sim.get_agent_pose()
                object_states = env._sim.get_current_object_states()
                data["agent_state"] = agent_state
                data["object_states"] = object_states
                reference_replay.append(data)
                i+=1
            episodes.append((episode, reference_replay))
            obs_list.append(observation_list)
        inst_file = open("instructions.json", "w")
        inst_file.write(json.dumps(instructions))
        
        save_episode_replay(episodes, output_path)
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
        "--output-path", type=str, default="task_1.json"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.replay_episode
    cfg.freeze()
    
    observations = rebuild_episode_replay(cfg, output_path=args.output_path, num_episodes=1)
    make_videos(observations, args.output_prefix)

if __name__ == "__main__":
    main()
