import argparse
import cv2
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

from PIL import Image

config = habitat.get_config("configs/tasks/object_rearrangement.yaml")


def make_videos(observations_list, output_prefix):
    for idx in range(len(observations_list)):
        prefix = output_prefix + "_{}".format(idx)
        make_video_cv2(observations_list[idx], prefix=prefix)


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
    elif data["action"] == "turnLeftTwice":
        return HabitatSimActions.TURN_LEFT_TWICE
    elif data["action"] == "turnRightTwice":
        return HabitatSimActions.TURN_RIGHT_TWICE
    elif data["action"] == "moveForwardTwice":
        return HabitatSimActions.MOVE_FORWARD_TWICE
    elif data["action"] == "moveBackwardTwice":
        return HabitatSimActions.MOVE_BACKWARD_TWICE
    return HabitatSimActions.STOP


def log_action_data(data, i):
    if data["action"] != "stepPhysics":
        print("Action: {} - {}".format(data["action"], i))
    else:
        print("Action {} - {}".format(data["action"], i))


def run_reference_replay(cfg, restore_state=False, log_action=False, num_episodes=None):
    instructions = []
    with habitat.Env(cfg) as env:
        obs_list = []

        if num_episodes is None:
            num_episodes = len(env.episodes)

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
            for data in env.current_episode.reference_replay:
                if log_action:
                    log_action_data(data, i)
                action = get_habitat_sim_action(data)
                print(restore_state)
                if not restore_state:
                    observations = env.step(action=action, replay_data=data)
                else:
                    agent_state = data["agent_state"]
                    sensor_states = data["agent_state"]["sensor_data"]
                    object_states = data["object_states"]
                    observations = env._sim.get_observations_at(agent_state["position"], agent_state["rotation"], sensor_states, object_states)
                
                if i + 1 <= len(env.current_episode.reference_replay) - 1:
                    prev_action = env.current_episode.reference_replay[i - 1]
                    prev_prev_action = env.current_episode.reference_replay[i - 2]
                    if data["action"] == "grabReleaseObject":
                        im = Image.fromarray(observations["rgb"])
                        im.save('outfile{}.jpg'.format(i))
                    if prev_action["action"] == "grabReleaseObject" and data["action"] == "stepPhysics":
                        im = Image.fromarray(observations["rgb"])
                        im.save('outfile{}.jpg'.format(i))
                    if prev_prev_action["action"] == "grabReleaseObject" and prev_action["action"] == "stepPhysics":
                        im = Image.fromarray(observations["rgb"])
                        im.save('outfile{}.jpg'.format(i))
                        
                observation_list.append(observations)
                i+=1
            obs_list.append(observation_list)
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
        "--restore-state", dest='restore_state', action='store_true'
    )
    parser.add_argument(
        "--log-action", dest='log_action', action='store_true'
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.replay_episode
    cfg.freeze()

    observations = run_reference_replay(cfg, args.restore_state, args.log_action, num_episodes=1)
    make_videos(observations, args.output_prefix)

if __name__ == "__main__":
    main()
