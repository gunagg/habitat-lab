import argparse
import cv2
import habitat
import json
import time

from habitat import Config
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs

from threading import Thread
from time import sleep

config = habitat.get_config("configs/tasks/object_rearrangement.yaml")
config.defrost()
config.FORWARD_STEP_SIZE = 0.25
config.TURN_ANGLE = 10.0
config.TILT_ANGLE = 10.0
config.SIMULATOR.TYPE = "RearrangementSim-v0"
config.SIMULATOR.ACTION_SPACE_CONFIG = "RearrangementActions-v0"
config.SIMULATOR.CROSSHAIR_POS = [320, 240]
config.SIMULATOR.GRAB_DISTANCE = 1.5
config.SIMULATOR.VISUAL_SENSOR = "rgb"
config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
config.TASK.TYPE = "RearrangementTask-v0"
config.TASK.ACTIONS.MOVE_BACKWARD = Config()
config.TASK.ACTIONS.MOVE_BACKWARD.TYPE = "MoveBackwardAction"
config.TASK.ACTIONS.GRAB_RELEASE = Config()
config.TASK.ACTIONS.GRAB_RELEASE.TYPE = "GrabOrReleaseAction"
config.TASK.ACTIONS.NO_OP = Config()
config.TASK.ACTIONS.NO_OP.TYPE = "NoOpAction"
config.TASK.SUCCESS_DISTANCE = 1.0
config.TASK.OBJECT_TO_GOAL_DISTANCE = Config()
config.TASK.OBJECT_TO_GOAL_DISTANCE.TYPE = "ObjectToGoalDistance"
config.TASK.AGENT_TO_OBJECT_DISTANCE = Config()
config.TASK.AGENT_TO_OBJECT_DISTANCE.TYPE = "AgentToObjectDistance"
config.TASK.SENSORS = [
    "INSTRUCTION_SENSOR",
]
config.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
config.DATASET.TYPE = "RearrangementDataset-v0"
config.DATASET.SPLIT = "train"
config.DATASET.DATA_PATH = (
    "data/datasets/object_rearrangement/v1/{split}/{split}.json.gz"
)
config.freeze()


def make_video_cv2(
    observations, cross_hair=None, prefix="", open_vid=True, fps=15, output_path="./demos/"
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(videodims)
    video_file = output_path + prefix + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = vut.get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        rgb_im_1st_person = cv2.cvtColor(ob["rgb"], cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        vut.display_video(video_file)


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
    return HabitatSimActions.STOP


def run_reference_replay(cfg, num_episodes=None):
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
                # if data["action"] != "stepPhysics":
                #     cnt += 1
                #     print("Action: {} - {},  Step: {}".format(data["action"], i, cnt))
                # else:
                #     print("Action {} - {}".format(data["action"], i))
                action = get_habitat_sim_action(data)
                observations = env.step(action=action, replay_data=data)
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
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.replay_episode
    cfg.freeze()

    observations = run_reference_replay(cfg, num_episodes=1)
    make_videos(observations, args.output_prefix)

if __name__ == "__main__":
    main()
