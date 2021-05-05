import argparse
import attr
import cv2
import habitat
import json
import sys
import time
import os

from habitat import Config, get_config as get_task_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import make_video_cv2, observations_to_image, images_to_video, append_text_to_image

from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.common import quat_to_coeffs

from psiturk_dataset.utils.utils import write_json, write_gzip

from PIL import Image

config = habitat.get_config("configs/tasks/rearrangement_shortest_trajectory.yaml")

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def get_action(action):
    if action == HabitatSimActions.TURN_RIGHT:
        return "TURN_RIGHT"
    elif action == HabitatSimActions.TURN_LEFT:
        return "TURN_LEFT"
    elif action == HabitatSimActions.MOVE_FORWARD:
        return "MOVE_FORWARD"
    elif action == HabitatSimActions.MOVE_BACKWARD:
        return "MOVE_BACKWARD"
    elif action == HabitatSimActions.LOOK_UP:
        return "LOOK_UP"
    elif action == HabitatSimActions.LOOK_DOWN:
        return "LOOK_DOWN"
    elif action == HabitatSimActions.GRAB_RELEASE:
        return "GRAB_RELEASE"
    elif action == HabitatSimActions.NO_OP:
        return "NO_OP"
    return "STOP"



def object_state_to_json(object_states):
    object_states_json = []
    for object_ in object_states:
        object_states_json.append(attr.asdict(object_))
    return object_states_json


def get_action_data(action, sim):
    data = {}
    if action == HabitatSimActions.GRAB_RELEASE:
        data["gripped_object_id"] = sim.gripped_object_id
        data["is_grab_action"] = sim._prev_step_data["is_grab_action"]
        data["is_release_action"] = sim._prev_step_data["is_release_action"]
        action_data = {}
        if data["is_release_action"]:
            action_data["new_object_translation"] = sim._prev_step_data["new_object_translation"]
            action_data["new_object_id"] = sim._prev_step_data["new_object_id"]
            action_data["object_handle"] = sim._prev_step_data["object_handle"]
            action_data["gripped_object_id"] = sim._prev_step_data["gripped_object_id"]
        else:
            action_data["gripped_object_id"] = sim._prev_step_data["gripped_object_id"]
        data["action_data"] = action_data
        data["collision"] = False
    else:
        data["collision"] = sim._prev_step_data["collided"]
        data["object_under_cross_hair"] = sim._prev_step_data["nearest_object_id"]
        data["nearest_object_id"] = sim._prev_step_data["nearest_object_id"]
        data["gripped_object_id"] = sim._prev_step_data["gripped_object_id"]
    data["action"] = get_action(action)
    data["agent_state"] = sim.get_agent_pose()
    data["object_states"] = object_state_to_json(sim.get_current_object_states())
    return data


def get_episode_json(episode, reference_replay):
    episode.reference_replay = reference_replay
    episode._shortest_path_cache = None
    episode.scene_id = episode.scene_id.split("/")[-1]
    return attr.asdict(episode)


def generate_trajectories(cfg, num_episodes=1, output_prefix="s_path"):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        goal_radius = 0.7
        total_success = 0.0
        total_episodes = 0.0

        dataset = {
            "episodes": [],
            "instruction_vocab": {
                "sentences": []
            }
        }

        print("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            follower = ShortestPathFollower(
                env._sim, goal_radius, False
            )
            observation_list = []
            obs = env.reset()
            goal_idx = 0
            prev_action = 0
            success = 0
            grab_count = 0
            reference_replay = []
            while not env.episode_over:
                best_action = follower.get_next_action(
                    env.current_episode.goals[goal_idx].position
                )

                if (best_action is None or best_action == 0) and goal_idx == 0:
                    best_action = HabitatSimActions.LOOK_DOWN

                if env._sim._prev_sim_obs["object_under_cross_hair"] != -1 and prev_action != HabitatSimActions.GRAB_RELEASE and grab_count < 2:
                    best_action = HabitatSimActions.GRAB_RELEASE
                    goal_idx = 1
                    grab_count += 1

                if success:
                    best_action = HabitatSimActions.STOP
                #print(best_action, env._sim._prev_sim_obs["object_under_cross_hair"])

                observations = env.step(best_action)
                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"], "depth": observations["depth"]}, {})
                frame = append_text_to_image(frame, "Instruction: {}".format(env.current_episode.instruction.instruction_text))
                observation_list.append(frame)
                success += info["success"]
                prev_action = best_action
                action_data = get_action_data(best_action, env._sim)
                reference_replay.append(action_data)

            ep_data = get_episode_json(env.current_episode, reference_replay)
            del ep_data["_shortest_path_cache"]
            print("Episode success: {}".format(success))
            total_success += success
            total_episodes += 1
            make_videos([observation_list], output_prefix, ep_id)

            if success:
                dataset["episodes"].append(ep_data)

        print("\n\nEpisode success: {}".format(total_success / total_episodes))
        write_json(data, "data/episodes/ep_{}.json".format(ep_id))
        write_gzip("data/episodes/ep_{}.json".format(ep_id), "data/episodes/ep_{}.json".format(ep_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-episodes", type=int, default=1
    )
    parser.add_argument(
        "--scene", type=str, default="empty_house"
    )
    parser.add_argument(
        "--episodes", type=str, default="data/episodes/sampled.json.gz"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    #cfg.DATASET.CONTENT_SCENES = [args.scene]
    cfg.DATASET.CONTENT_SCENES = ["S9hNv5qa7GM"]
    cfg.DATASET.DATA_PATH = args.episodes
    cfg.freeze()

    observations = generate_trajectories(cfg, args.num_episodes)

if __name__ == "__main__":
    main()
