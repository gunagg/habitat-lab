import argparse
import cv2
import habitat
import json
import sys
import time
import os
import math
import numpy as np

from habitat import Config
from habitat_sim.utils import viz_utils as vut
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
from psiturk_dataset.utils.utils import write_json


config = habitat.get_config("configs/tasks/object_rearrangement.yaml")


def are_points_navigable(sim, points):
    pathfinder = sim.pathfinder
    is_navigable_list = []
    for point in points:
        is_navigable_list.append(pathfinder.is_navigable(point))

    for i in range(len(points)):
        for j in range(len(points)):
            if i <= j:
                continue
            dist = sim.geodesic_distance(points[i], points[j])
            if dist == np.inf or dist == math.inf:
                return False
    
    if np.sum(is_navigable_list) != len(is_navigable_list):
        return False
    return True


def get_object_and_agent_state(sim):
    points = []
    # Append agent state
    agent_position = sim.get_agent_state().position
    points.append(agent_position)

    # Append object state
    object_ids = sim.get_existing_object_ids()
    for object_id in object_ids:
        points.append(sim.get_translation(object_id))
    
    return points


def run_validation(cfg, num_steps=5):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        obs_list = []
        non_navigable_episodes = []
        navigable_episodes = 0

        print("Total episodes: {}".format(len(env.episodes)))
        for ep_id in range(len(env.episodes)):
            observation_list = []

            obs = env.reset()

            print('Scene has physiscs {}'.format(cfg.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS))
            physics_simulation_library = env._sim.get_physics_simulation_library()
            print("Physics simulation library: {}".format(physics_simulation_library))
            print("Episode length: {}, Episode index: {}".format(len(env.current_episode.reference_replay), ep_id))
            print("Scene Id : {}".format(env.current_episode.scene_id))
            
            action = possible_actions.index("NO_OP")
            for i in range(num_steps):
                observations = env.step(action=action)
            sim = env._sim
            points = get_object_and_agent_state(sim)
            is_navigable = are_points_navigable(sim, points)
            navigable_episodes += int(is_navigable)

            if not is_navigable:
                non_navigable_episodes.append(env.current_episode.episode_id)

            if ep_id % 10 == 0:
                print("Total {}/{} episodes are navigable".format(navigable_episodes, len(env.episodes)))
        print("Total {}/{} episodes are navigable".format(navigable_episodes, len(env.episodes)))
        print(non_navigable_episodes)
        write_json(non_navigable_episodes, "data/hit_data/non_navigable_episodes_ll.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--num-steps", type=int, default=5
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.episodes
    cfg.freeze()
    print(args)
    run_validation(cfg, args.num_steps)

if __name__ == "__main__":
    main()
