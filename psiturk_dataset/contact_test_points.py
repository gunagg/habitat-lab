#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import sys
import gzip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import json
import magnum as mn

import habitat
from habitat.sims import make_sim
from habitat_sim.utils.common import quat_from_coeffs


ISLAND_RADIUS_LIMIT = 1.5
VISITED_POINT_DICT = {}


def contact_test(sim, object_name, position):
    object_handle = "./data/test_assets/objects/{}.phys_properties.json".format(object_name)
    return sim.pre_add_contact_test(object_handle, mn.Vector3(position))


def contact_test_rotation(sim, object_name, position, rotation):
    object_handle = "./data/test_assets/objects/{}.phys_properties.json".format(object_name)
    return sim.pre_add_contact_test(object_handle, mn.Vector3(position), quat_from_coeffs(rotation))


def add_contact_test_object(sim, object_name):
    object_handle = "./data/test_assets/objects/{}.phys_properties.json".format(object_name)
    sim.add_contact_test_object(object_handle)


def generate_points(
    config,
    episode_path
):
    # Initialize simulator
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    episode_count = 0
    episode_file = open(episode_path, "r")
    episodes = json.loads(episode_file.read())

    all_points = []

    for episode in episodes["episodes"]:
        print("\nEpisode {}\n".format(episode["episode_id"]))
        objects = episode["objects"]

        for object_ in objects:
            # Adding contact test objects to test object positions
            object_name = object_["objectHandle"].split("/")[-1].split(".")[0]
            add_contact_test_object(sim, object_name)
            collision_no_rot =  contact_test(sim, object_name, object_["position"])
            collision =  contact_test_rotation(sim, object_name, object_["position"], object_["rotation"])
            print("Object: {}, Contact: {} - {}".format(object_name, collision, collision_no_rot))
            if collision:
                print("Bad point!!!")
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Validate episodes."
    )
    parser.add_argument(
        "--task-config",
        default="psiturk_dataset/rearrangement.yaml",
        help="Task configuration file for initializing a Habitat environment",
    )
    parser.add_argument(
        "--scenes",
        help="Scenes"
    )
    parser.add_argument(
        "--path",
        default="episode_2.json",
        help="Episode path",
    )

    args = parser.parse_args()
    opts = []
    config = habitat.get_config(args.task_config.split(","), opts)

    dataset_type = config.DATASET.TYPE
    if args.scenes is not None:
        config.defrost()
        config.SIMULATOR.SCENE = args.scenes
        config.freeze()

    if dataset_type == "Interactive":
        dataset = generate_points(
            config,
            args.path
        )
    else:
        print(f"Unknown dataset type: {dataset_type}")