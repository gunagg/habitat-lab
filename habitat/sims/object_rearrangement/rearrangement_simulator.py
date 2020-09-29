#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import numpy as np
import magnum as mn
from gym import spaces

import habitat_sim
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import Space
from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum
from habitat_sim.physics import MotionType
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


@registry.register_simulator(name="RearrangementSim-v0")
class RearrangementSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim with
    object rearrangement functionalities.
    """

    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        #self.gripped_object_id = -1
        self.gripped_object_transformation = np.eye(4)

        agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_config = self._get_agent_config(agent_id)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        self._initialize_objects()

    def reset(self):
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def _initialize_objects(self):
        objects = self.habitat_config.objects
        obj_attr_mgr = self.get_object_template_manager()

        # first remove all existing objects
        existing_object_ids = self.get_existing_object_ids()

        if len(existing_object_ids) > 0:
            for obj_id in existing_object_ids:
                self.remove_object(obj_id)

        self.sim_object_to_objid_mapping = {}
        self.objid_to_sim_object_mapping = {}

        if objects is not None:
            for object_ in objects:
                object_handle = object_["object_template"].split('/')[-1].split('.')[0]
                object_template = "data/test_assets/objects/{}".format(object_handle)
                object_pos = object_["position"]
                # object_rot = objects["rotation"]

                object_template_id = obj_attr_mgr.load_object_configs(
                    object_template
                )[0]
                object_attr = obj_attr_mgr.get_template_by_ID(object_template_id)
                obj_attr_mgr.register_template(object_attr)

                object_id = self.add_object_by_handle(object_attr.handle)

                self.sim_object_to_objid_mapping[object_id] = object_["object_id"]
                self.objid_to_sim_object_mapping[object_["object_id"]] = object_id

                self.set_translation(object_pos, object_id)
                self.sample_object_state(object_id)
                object_['object_handle'] = "data/test_assets/objects/{}".format(object_["object_template"].split('/')[-1])
                # if isinstance(object_rot, list):
                #    object_rot = quat_from_coeffs(object_rot)

                # object_rot = quat_to_magnum(object_rot)
                # self.set_rotation(object_rot, object_id)
                self.add_object_in_scene(object_id, object_)

                self.set_object_motion_type(MotionType.DYNAMIC, object_id)

        # Recompute the navmesh after placing all the objects.
        self.recompute_navmesh(self.pathfinder, self.navmesh_settings, True)

    def get_resolution(self):
        resolution = self._default_agent.agent_config.sensor_specifications[
            0
        ].resolution
        return mn.Vector2(list(map(int, resolution)))

    def add_object_in_scene(self, objectId, data):
        data["objectId"] = objectId
        self._scene_objects.append(data)

    def update_object_in_scene(self, prevObjectId, newObjectId):
        for index in range(len(self._scene_objects)):
            if self._scene_objects[index]["objectId"] == prevObjectId:
                self._scene_objects[index]["objectId"] = newObjectId

    def get_object_from_scene(self, objectId):
        for index in range(len(self._scene_objects)):
            if self._scene_objects[index]["objectId"] == objectId:
                return self._scene_objects[index]
        return None

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def step(self, action: int):
        dt = 1.0 / 10.0
        self._num_total_frames += 1
        collided = False
        gripped_object_id = self.gripped_object_id

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        if action_spec.name == "grab_or_release_object_under_crosshair":
            ray = self.unproject(action_spec.actuation.crosshair_pos)
            cross_hair_point = ray.direction
            ref_point = self._default_agent.body.object.absolute_translation

            nearest_object_id = self.find_nearest_object_under_crosshair(
                cross_hair_point, ref_point, self.get_resolution(), action_spec.actuation.amount
            )

            # already gripped an object
            if gripped_object_id != -1:
                ref_transform = self._default_agent.body.object.transformation
                ray_hit_info = self.find_floor_position_under_crosshair(
                    cross_hair_point, ref_transform,
                     self.get_resolution(), action_spec.actuation.amount
                )

                floor_position = ray_hit_info.point
                if floor_position is None:
                    return True

                y_value = floor_position.y
                y_value = max(self.gripped_object_transformation.translation.y, y_value)

                new_object_position = mn.Vector3(
                    floor_position.x, y_value, floor_position.z
                )
                scene_object = self.get_object_from_scene(gripped_object_id)
                new_object_id = self.add_object_by_handle(
                    scene_object["object_handle"]
                )

                self.set_translation(new_object_position, new_object_id)
                while self.contact_test(new_object_id):
                    new_object_position = mn.Vector3(
                        new_object_position.x,
                        new_object_position.y + 0.25,
                        new_object_position.z,
                    )
                    self.set_translation(new_object_position, new_object_id)
                self.update_object_in_scene(gripped_object_id, new_object_id)

                gripped_object_id = -1
            elif nearest_object_id != -1:
                self.gripped_object_transformation = self.get_transformation(
                    nearest_object_id
                )
                self.remove_object(nearest_object_id)
                gripped_object_id = nearest_object_id
        elif action_spec.name == "no_op":
            super().step_world(action_spec.actuation.amount)
        else:
            collided = self._default_agent.act(action)
            self._last_state = self._default_agent.get_state()

        # step physics by dt
        # super().step_world(dt)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations
