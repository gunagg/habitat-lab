#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import numpy as np
import magnum as mn
import sys
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
        self.nearest_object_id = -1
        self.gripped_object_id = -1
        self.gripped_object_transformation = None
        self.agetn_object_handle = "cylinderSolid_rings_1_segments_12_halfLen_1_useTexCoords_false_useTangents_false_capEnds_true"

        agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_config = self._get_agent_config(agent_id)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT

    def reconfigure(self, config: Config) -> None:
        super().reconfigure(config)
        self._initialize_objects(config)

    def reset(self):
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def _initialize_objects(self, sim_config):
        # contact test object for agent
        self.add_contact_test_object(self.agetn_object_handle)
        objects = sim_config.objects
        obj_attr_mgr = self.get_object_template_manager()

        # first remove all existing objects
        existing_object_ids = self.get_existing_object_ids()

        if len(existing_object_ids) > 0:
            for obj_id in existing_object_ids:
                self.remove_object(obj_id)
                object_ = self.get_object_from_scene(obj_id)
                if object_ is not None:
                    self.remove_contact_test_object(object_["object_handle"])
            self.clear_recycled_object_ids()
            self.clear_scene_objects()

        self.sim_objid_to_replay_objid_mapping = {}
        self.replay_objid_to_sim_objid_mapping = {}

        if objects is not None:
            # Sort objects by object id
            object_map = {}
            for object_ in objects:
                object_map[object_["object_id"]] = object_

            for key in sorted(object_map.keys()):
                object_ = object_map[key]
                object_handle = object_["object_template"].split('/')[-1].split('.')[0]
                object_template = "data/test_assets/objects/{}".format(object_handle)
                object_pos = object_["position"]
                rotation = quat_from_coeffs(object_["rotation"])
                object_rotation = quat_to_magnum(rotation)

                object_template_id = obj_attr_mgr.load_object_configs(
                    object_template
                )[0]
                object_attr = obj_attr_mgr.get_template_by_ID(object_template_id)
                obj_attr_mgr.register_template(object_attr)

                object_id = self.add_object_by_handle(object_attr.handle)
                self.add_contact_test_object(object_attr.handle)

                self.set_translation(object_pos, object_id)
                self.set_rotation(object_rotation, object_id)

                object_['object_handle'] = "data/test_assets/objects/{}.object_config.json".format(object_handle)
                self.add_object_in_scene(object_id, object_)

                self.set_object_motion_type(MotionType.DYNAMIC, object_id)

                self.sim_objid_to_replay_objid_mapping[object_id] = object_["object_id"]
                self.replay_objid_to_sim_objid_mapping[object_["object_id"]] = object_id

    def get_resolution(self):
        resolution = self._default_agent.agent_config.sensor_specifications[
            0
        ].resolution
        return mn.Vector2(list(map(int, resolution)))

    def clear_scene_objects(self):
        self._scene_objects = []

    def add_object_in_scene(self, objectId, data):
        data["object_id"] = objectId
        self._scene_objects.append(data)

    def update_object_in_scene(self, prevObjectId, newObjectId):
        for index in range(len(self._scene_objects)):
            if self._scene_objects[index]["object_id"] == prevObjectId:
                self._scene_objects[index]["object_id"] = newObjectId

    def get_object_from_scene(self, objectId):
        for index in range(len(self._scene_objects)):
            if self._scene_objects[index]["object_id"] == objectId:
                return self._scene_objects[index]
        return None
    
    def is_collision(self, handle, translation, is_navigation_test = False):
        return self.pre_add_contact_test(handle, translation, is_navigation_test)

    def is_agent_colliding(self, action, agentTransform):
        stepSize = self.config.agents[0].action_space.FORWARD_STEP_SIZE
        if action == "move_forward":
            position = agentTransform.backward * (-1 * stepSize)
            newPosition = agentTransform.translation + position
            filteredPoint = self.pathfinder.try_step(
                agentTransform.translation,
                newPosition
            )
            filterDiff = filteredPoint - newPosition
            # adding buffer of 0.1 y to avoid collision with navmesh
            finalPosition = newPosition + filterDiff + mn.Vector3(0.0, 0.1, 0.0)
            collision = self.is_collision(self.agetn_object_handle, finalPosition, True)
            return {
                "collision": collision,
                "position": finalPosition
            }
        elif action == "move_backward":
            position = agentTransform.backward * stepSize
            newPosition = agentTransform.translation + position
            filteredPoint = self.pathfinder.try_step(
                agentTransform.translation,
                newPosition
            )
            filterDiff = filteredPoint - newPosition
            # adding buffer of 0.1 y to avoid collision with navmesh
            finalPosition = newPosition + filterDiff + mn.Vector3(0.0, 0.1, 0.0)
            collision = self.is_collision(self.agetn_object_handle, finalPosition, True)
            return {
                "collision": collision,
                "position": finalPosition
            }
        return {
            "collision": False
        }

    def get_object_under_cross_hair(self):
        ray = self.unproject(self.get_crosshair_position())
        cross_hair_point = ray.direction
        ref_point = self._default_agent.body.object.absolute_translation

        nearest_object_id = self.find_nearest_object_under_crosshair(
            cross_hair_point, ref_point, self.get_resolution(), 1.5
        )
        return nearest_object_id

    def draw_bb_around_nearest_object(self, object_id):
        if object_id == -1:
            if self.nearest_object_id != -1 and self.gripped_object_id != self.nearest_object_id:
                self.set_object_bb_draw(False, self.nearest_object_id)
                self.nearest_object_id = object_id
        else:
            if self.nearest_object_id != -1 and self.gripped_object_id != self.nearest_object_id:
                self.set_object_bb_draw(False, self.nearest_object_id)
                self.nearest_object_id = -1
            object_ = self.get_object_from_scene(object_id)
            if object_["is_receptacle"] == True:
                return
            if self.nearest_object_id != object_id:
                self.nearest_object_id = object_id
                self.set_object_bb_draw(True, self.nearest_object_id, 0)

    def step(self, action: int):
        dt = 1.0 / 10.0
        self._num_total_frames += 1
        collided = False

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
            if self.gripped_object_id != -1:
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
                scene_object = self.get_object_from_scene(self.gripped_object_id)

                # find no collision point
                contact = self.is_collision(scene_object["object_handle"], new_object_position)
                while contact:
                    new_object_position = mn.Vector3(
                        new_object_position.x,
                        new_object_position.y + 0.25,
                        new_object_position.z,
                    )
                    contact = self.is_collision(scene_object["object_handle"], new_object_position)

                new_object_id = self.add_object_by_handle(
                    scene_object["object_handle"]
                )
                self.set_translation(new_object_position, new_object_id)

                self.update_object_in_scene(self.gripped_object_id, new_object_id)
                self.gripped_object_id = -1
            elif nearest_object_id != -1:
                self.gripped_object_transformation = self.get_transformation(
                    nearest_object_id
                )
                self.remove_object(nearest_object_id)
                self.gripped_object_id = nearest_object_id
        elif action_spec.name == "no_op":
            super().step_world(action_spec.actuation.amount)
        else:
            agent_transform = self._default_agent.body.object.transformation
            data = self.is_agent_colliding(action_spec.name, agent_transform)
            if not data["collision"]:
                self._default_agent.act(action)
                collided = data["collision"]
                self._last_state = self._default_agent.get_state()

        object_id = self.get_object_under_cross_hair()
        self.draw_bb_around_nearest_object(object_id)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = self.gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def restore_object_states(self, object_states: Dict = {}):
        for object_state in object_states:
            object_id = object_state["object_id"]
            translation = object_state["translation"]
            rotation = object_state["rotation"]

            object_translation = mn.Vector3(translation)
            if isinstance(rotation, list):
                object_rotation = quat_from_coeffs(rotation)

            object_rotation = quat_to_magnum(object_rotation)
            self.set_translation(object_translation, object_id)
            self.set_rotation(object_rotation, object_id)
    
    def update_drop_point(self, replay_data=None, show=False):
        if self.gripped_object_id == -1 or show == False:
            position = self._default_agent.body.object.absolute_translation
            position = mn.Vector3(position.x, position.y - 0.5, position.z)
            self.update_drop_point_node(position)
        else:
            if "object_drop_point" in replay_data.keys() and len(replay_data["object_drop_point"]) > 0:
                position = mn.Vector3(replay_data["object_drop_point"])
                self.update_drop_point_node(position)

    def step_from_replay(self, action: int, replay_data: Dict = {}):
        dt = 1.0 / 10.0
        self._num_total_frames += 1
        collided = False

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        if action_spec.name == "grab_or_release_object_under_crosshair":
            action_data = replay_data["action_data"]
            if action_data["gripped_object_id"] != -1:
                if replay_data["is_release_action"]:
                    # Fetch object handle and drop point from replay
                    new_object_position = mn.Vector3(action_data["new_object_translation"])
                    scene_object = self.get_object_from_scene(action_data["gripped_object_id"])
                    new_object_id = self.add_object_by_handle(
                        scene_object["object_handle"]
                    )
                    self.set_translation(new_object_position, new_object_id)

                    self.update_object_in_scene(new_object_id, action_data["gripped_object_id"])
                    self.gripped_object_id = replay_data["gripped_object_id"]
                elif replay_data["is_grab_action"]:
                    self.gripped_object_transformation = self.get_transformation(
                        action_data["gripped_object_id"]
                    )
                    self.remove_object(action_data["gripped_object_id"])
                    self.gripped_object_id = replay_data["gripped_object_id"]
        elif action_spec.name == "no_op":
            self.restore_object_states(replay_data["object_states"])
        else:
            if action_spec.name == "look_up" or action_spec.name == "look_down":
                sensor_data = replay_data["agent_state"]["sensor_data"]
                for sensor_key, v in self._default_agent._sensors.items():
                    rotation = None
                    if sensor_key in sensor_data.keys():
                        rotation = sensor_data[sensor_key]["rotation"]
                    else:
                        rotation = sensor_data["rgb"]["rotation"]
                    rotation = quat_from_coeffs(rotation)
                    agent_rotation = quat_to_magnum(rotation)
                    v.object.rotation = agent_rotation
            else:
                success = self.set_agent_state(
                    replay_data["agent_state"]["position"], replay_data["agent_state"]["rotation"], reset_sensors=False
                )
            collided = replay_data["collision"]
            self._last_state = self._default_agent.get_state()

        self.draw_bb_around_nearest_object(replay_data["object_under_cross_hair"])
        self.update_drop_point(replay_data)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = self.gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations
