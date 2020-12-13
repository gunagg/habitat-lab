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
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum, quat_to_coeffs, quat_from_magnum
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
        self.agent_object_handle = "cylinderSolid_rings_1_segments_12_halfLen_1_useTexCoords_false_useTangents_false_capEnds_true"

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
            sim_obs = self.get_sensor_observations(agent_ids=self._default_agent_id)

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def remove_existing_objects(self):
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

    def _initialize_objects(self, sim_config):
        # contact test object for agent
        self.add_contact_test_object(self.agent_object_handle)
        objects = sim_config.objects
        obj_attr_mgr = self.get_object_template_manager()

        self.remove_existing_objects()

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
    
    def check_object_exists_in_scene(self, object_id):
        exists = object_id in self.get_existing_object_ids()
        return exists
    
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
            collision = self.is_collision(self.agent_object_handle, finalPosition, True)
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
            collision = self.is_collision(self.agent_object_handle, finalPosition, True)
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
            pass
        else:
            agent_transform = self._default_agent.body.object.transformation
            data = self.is_agent_colliding(action_spec.name, agent_transform)
            if not data["collision"]:
                self._default_agent.act(action)
                collided = data["collision"]
                self._last_state = self._default_agent.get_state()

        # step world physics
        super().step_world(dt)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations(agent_ids=self._default_agent_id)
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = self.gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def restore_object_states(self, object_states: Dict = {}):
        object_ids = []
        for object_state in object_states:
            object_id = object_state["object_id"]
            translation = object_state["translation"]
            object_rotation = object_state["rotation"]

            object_translation = mn.Vector3(translation)
            if isinstance(object_rotation, list):
                object_rotation = quat_from_coeffs(object_rotation)

            object_rotation = quat_to_magnum(object_rotation)
            self.set_translation(object_translation, object_id)
            self.set_rotation(object_rotation, object_id)
            object_ids.append(object_id)
        return object_ids
    
    def get_current_object_states(self):
        existing_object_ids = self.get_existing_object_ids()
        object_states = []
        for object_id in existing_object_ids:
            translation = self.get_translation(object_id)
            rotation = self.get_rotation(object_id)
            rotation = quat_from_magnum(rotation)
            scene_object = self.get_object_from_scene(object_id)

            object_state = {}
            object_state["object_id"] = object_id
            object_state["translation"] = np.array(translation).tolist()
            object_state["rotation"] = quat_to_coeffs(rotation).tolist()
            object_state["object_handle"] = scene_object["object_handle"]
            object_states.append(object_state)
        return object_states
    
    def get_agent_pose(self):
        agent_translation = self._default_agent.body.object.translation
        agent_rotation = self._default_agent.body.object.rotation
        sensor_data = {}
        for sensor_key, v in self._default_agent._sensors.items():
            rotation = quat_from_magnum(v.object.rotation)
            rotation = quat_to_coeffs(rotation).tolist()
            sensor_data[sensor_key] = {
                "rotation": rotation
            }
        
        return {
            "position": np.array(agent_translation).tolist(),
            "rotation": quat_to_coeffs(quat_from_magnum(agent_rotation)).tolist(),
            "sensor_data": sensor_data
        }
    
    def restore_sensor_states(self, sensor_data: Dict = {}):
        for sensor_key, v in self._default_agent._sensors.items():
            rotation = None
            if sensor_key in sensor_data.keys():
                rotation = sensor_data[sensor_key]["rotation"]
            else:
                rotation = sensor_data["rgb"]["rotation"]
            rotation = quat_from_coeffs(rotation)
            agent_rotation = quat_to_magnum(rotation)
            v.object.rotation = agent_rotation
    
    def add_objects_by_handle(self, objects):
        for object_ in objects:
            object_handle = object_["object_handle"]
            self.add_object_by_handle(object_handle)
    
    def update_drop_point(self, replay_data=None):
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
            if len(action_data.keys()) == 0 or action_data["gripped_object_id"] != -1:
                if replay_data["is_release_action"]:
                    # Fetch object handle and drop point from replay
                    new_object_position = mn.Vector3(action_data["new_object_translation"])
                    new_object_position.y = new_object_position.y + 0.5
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
            if "agent_state" in replay_data.keys():
                if action_spec.name == "look_up" or action_spec.name == "look_down":
                    sensor_data = replay_data["agent_state"]["sensor_data"]
                    self.restore_sensor_states(sensor_data)
                else:
                    success = self.set_agent_state(
                        replay_data["agent_state"]["position"], replay_data["agent_state"]["rotation"], reset_sensors=False
                    )
                collided = replay_data["collision"]
                self._last_state = self._default_agent.get_state()
            else:
                collided = replay_data["collision"]
                if not collided:
                    self._default_agent.act(action)

        self.draw_bb_around_nearest_object(replay_data["object_under_cross_hair"])

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations(agent_ids=self._default_agent_id, draw_crosshair=True)
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = self.gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        sensor_states: Optional[List[Dict]] = None,
        object_states: Optional[List[Dict]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        current_object_states = self.get_current_object_states()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )
        self.restore_sensor_states(sensor_states)
        current_state_objects = self.restore_object_states(object_states)

        object_to_re_add = []
        for object_id in self.get_existing_object_ids():
            if object_id not in current_state_objects:
                self.remove_object(object_id)
                object_to_re_add.append(self.get_object_from_scene(object_id))

        if success:
            sim_obs = self.get_sensor_observations(agent_ids=self._default_agent_id)

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
                self.add_objects_by_handle(object_to_re_add)
                self.restore_object_states(current_object_states)
                
            return observations
        else:
            return None
