#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type

import attr
import habitat_sim
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, Simulator
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import merge_sim_episode_config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import SimulatorTaskAction, Measure
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: List[int]


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementObjectSpec(RearrangementSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """
    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_handle: Optional[str] = attr.ib(
        default="", validator=not_none_validator
    )
    object_template: Optional[str] = attr.ib(
        default="", validator=not_none_validator
    )
    object_icon: Optional[str] = attr.ib(default="")
    motion_type: Optional[str] = attr.ib(default=None)
    is_receptacle: Optional[bool] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action.
    """
    action: str = attr.ib(default=None, validator=not_none_validator)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(Episode):
    r"""Specification of episode that includes initial position and rotation
    of agent, goal specifications, instruction specifications, reference path,
    and optional shortest paths.

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        instruction: single natural language instruction for the task.
        reference_replay: List of keypresses which gives the reference
            actions to the goal that aligns with the instruction.
    """
    goals: List[RearrangementSpec] = attr.ib(
        default=None, validator=not_none_validator
    )
    reference_replay: List[Dict] = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction: InstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    objects: List[RearrangementObjectSpec] = attr.ib(
        default=None, validator=not_none_validator
    )


@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "instruction"
        self.observation_space = spaces.Discrete(0)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: RearrangementEpisode,
        **kwargs
    ):
        return episode.instruction.instruction_tokens

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_measure
class ObjectToReceptacleDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_to_receptacle_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToReceptacleDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()
        obj_id = -1
        receptacle_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object["is_receptacle"] == False:
                obj_id = scene_object["object_id"]
            else:
                receptacle_id = scene_object["object_id"]

        if receptacle_id == -1:
            self._metric = 100
        elif obj_id != -1:
            object_position = np.array(
                self._sim.get_translation(obj_id)
            ).tolist()

            receptacle_position = np.array(
                self._sim.get_translation(receptacle_id)
            ).tolist()

            self._metric = self._geo_dist(
                object_position, receptacle_position
            )
        else:
            receptacle_position = np.array(
                self._sim.get_translation(receptacle_id)
            ).tolist()

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                agent_position, receptacle_position
            )


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def _geo_dist(self, src_pos, object_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [object_pos])

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()

        sim_obj_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object["is_receptacle"] == False:
                sim_obj_id = scene_object["object_id"]

        if sim_obj_id != -1:
            previous_position = np.array(
                self._sim.get_translation(sim_obj_id)
            ).tolist()

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                previous_position, agent_position
            )
        else:
            self._metric = 0


@registry.register_measure
class AgentToReceptacleDistance(Measure):
    """The measure calculates the distance of receptacle from the agent"""

    cls_uuid: str = "agent_to_receptacle_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToReceptacleDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def _geo_dist(self, src_pos, object_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [object_pos])

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_ids = self._sim.get_existing_object_ids()

        sim_obj_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object["is_receptacle"] == True:
                sim_obj_id = scene_object["object_id"]

        if sim_obj_id != -1:
            previous_position = np.array(
                self._sim.get_translation(sim_obj_id)
            ).tolist()

            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position

            self._metric = self._geo_dist(
                previous_position, agent_position
            )
        else:
            self._metric = 100


@registry.register_measure
class RearrangementSuccess(Measure):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "rearrangement_success"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [ObjectToReceptacleDistance.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            ObjectToReceptacleDistance.cls_uuid
        ].get_metric()
        object_ids = self._sim.get_existing_object_ids()

        obj_id = -1
        receptacle_id = -1
        for object_id in object_ids:
            scene_object = self._sim.get_object_from_scene(object_id)
            if scene_object["is_receptacle"] == False:
                obj_id = scene_object["object_id"]
            else:
                receptacle_id = scene_object["object_id"]

        is_object_stacked = False
        is_object_not_in_air = False
        is_receptacle_not_in_air = False
        if obj_id != -1 and receptacle_id != -1:
            object_position = self._sim.get_translation(obj_id)
            receptacle_position = self._sim.get_translation(receptacle_id)

            object_y = object_position.y
            receptacle_y = receptacle_position.y + self._sim.get_object_bb_y_coord(receptacle_id)
            is_object_stacked = (object_y > receptacle_y)
            is_object_not_in_air = self._sim.contact_test(obj_id)
            is_receptacle_not_in_air = self._sim.contact_test(receptacle_id)

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called # type: ignore
            and distance_to_target <= self._config.SUCCESS_DISTANCE
            and is_object_stacked
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()
    sim_config.objects = episode.objects
    sim_config.freeze()

    return sim_config


@registry.register_task(name="RearrangementTask-v0")
class RearrangementTask(EmbodiedTask):
    r"""Language based Object Rearrangement Task
    Goal: An agent must rearrange objects in a 3D environment
        specified by a natural language instruction.
    Usage example:
        examples/object_rearrangement_example.py
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)
