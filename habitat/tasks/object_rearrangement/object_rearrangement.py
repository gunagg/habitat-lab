#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import attr
import habitat_sim
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationTask, NavigationEpisode
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


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
    object_template: Optional[str] = attr.ib(
        default="data/test_assets/objects/chair"
    )


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(NavigationEpisode):
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
    reference_replay: List[str] = attr.ib(
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
        return {
            "text": episode.instruction.instruction_text,
            "tokens": episode.instruction.instruction_tokens,
            "reference_replay": episode.reference_replay,
        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_task(name="RearrangementTask-v0")
class RearrangementTask(NavigationTask):
    r"""Language based Object Rearrangement Task
    Goal: An agent must rearrange objects in a 3D environment
        specified by a natural language instruction.
    Usage example:
        examples/object_rearrangement_example.py
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
