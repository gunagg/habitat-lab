
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, List, Any

import attr

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.core.utils import Singleton

from habitat.core.embodied_task import SimulatorTaskAction
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat.sims.habitat_simulator.actions import (
    HabitatSimV1ActionSpaceConfiguration,
    HabitatSimActions
)


@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 1.5


@registry.register_action_space_configuration(name="RearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")
        if not HabitatSimActions.has_action("MOVE_BACKWARD"):
            HabitatSimActions.extend_action_space("MOVE_BACKWARD")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.MOVE_BACKWARD: habitat_sim.ActionSpec(
                "move_backward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            )
        }

        config.update(new_config)
        return config


@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)


@registry.register_task_action
class MoveBackwardAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD)
