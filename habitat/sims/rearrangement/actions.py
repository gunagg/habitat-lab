
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, List, Any

import attr

import habitat_sim
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.core.utils import Singleton

from habitat.core.embodied_task import SimulatorTaskAction
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration
)


@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [320, 240]
    amount: float = 1.5


@registry.register_action_space_configuration(name="RearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(ActionSpaceConfiguration):
    def __init__(self, config):
        # super().__init__(config)
        self.config = config
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")
        if not HabitatSimActions.has_action("MOVE_BACKWARD"):
            HabitatSimActions.extend_action_space("MOVE_BACKWARD")
        if not HabitatSimActions.has_action("NO_OP"):
            HabitatSimActions.extend_action_space("NO_OP")
        if not HabitatSimActions.has_action("STOP"):
            HabitatSimActions.extend_action_space("STOP")
        if not HabitatSimActions.has_action("START"):
            HabitatSimActions.extend_action_space("START")
        if not HabitatSimActions.has_action("TURN_LEFT_TWICE"):
            HabitatSimActions.extend_action_space("TURN_LEFT_TWICE")
        if not HabitatSimActions.has_action("TURN_RIGHT_TWICE"):
            HabitatSimActions.extend_action_space("TURN_RIGHT_TWICE")
        if not HabitatSimActions.has_action("MOVE_FORWARD_TWICE"):
            HabitatSimActions.extend_action_space("MOVE_FORWARD_TWICE")
        if not HabitatSimActions.has_action("MOVE_BACKWARD_TWICE"):
            HabitatSimActions.extend_action_space("MOVE_BACKWARD_TWICE")

    def get(self):
        config = self.config
        new_config = {
            HabitatSimActions.START: habitat_sim.ActionSpec("start"),
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.LOOK_UP: habitat_sim.ActionSpec(
                "look_up",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.LOOK_DOWN: habitat_sim.ActionSpec(
                "look_down",
                habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE),
            ),
            HabitatSimActions.MOVE_BACKWARD: habitat_sim.ActionSpec(
                "move_backward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            HabitatSimActions.NO_OP: habitat_sim.ActionSpec(
                "no_op",
                habitat_sim.ActuationSpec(amount=0.1),
            ),
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            ),
            HabitatSimActions.TURN_LEFT_TWICE: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE * 2.0),
            ),
            HabitatSimActions.TURN_RIGHT_TWICE: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE * 2.0),
            ),
            HabitatSimActions.MOVE_FORWARD_TWICE: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE * 2.0),
            ),
            HabitatSimActions.MOVE_BACKWARD_TWICE: habitat_sim.ActionSpec(
                "move_backward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE * 2.0),
            ),
        }

        config.update(new_config)
        return config


@registry.register_task_action
class NoOpAction(SimulatorTaskAction):
    name: str = "NO_OP"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.NO_OP,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.NO_OP)


@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.GRAB_RELEASE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)


@registry.register_task_action
class MoveBackwardAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_BACKWARD,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD)


@registry.register_task_action
class StartAction(SimulatorTaskAction):
    name: str = "START"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_start_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_start_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore


@registry.register_task_action
class TurnLeftTwiceAction(SimulatorTaskAction):
    name: str = "TURN_LEFT_TWICE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.TURN_LEFT_TWICE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.TURN_LEFT_TWICE)


@registry.register_task_action
class TurnRightTwiceAction(SimulatorTaskAction):
    name: str = "TURN_RIGHT_TWICE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.TURN_RIGHT_TWICE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.TURN_RIGHT_TWICE)


@registry.register_task_action
class MoveForwardTwiceAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD_TWICE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_FORWARD_TWICE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_FORWARD_TWICE)


@registry.register_task_action
class MoveBackwardTwiceAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD_TWICE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        print("move back action")
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_BACKWARD_TWICE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD_TWICE)


@registry.register_task_action
class LookDownTwiceAction(SimulatorTaskAction):
    name: str = "LOOK_DOWN_TWICE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_DOWN_TWICE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_DOWN_TWICE)


@registry.register_task_action
class LookUpTwiceAction(SimulatorTaskAction):
    name: str = "LOOK_UP_TWICE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        print("move back action")
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_UP_TWICE,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_UP_TWICE)


@registry.register_task_action
class MoveForwardTurnLeftAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD_TURN_LEFT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_FORWARD_TURN_LEFT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_FORWARD_TURN_LEFT)


@registry.register_task_action
class MoveForwardTurnRightAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD_TURN_RIGHT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_FORWARD_TURN_RIGHT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_FORWARD_TURN_RIGHT)


@registry.register_task_action
class MoveForwardLookDownAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD_LOOK_DOWN"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_FORWARD_LOOK_DOWN,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_FORWARD_LOOK_DOWN)


@registry.register_task_action
class MoveForwardLookUpAction(SimulatorTaskAction):
    name: str = "MOVE_FORWARD_LOOK_UP"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_FORWARD_LOOK_UP,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_FORWARD_LOOK_UP)


@registry.register_task_action
class MoveBackwardTurnLeftAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD_TURN_LEFT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_BACKWARD_TURN_LEFT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD_TURN_LEFT)


@registry.register_task_action
class MoveBackwardTurnRightAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD_TURN_RIGHT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_BACKWARD_TURN_RIGHT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD_TURN_RIGHT)


@registry.register_task_action
class MoveBackwardLookDownAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD_LOOK_DOWN"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_BACKWARD_LOOK_DOWN,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD_LOOK_DOWN)


@registry.register_task_action
class MoveBackwardLookUpAction(SimulatorTaskAction):
    name: str = "MOVE_BACKWARD_LOOK_UP"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.MOVE_BACKWARD_LOOK_UP,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD_LOOK_UP)


@registry.register_task_action
class LookUpTurnLeftAction(SimulatorTaskAction):
    name: str = "LOOK_UP_TURN_LEFT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_UP_TURN_LEFT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_UP_TURN_LEFT)


@registry.register_task_action
class LookUpTurnRightAction(SimulatorTaskAction):
    name: str = "LOOK_UP_TURN_RIGHT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_UP_TURN_RIGHT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_UP_TURN_RIGHT)


@registry.register_task_action
class LookDownTurnLeftAction(SimulatorTaskAction):
    name: str = "LOOK_DOWN_TURN_LEFT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_DOWN_TURN_LEFT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_DOWN_TURN_LEFT)


@registry.register_task_action
class LookDownTurnRightAction(SimulatorTaskAction):
    name: str = "LOOK_DOWN_TURN_RIGHT"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if "replay_data" in kwargs.keys() and len(kwargs["replay_data"].keys()) > 0:
            return self._sim.step_from_replay(
                HabitatSimActions.LOOK_DOWN_TURN_RIGHT,
                replay_data=kwargs["replay_data"]
            )
        return self._sim.step(HabitatSimActions.LOOK_DOWN_TURN_RIGHT)
