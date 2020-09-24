import argparse
from math import pi

import numpy as np

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

class ReplayAgent(habitat.Agent):
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, replay_sample):
        if replay_sample["data"]["action"] == "lookUp":
            action = HabitatSimActions.LOOK_UP
        elif replay_sample["data"]["action"] == "lookDown":
            action = HabitatSimActions.LOOK_DOWN
        elif replay_sample["data"]["action"] == "moveForward":
            action = HabitatSimActions.MOVE_FORWARD
        elif replay_sample["data"]["action"] == "moveBackward":
            action = HabitatSimActions.MOVE_BACKWARD
        elif replay_sample["data"]["action"] == "turnLeft":
            action = HabitatSimActions.TURN_LEFT
        elif replay_sample["data"]["action"] == "turnRight":
            action = HabitatSimActions.TURN_RIGHT
        elif replay_sample["data"]["action"] == "grabReleaseObject":
            action = HabitatSimActions.GRAB_RELEASE
        else:
            action = HabitatSimActions.STOP

        return {"action": action}