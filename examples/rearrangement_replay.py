import habitat

from habitat import Config

config = habitat.get_config("configs/tasks/object_rearrangement.yaml")
config.defrost()
config.ENVIRONMENT.MAX_EPISODE_STEPS = 50
config.SIMULATOR.TYPE = "RearrangementSim-v0"
config.SIMULATOR.ACTION_SPACE_CONFIG = "RearrangementActions-v0"
config.SIMULATOR.CROSSHAIR_POS = [128, 160]
config.SIMULATOR.GRAB_DISTANCE = 2.0
config.SIMULATOR.VISUAL_SENSOR = "rgb"
config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
config.TASK.TYPE = "RearrangementTask-v0"
config.TASK.ACTIONS.GRAB_RELEASE = Config()
config.TASK.ACTIONS.GRAB_RELEASE.TYPE = "GrabOrReleaseAction"
config.TASK.SUCCESS_DISTANCE = 1.0
config.TASK.OBJECT_TO_GOAL_DISTANCE = Config()
config.TASK.OBJECT_TO_GOAL_DISTANCE.TYPE = "ObjectToGoalDistance"
config.TASK.AGENT_TO_OBJECT_DISTANCE = Config()
config.TASK.AGENT_TO_OBJECT_DISTANCE.TYPE = "AgentToObjectDistance"
config.TASK.SENSORS = [
    "INSTRUCTION_SENSOR",
]
config.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
config.DATASET.TYPE = "RearrangementDataset-v0"
config.DATASET.SPLIT = "train"
config.DATASET.DATA_PATH = (
    "data/datasets/object_rearrangement/v1/{split}/{split}.json.gz"
)
config.freeze()

with habitat.Env(config) as env:
    obs = env.reset()
    obs_list = []
