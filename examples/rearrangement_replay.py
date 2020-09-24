import habitat

config = habitat.get_config("configs/tasks/pointnav.yaml")
config.defrost()
config.ENVIRONMENT.MAX_EPISODE_STEPS = 50
config.SIMULATOR.TYPE = "RearrangementSim-v0"
config.SIMULATOR.ACTION_SPACE_CONFIG = "RearrangementActions-v0"
config.SIMULATOR.GRAB_DISTANCE = 2.0
config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
config.TASK.TYPE = "RearrangementTask-v0"
config.TASK.SUCCESS_DISTANCE = 1.0
config.TASK.SENSORS = [
    "INSTRUCTION_SENSOR",
]
config.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
config.TASK.MEASUREMENTS = [
    "DISTANCE_TO_GOAL",
]
config.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "GRAB_RELEASE"]
config.DATASET.TYPE = "RearrangementDataset-v0"
config.DATASET.SPLIT = "train"
config.DATASET.DATA_PATH = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)
config.freeze()

with habitat.Env(config) as env:
    obs = env.reset()
    obs_list = []
    print(env.sim)