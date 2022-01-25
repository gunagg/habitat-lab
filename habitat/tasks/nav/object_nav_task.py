# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional

import attr
import numpy as np
from gym import spaces

from collections import defaultdict
from habitat.config import Config
from habitat.core.dataset import SceneState
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Observations, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
    TopDownMap,
)
from habitat.tasks.utils import get_habitat_sim_action

try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass


task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    5,  # ('table', 4, 1)
    6,  # ('picture', 5, 2)
    7,  # ('cabinet', 6, 3)
    8,  # ('cushion', 7, 4)
    10,  # ('sofa', 9, 5),
    11,  # ('bed', 10, 6)
    13,  # ('chest_of_drawers', 12, 7),
    14,  # ('plant', 13, 8)
    15,  # ('sink', 14, 9)
    18,  # ('toilet', 17, 10),
    19,  # ('stool', 18, 11),
    20,  # ('towel', 19, 12)
    22,  # ('tv_monitor', 21, 13)
    23,  # ('shower', 22, 14)
    25,  # ('bathtub', 24, 15)
    26,  # ('counter', 25, 16),
    27,  # ('fireplace', 26, 17),
    33,  # ('gym_equipment', 32, 18),
    34,  # ('seating', 33, 19),
    38,  # ('clothes', 37, 20),
    43,  # ('foodstuff', 42, 21),
    44,  # ('stationery', 43, 22),
    45,  # ('fruit', 44, 23),
    46,  # ('plaything', 45, 24),
    47,  # ('hand_tool', 46, 25),
    48,  # ('game_equipment', 47, 26),
    49,  # ('kitchenware', 48, 27)
]

mapping_mpcat40_to_goal21 = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}


@attr.s(auto_attribs=True, kw_only=True)
class AgentStateSpec:
    r"""Agent data specifications that capture states of agent and sensor in replay state.
    """
    position: Optional[List[float]] = attr.ib(default=None)
    rotation: Optional[List[float]] = attr.ib(default=None)
    sensor_data: Optional[dict] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ReplayActionSpec:
    r"""Replay specifications that capture metadata associated with action.
    """
    action: str = attr.ib(default=None, validator=not_none_validator)
    agent_state: Optional[AgentStateSpec] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    reference_replay: Optional[List[ReplayActionSpec]] = None
    scene_state: Optional[List[SceneState]] = None
    is_thda: Optional[bool] = False

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.GOAL_SPEC_MAX_VAL - 1
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


@registry.register_task(name="ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """


@registry.register_sensor(name="DemonstrationSensor")
class DemonstrationSensor(Sensor):
    def __init__(self, **kwargs):
        self.uuid = "demonstration"
        self.observation_space = spaces.Discrete(0)
        self.timestep = 0
        self.prev_action = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):

        # Fetch next action as observation
        if task.is_resetting:  # reset
            self.timestep = 1
            # logger.info("Episode start: {}".format(episode.episode_id))
        
        if self.timestep < len(episode.reference_replay):
            action_name = episode.reference_replay[self.timestep].action
            action = get_habitat_sim_action(action_name)
            # logger.info("{} -- {}".format(self.timestep, action_name))
        else:
            action = 0
            # logger.info("{} -- {}".format(self.timestep, "STOP"))

        # logger.info("{} -- {}".format(self.timestep, action))
        self.timestep += 1
        return action

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="InflectionWeightSensor")
class InflectionWeightSensor(Sensor):
    def __init__(self, config: Config, **kwargs):
        self.uuid = "inflection_weight"
        self.observation_space = spaces.Discrete(0)
        self._config = config
        self.timestep = 0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode,
        task: EmbodiedTask,
        **kwargs
    ):
        if task.is_resetting:  # reset
            # logger.info("Reset: {}, lim: {}".format(self.timestep, len(episode.reference_replay)))
            self.timestep = 0
        
        inflection_weight = 1.0
        # logger.info("Ts: {}, lim: {}".format(self.timestep, len(episode.reference_replay)))
        if self.timestep == 0:
            inflection_weight = 1.0
        elif self.timestep >= len(episode.reference_replay):
            # logger.info("EP: {} Ts: {}, lim: {}".format(episode.episode_id, self.timestep, len(episode.reference_replay)))
            inflection_weight = 1.0 
        elif episode.reference_replay[self.timestep - 1].action != episode.reference_replay[self.timestep].action:
            inflection_weight = self._config.INFLECTION_COEF
        self.timestep += 1
        return inflection_weight

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_measure
class GoalObjectVisible(Measure):
    r"""GoalObjectVisible"""
    cls_uuid = "goal_vis_pixels"

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.task_cat2mpcat40 = task_cat2mpcat40

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.update_metric(
            episode=episode, task=task, *args, **kwargs
        )

    def update_metric(
        self,
        episode,
        task: EmbodiedTask,
        observations,
        *args: Any,
        **kwargs: Any,
    ):
        self._metric = 0
        if "semantic" in observations:
            semantic_obs = observations["semantic"]

            if self._config.INSERTED_OBJECTS:
                object_ids = self._sim.get_existing_object_ids()
                goal_visible_pixels = 0
                semantic_object_id = self._sim.obj_id_to_semantic_obj_id_map[0]
                # If object is gripped caclulate visible pixels for receptacle
                if observations["gripped_object_id"] != -1:
                    semantic_object_id = self._sim.obj_id_to_semantic_obj_id_map[1]
                
                goal_visible_pixels += (semantic_obs == semantic_object_id).sum() # Sum over all since we're not batched
                goal_visible_area = goal_visible_pixels / semantic_obs.size
                self._metric = goal_visible_area
            else:
                # permute tensor to dimension [CHANNEL x HEIGHT X WIDTH]
                idx = self.task_cat2mpcat40[
                    observations["objectgoal"][0]
                ]  # task._dataset.category_to_task_category_id[episode.object_category], task._dataset.category_to_scene_annotation_category_id[episode.object_category], observations["objectgoal"][0]

                goal_visible_pixels = (semantic_obs == idx).sum() # Sum over all since we're not batched
                goal_visible_area = goal_visible_pixels / semantic_obs.size
                self._metric = goal_visible_area


@registry.register_measure
class RoomVisitationMap(Measure):
    """Semantic exploration measure."""

    cls_uuid: str = "room_visitation_map"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.goal_rooms = defaultdict(list)
        self.room_aabbs = defaultdict(list)
        self.goal_room_visitation_map = defaultdict(int)
        self.room_visitation_map = defaultdict(int)

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return RoomVisitationMap.cls_uuid

    def aabb_contains(self, position, aabb):
        aabb_min = aabb.min()
        aabb_max = aabb.max()
        if aabb_min[0] <= position[0] and aabb_max[0] >= position[0] and aabb_min[2] <= position[2] and aabb_max[2] >= position[2] and aabb_min[1] <= position[1] and aabb_max[1] >= position[1]:
            return True
        return False

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)
        self.room_aabbs = defaultdict(list)
        self.room_visitation_map = defaultdict(int)
        self.goal_room_visitation_map = defaultdict(int)
        self.goal_rooms = defaultdict(list)

        semantic_scene = self._sim.semantic_scene
        for region in semantic_scene.regions:
            region_name = region.category.name()
            aabb = region.aabb
            self.room_aabbs[region_name].append(aabb)

            contains = False
            for goal in episode.goals:
                goal_position = np.array(goal.position)
                if hasattr(goal, "view_points"):
                    goal_position = np.array(goal.view_points[0].agent_state.position)
                if self.aabb_contains(goal_position, aabb):
                    contains = True
            
            if contains:
                self.goal_rooms[region_name].append(aabb)

        self._metric = self.room_visitation_map

    def update_metric(self, episode, *args: Any, **kwargs: Any):        
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        
        for region, room_aabbs in self.room_aabbs.items():
            for aabb in room_aabbs:
                if self.aabb_contains(agent_position, aabb):
                    self.room_visitation_map[region] += 1

        for region, room_aabbs in self.goal_rooms.items():
            for aabb in room_aabbs:
                if self.aabb_contains(agent_position, aabb):
                    self.goal_room_visitation_map[region] += 1
        
        self._metric =  {
            "time_spent_goal_room": sum(self.goal_room_visitation_map.values()),
            "room_visitation_map": self.room_visitation_map,
        }


@registry.register_measure
class ExplorationMetrics(Measure):
    """Semantic exploration measure."""

    cls_uuid: str = "exploration_metrics"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self.room_aabbs = defaultdict(list)
        self.previous_room_stack = []
        self.steps_between_rooms = 0
        self.room_visitation_map = defaultdict(int)
        self.room_revisitation_map = defaultdict(int)
        self.room_revisitation_map_strict = defaultdict(int)
        self.last_20_actions = []
        self.total_left_turns = 0
        self.total_right_turns = 0
        self.panoramic_turns = 0
        self.panoramic_turns_strict = 0
        self.delta_sight_coverage = 0
        self.prev_sight_coverage = 0
        self.avg_delta_coverage = 0

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ExplorationMetrics.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [TopDownMap.cls_uuid]
        )
        # self.update_metric(*args, episode=episode, task=task, action={"action": 0}, **kwargs)
        self.room_aabbs = defaultdict(list)
        self.room_visitation_map = defaultdict(int)
        self.room_revisitation_map = defaultdict(int)
        self.room_revisitation_map_strict = defaultdict(int)
        semantic_scene = self._sim.semantic_scene
        current_room = None
        self.steps_between_rooms = 0
        agent_state = self._sim.get_agent_state()
        self.last_20_actions = []
        self.total_left_turns = 0
        self.total_right_turns = 0
        self.panoramic_turns = 0
        self.panoramic_turns_strict = 0
        self.delta_sight_coverage = 0
        self.prev_sight_coverage = 0
        self.avg_delta_coverage = 0
        i = 0
        for level in semantic_scene.levels:
            for region in level.regions:
                region_name = region.category.name()
                if "bedroom" in region_name:
                    region_name = region.category.name() + "_{}".format(i)
                aabb = region.aabb
                self.room_aabbs[region_name].append(aabb)

                if self.aabb_contains(agent_state.position, aabb):
                    current_room = region_name
                i += 1

        self._metric = self.room_revisitation_map
        self.previous_room = current_room
    
    def aabb_contains(self, position, aabb):
        aabb_min = aabb.min()
        aabb_max = aabb.max()
        if aabb_min[0] <= position[0] and aabb_max[0] >= position[0] and aabb_min[2] <= position[2] and aabb_max[2] >= position[2] and aabb_min[1] <= position[1] and aabb_max[1] >= position[1]:
            return True
        return False
    
    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )
    
    def _is_peeking(self, current_room):
        prev_prev_room = None
        prev_room = None
        if len(self.previous_room_stack) >= 2:
            prev_prev_room = self.previous_room_stack[-2]
        if len(self.previous_room_stack) >= 1:
            prev_room = self.previous_room_stack[-1]
        
        if prev_prev_room is not None and prev_room is not None:
            if prev_prev_room == current_room and prev_room != current_room:
                return True
        return False
    

    def get_coverage(self, info):
        if info is None:
            return 0
        top_down_map = info["map"]
        visted_points = np.where(top_down_map <= 9, 0, 1)
        coverage = np.sum(visted_points) / self.get_navigable_area(info)
        return coverage


    def get_navigable_area(self, info):
        if info is None:
            return 0
        top_down_map = info["map"]
        navigable_area = np.where(((top_down_map == 1) | (top_down_map >= 10)), 1, 0)
        return np.sum(navigable_area)


    def get_visible_area(self, info):
        if info is None:
            return 0
        fog_of_war_mask = info["fog_of_war_mask"]
        visible_area = fog_of_war_mask.sum() / self.get_navigable_area(info)
        if visible_area > 1.0:
            visible_area = 1.0
        return visible_area

    def is_beeline(self):
        count_move_forwards = 0
        max_move_forwards = 0
        for action in self.last_20_actions:
            if action != "MOVE_FORWARD":
                count_move_forwards = 0
            else:
                count_move_forwards += 1
            max_move_forwards = max(max_move_forwards , count_move_forwards)
        return (max_move_forwards / len(self.last_20_actions)) >= 0.5 

    def update_metric(self, episode, task, action, *args: Any, **kwargs: Any):        
        top_down_map = task.measurements.measures[
            TopDownMap.cls_uuid
        ].get_metric()
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        
        current_room = None
        all_rooms = []
        for region, room_aabbs in self.room_aabbs.items():
            for aabb in room_aabbs:
                if self.aabb_contains(agent_position, aabb):
                    self.room_visitation_map[region] += 1
                    current_room = region
                    all_rooms.append(current_room)

        if self._is_peeking(current_room) and self.room_visitation_map[current_room] >= 1 and self.steps_between_rooms <= 10:
            # Count total visits to the room
            if self.room_revisitation_map[current_room] == 0:
                self.room_revisitation_map[current_room] += 1
            self.room_revisitation_map[current_room] += 1
        
        if self._is_peeking(current_room) and self.room_visitation_map[current_room] >= 1 and self.steps_between_rooms >= 8 and self.steps_between_rooms <= 14:
            # Count total visits to the room
            if self.room_revisitation_map_strict[current_room] == 0:
                self.room_revisitation_map_strict[current_room] += 1
            self.room_revisitation_map_strict[current_room] += 1
        
        if (len(self.previous_room_stack) == 0 or self.previous_room_stack[-1] != current_room) and current_room is not None:
            self.previous_room_stack.append(current_room)
            self.steps_between_rooms = 0

        self.steps_between_rooms += 1
        # print(top_down_map)
        self.coverage = self.get_coverage(top_down_map)
        self.sight_coverage = self.get_visible_area(top_down_map)

        self.delta_sight_coverage = self.sight_coverage - self.prev_sight_coverage
        self.prev_sight_coverage = self.sight_coverage

        self.last_20_actions.append(task.get_action_name(action["action"]))
        if len(self.last_20_actions) > 20:
            self.last_20_actions.pop(0)
        if "TURN" not in task.get_action_name(action["action"]):
            self.total_left_turns = 0
            self.total_right_turns = 0
            self.delta_sight_coverage = 0
        else:
            if task.get_action_name(action["action"]) == "TURN_LEFT":
                self.total_left_turns += 1
            elif task.get_action_name(action["action"]) == "TURN_RIGHT":
                self.total_right_turns += 1
        if self.total_left_turns >= 3 and self.total_right_turns >= 3 and (self.total_right_turns + self.total_left_turns) >= 8 and self.delta_sight_coverage > 0.015:
            self.panoramic_turns += 1

        if self.total_left_turns >= 3 and self.total_right_turns >= 3 and (self.total_right_turns + self.total_left_turns) >= 8 and self.delta_sight_coverage > 0.01:
            self.panoramic_turns_strict += 1
            self.avg_delta_coverage += self.delta_sight_coverage
    
        avg_cov = 0
        if self.panoramic_turns_strict > 0:
            avg_cov = self.avg_delta_coverage / self.panoramic_turns_strict

        self._metric = {
            "room_revisitation_map": self.room_revisitation_map,
            "coverage": self.coverage,
            "sight_coverage": self.sight_coverage,
            "panoramic_turns": self.panoramic_turns,
            "panoramic_turns_strict": self.panoramic_turns_strict,
            "beeline": self.is_beeline(),
            "last_20_actions": self.last_20_actions,
            "room_revisitation_map_strict": self.room_revisitation_map_strict,
            "delta_sight_coverage": self.delta_sight_coverage,
            "avg_delta_coverage": avg_cov
        }
