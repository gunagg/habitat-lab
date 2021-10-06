# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

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
    _is_episode_active: bool
    _prev_action: int

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False

    def _check_episode_is_active(self,  action, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


@registry.register_task(name="ObjectNavAddedObj-v1")
class ObjectNavAddedObj(ObjectNavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from habitat.datasets.object_nav.create.create_objectnav_dataset_with_added_objects import (
            read_ycb_google_objects,
        )

        self.ycb_objects_lib, _ = read_ycb_google_objects(
            "habitat/datasets/object_nav/create/ycb_google_16k_19_objs_meta.csv"
        )
        self.selected_objects = None
        self.episode_counter = 0
        self.prev_episode = None
        self.is_generated_episode = False

        from habitat.datasets.object_nav.create.create_objectnav_dataset_with_added_objects import (
            category_to_mp3d_category_id,
            category_to_task_category_id,
        )

        self._dataset.category_to_scene_annotation_category_id = (
            category_to_mp3d_category_id
        )
        self._dataset.category_to_task_category_id = (
            category_to_task_category_id
        )

        self.near_dist = (
            self._config.OBJ_GEN_NEAR_DIST
            if hasattr(self._config, "OBJ_GEN_NEAR_DIST")
            else 1
        )
        self.far_dist = (
            self._config.OBJ_GEN_FAR_DIST
            if hasattr(self._config, "OBJ_GEN_FAR_DIST")
            else 5
        )

    def overwrite_sim_config(self, sim_config, episode):
        from habitat.datasets.object_nav.create.create_objectnav_dataset_with_added_objects import (
            sample_objects_for_episode,
            sample_scene_state,
        )

        super().overwrite_sim_config(sim_config, episode)

        # if hasattr(episode, "scene_state") and episode.scene_state is not None:
        sim_config.defrost()
        self.selected_objects = sample_objects_for_episode(
            objects_lib=self.ycb_objects_lib,
            num_uniq_selected_objects=self._config.NUM_UNIQ_SELECTED_OBJECTS,
            num_copies=self._config.NUM_OBJECT_COPIES,
        )
        # logger.info(f"new selected_objects {len(self.selected_objects)}")
        sim_config.scene_state = [
            sample_scene_state(self.selected_objects).__dict__
        ]
        sim_config.freeze()

        if (
            not self.is_generated_episode
        ):  # Agent position isn't important if episode wasn't generated on the fly
            sim_config.defrost()
            agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
            agent_cfg = getattr(sim_config, agent_name)
            agent_cfg.IS_SET_START_STATE = False
            sim_config.freeze()

        return sim_config

    def reset(self, episode):
        from habitat.datasets.object_nav.create.create_objectnav_dataset_with_added_objects import (
            category_to_mp3d_category_id,
            category_to_task_category_id,
            reset_episode,
        )

        original_episode = episode
        episode_with_goal = reset_episode(
            episode=episode,
            sim=self._sim,
            selected_objects=self.selected_objects,
            near_dist=self.near_dist,
            far_dist=self.far_dist,
            episode_counter=self.episode_counter,
        )
        self.episode_counter += 1
        
        if episode_with_goal is None:
            # Clone states from previous episode
            episode.start_position = self.prev_episode.start_position
            episode.start_rotation = self.prev_episode.start_rotation
            episode.goals = self.prev_episode.goals
            episode.scene_state = self.prev_episode.scene_state
            episode.object_category = self.prev_episode.object_category

            if episode is None:
                episode = original_episode
                logger.error(
                    f"Prev episode doesn't exist, falling back to original episode {episode.scene_id}"
                )
            episode.episode_id = str(
                self.episode_counter - 1
            )  # set unique id even if episode is old
            # raise BaseException #used to test for scenes that can't generate episode
        else:
            episode = episode_with_goal

        self.prev_episode = episode
        self.is_generated_episode = True
        sim_config = super().overwrite_sim_config(
            self._sim.habitat_config, episode
        )

        self._sim.reconfigure(sim_config)
        self.is_generated_episode = False
        return super().reset(episode)
