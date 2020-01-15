# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from enum import Enum
import numpy as np
from PIL import Image
import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim
import habitat_sim.utils as utils
from utils.settings import default_sim_settings, make_cfg
import quaternion as q

_barrier = None


class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2


class DemoRunner:
    def __init__(self, sim_settings, simulator_demo_type):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._shortest_path = hsim.ShortestPath()
        self._cfg = make_cfg(self._sim_settings)
        self._sim = habitat_sim.Simulator(self._cfg)

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    def init_agent_state(self, agent_id):
        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()

        if (start_state.position != self.init_position).any():
            # print('start state position :', start_state.position)
            # print('start state roatation :', start_state.rotation)
            # print('init position:', self.init_position)
            # print('init rotation:', self.init_rotation)
            start_state.position = self.init_position
            start_state.rotation = q.from_float_array(self.init_rotation)
            start_state.sensor_states = dict()  ## Initialize sensor

        agent.set_state(start_state)
        return start_state

    def compute_shortest_path(self, start_pos, end_pos):
        self._shortest_path.requested_start = start_pos
        self._shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(self._shortest_path)
        print("shortest_path.geodesic_distance", self._shortest_path.geodesic_distance)

    def geodesic_distance(self, position_a, position_b):
        self._shortest_path.requested_start = np.array(position_a, dtype=np.float32)
        self._shortest_path.requested_end = np.array(position_b, dtype=np.float32)
        self._sim.pathfinder.find_path(self._shortest_path)
        return self._shortest_path.geodesic_distance

    def euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def step(self, in_action):
        action_names = list(self._cfg.agents[self._sim_settings["default_agent"]].action_space.keys())
        action = action_names[in_action]
        observations = self._sim.step(action)
        color_obs = observations["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA").convert("RGB")
        current_position = self._sim.agents[0].get_state().position
        # self.dist_to_goal = np.linalg.norm(current_position - self.end_position)
        self.dist_to_goal = self.euclidean_distance(current_position, self.end_position)
        self.agent_episode_distance = self.euclidean_distance(current_position, self.previous_position)
        done = False
        if self.dist_to_goal / self.initial_dist_to_goal <= 0.1 or self.dist_to_goal <= 0.8:
            done = True
        self.previous_position = current_position
        return np.asarray(color_img), done

    def get_curposition(self):
        return self._sim.agents[0].get_state().position

    def init_episode(self, init_position, init_rotation, end_position, end_rotation):
        self.init_position = init_position
        self.init_rotation = init_rotation
        self.end_position = end_position
        self.end_rotation = end_rotation

    def init_common(self):
        # self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        if (not os.path.exists(scene_file) and scene_file == default_sim_settings["test_scene"]):
            print("Test scenes not downloaded locally, downloading and extracting now...")
            utils.download_and_unzip(default_sim_settings["test_scene_data_url"], ".")
            print("Downloaded and extracted test scenes data.")

        # self._sim = habitat_sim.Simulator(self._cfg)
        self._sim.reset()
        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

        # initialize the agent at a random start state
        start_state = self.init_agent_state(self._sim_settings["default_agent"])
        self.initial_dist_to_goal = np.linalg.norm(start_state.position - self.end_position)
        self.dist_to_goal = np.linalg.norm(start_state.position - self.end_position)
        self.previous_position = start_state.position
        self.agent_episode_distance = 0

        return start_state

