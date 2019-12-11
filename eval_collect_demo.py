import torch
import torch.nn as nn
import imageio
import os
import cv2
import numpy as np
import argparse
import runner as dr
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from load_data import HabitatDemoDataset
from model import build_net
from utils.augmentations import BasicAugmentation
from torch.autograd import Variable
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.simulator import Observations
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
import quaternion as q
import shutil

parser = argparse.ArgumentParser(description='PyTorch RPF Training')
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--GRU-size', default=128, type=int)
parser.add_argument('--action-dim', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr-decay', default=0.5, type=float)
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--demo-length', default=30, type=int)
parser.add_argument('--max-follow-length', default=70, type=int)
parser.add_argument('--memory-dim', default=256, type=int)
parser.add_argument('--max-iter', default=120000, type=int)
parser.add_argument('--img-size', default=224, type=int)
parser.add_argument('--feature-dim', default=256, type=int)
parser.add_argument('--data-dir', default='/home/blackfoot/datasets/habitat/preprocessed_habitat_data', type=str)
parser.add_argument('--model-name', default='rpf_nuri', type=str)
parser.add_argument('--test-step', default=200, type=int)
parser.add_argument('--print-step', default=20, type=int)
parser.add_argument('--gpu-num', default=1, type=int)
parser.add_argument('--is-training', default=False, type=bool)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args('')

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

RENDER = True
DATA_SPLIT = 'val'
CONTENT_PATH = 'data/datasets/pointnav/mp3d/v1/{}/'.format(DATA_SPLIT)
VIDEO_DIR = os.path.join("/home/blackfoot/datasets/habitat/preprocessed_habitat_data/%s_videos" % DATA_SPLIT)
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config):
        super().__init__(config)
        self.follower = ShortestPathFollower(self.habitat_env.sim, 0.25, False)

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def get_agent_state(self):
        curr_state = self.habitat_env.sim.get_agent_state()
        return [curr_state.position, q.as_float_array(curr_state.rotation)]

    def get_best_act(self):
        return self.follower.get_next_action(self.habitat_env.current_episode.goals[0].position)

    def get_episode_over(self):
        return self.habitat_env.episode_over

    def get_goal(self):
        return self.habitat_env.current_episode.goals[0].position

    def get_distance(self):
        # dist = np.linalg.norm(np.array(end_pos)[:2]- np.array(goal_pos)[:2])
        return np.linalg.norm(self.habitat_env._sim.agents[0].get_state().position[:2] - self.habitat_env.current_episode.goals[0].position[:2])

    def reset_curr_episode(self) -> Observations:
        self._env._reset_stats()
        assert len(self.episodes) > 0, "Episodes list is empty"
        self._env.sim.reconfigure(self._env._config.SIMULATOR)
        observations = self.habitat_env._sim.reset()
        observations.update(
            self._env.task.sensor_suite.get_observations(
                observations=observations, episode=self.current_episode
            )
        )
        self._env._task.measurements.reset_measures(episode=self._env.current_episode)
        return observations

    def reconfigure(self, config):
        return self.habitat_env.sim.reconfigure(config.SIMULATOR)


def main(config):
    envs = SimpleRLEnv(config)
    save_dir = os.path.join("./data/preprocessed_habitat_data")
    if RENDER:
        dirname = os.path.join(VIDEO_DIR)
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        render_dirname = dirname
    for episode in range(len(envs.episodes)):
        try:
            obs = envs.reset()
        except:
            envs.close()
            envs = SimpleRLEnv(config)
            obs = envs.reset()
        cur_house = envs.current_episode.scene_id.split('/')[-2]
        episode_name = '%s/%s_%03d'%(DATA_SPLIT, cur_house, episode)
        data = {'image': [], 'position': [], 'rotation': [], 'action': []}
        images = []
        step = 0
        done = envs.get_episode_over()
        while not done:
            best_actions = envs.get_best_act()
            past_obs = obs
            curr_states = envs.get_agent_state()
            if best_actions is None:
                break
            obs, reward, done, info = envs.step(best_actions)
            data['image'].append(past_obs['rgb'])
            data['position'].append(curr_states[0])
            data['rotation'].append(curr_states[1])
            data['action'].append(best_actions)
            if RENDER:
                top_down_map = draw_top_down_map(info, obs["heading"], obs["rgb"].shape[0])
                output_im = np.concatenate((obs["rgb"], top_down_map), axis=1)
                images.append(output_im)
            step += 1

        if step >= 30:
            if RENDER:
                images_to_video(images, render_dirname, "%s_%02d" % (cur_house, episode))
            np.save(os.path.join(save_dir, episode_name + '.npy'), data)


if __name__ == "__main__":
    config = habitat.get_config(config_paths="./configs/pathfollow_mp3d.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.SIMULATOR.TURN_ANGLE = 30
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.4
    config.DATASET.SPLIT = DATA_SPLIT
    config.DATASET.DATA_PATH = CONTENT_PATH + os.listdir(CONTENT_PATH)[0]
    config.freeze()
    main(config)
