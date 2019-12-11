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
# from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.simulator import Observations
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
import shutil

parser = argparse.ArgumentParser(description='PyTorch RPF Training')
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--GRU-size', default=128, type=int)
parser.add_argument('--action-dim', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr-decay', default=0.5, type=float)
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--demo-length', default=30, type=int)
parser.add_argument('--max-follow-length', default=40, type=int)
parser.add_argument('--memory-dim', default=256, type=int)
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
load_name = './outputs/rpf_nuri/best.pth'
DATA_SPLIT = 'val'
CONTENT_PATH = 'data/datasets/pointnav/mp3d/v1/{}/'.format(DATA_SPLIT)
valid_list = [os.path.join(args.data_dir, 'val', x) for x in os.listdir(os.path.join(args.data_dir, 'val'))]
VIDEO_DIR = os.path.join("/home/blackfoot/datasets/habitat/preprocessed_habitat_data/%s_test_videos" % DATA_SPLIT)
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
rpf = build_net('test', args)
statistic = []

rpf.load_state_dict(torch.load(load_name))
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs")
    rpf = nn.DataParallel(rpf)

if args.cuda:
    rpf.cuda()
    cudnn.benchmark = True

rpf = rpf.eval()


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


def make_settings():
    settings = dr.default_sim_settings.copy()
    settings["max_frames"] = args.max_follow_length
    settings["width"] = 224
    settings["height"] = 224
    settings["scene"] = ''
    settings["save_png"] = False  # args.save_png
    settings["sensor_height"] = 1.5
    settings["color_sensor"] = True
    settings["semantic_sensor"] = False
    settings["depth_sensor"] = False
    settings["print_semantic_scene"] = False
    settings["print_semantic_mask_stats"] = False
    settings["compute_shortest_path"] = False
    settings["compute_action_shortest_path"] = False
    settings["seed"] = 1
    settings["silent"] = False
    return settings


def main(settings):
    if RENDER:
        dirname = os.path.join(VIDEO_DIR)
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        render_dirname = dirname
    statistic = {'success': [], 'lengths': []}
    valid_dataset = HabitatDemoDataset(args, valid_list, transform=BasicAugmentation(size=args.img_size))
    batch_iterator = iter(data.DataLoader(dataset=valid_dataset, batch_size=1))
    for episode in range(len(valid_dataset)):
        demo_im, demo_im_orig, demo_action_onehot, start_position, start_rotation, end_position, end_rotation, cur_house = next(batch_iterator)
        demo_im_orig = demo_im_orig.view(-1, args.img_size, args.img_size, 3).detach().numpy()
        start_position = start_position.view(-1).detach().numpy()
        start_rotation = start_rotation.view(-1).detach().numpy()
        end_position = end_position.view(-1).detach().numpy()
        end_rotation = end_rotation.view(-1).detach().numpy()
        if args.cuda:
            demo_im = Variable(demo_im.cuda(async=True))
            demo_action_onehot = Variable(demo_action_onehot.cuda(async=True))

        settings["scene"] = '/home/blackfoot/datasets/habitat/scene_datasets/mp3d/{}/{}.glb'.format(cur_house[0], cur_house[0])
        demo_runner = dr.DemoRunner(settings, dr.DemoRunnerType.EXAMPLE)
        demo_runner.init_episode(start_position, start_rotation, end_position, end_rotation)
        demo_runner.init_common()
        record, result = rpf(demo_im, demo_action_onehot, start_position=start_position, start_rotation=start_rotation, sim=demo_runner)
        ## 'eta', 'h_ts', 'attention_t', 'in_rgb_feats_t', 'mu_t', 'out_pred_t' ###
        statistic['success'].append(result)
        statistic['lengths'].append(len(record['follower_im']))
        out_gif = []
        if RENDER:
            for t in range(len(record['eta'])):
                out_img = np.zeros([args.img_size, args.img_size*2, 3], dtype=np.uint8)
                memory_img = demo_im_orig[int(record['eta'][t])]
                out_img[:, :args.img_size, :] = memory_img.astype(np.uint8)
                current_img = record['follower_im'][t]
                if record['follower_action'][t] == 0:
                    f_angle = 0
                elif record['follower_action'][t] == 1: #Left
                    f_angle = 1
                elif record['follower_action'][t] == 2: #Right
                    f_angle = -1
                cv2.line(current_img, (int(args.img_size/2), args.img_size), (int(int(args.img_size/2) - 40 * f_angle), args.img_size - 40), (0, 255, 0), 3)
                cv2.putText(current_img, 'length: %02d' % (t), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), lineType=cv2.LINE_AA)
                cv2.putText(current_img, 'dist2goal: %0.3f' % (record['dist2goal'][t]), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), lineType=cv2.LINE_AA)

                if t == len(record['eta']) - 1:
                    cv2.putText(current_img, '%s' % (result), (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), lineType=cv2.LINE_AA)

                out_img[:, args.img_size:, :] = current_img
                out_gif.append(out_img)

            for i in range(3):
                out_gif.append(out_img)

            images_to_video(out_gif, render_dirname, "%s_%02d" % (cur_house[0], episode))
        print('INTERMEDIATE SUCCESS RATE:', np.mean(statistic['success']))
        print('INTERMEDIATE SPL : ', np.mean(np.multiply(statistic['success'], args.max_follow_length / np.maximum(statistic['lengths'], args.max_follow_length))))
        demo_runner._sim.close()
    np.save('./outputs/rpf_nuri/stats', statistic)

    print('SUCCESS RATE:', np.mean(statistic['success']))
    print('SPL : ', np.mean(np.multiply(statistic['success'], args.max_follow_length / np.maximum(statistic['lengths'], args.max_follow_length ))))


if __name__ == "__main__":
    settings = make_settings()
    main(settings)
