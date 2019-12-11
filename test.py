import torch
import torch.nn as nn
import imageio
import os
import cv2
import numpy as np
import argparse
from load_data import HabitatDemoDataset, load_data_demo
from model import build_net
import runner as dr

parser = argparse.ArgumentParser(description='PyTorch RPF Training')
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--GRU-size', default=256, type=int)
parser.add_argument('--action-dim', default=4, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--lr-decay', default=0.997, type=float)
parser.add_argument('--lr-decay-step', default=10, type=int)
parser.add_argument('--wd', default=1e-6, type=float)
parser.add_argument('--demo-length', default=30, type=int)
parser.add_argument('--max-follow-length', default=50, type=int)
parser.add_argument('--memory-dim', default=512, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--img-size', default=256, type=int)
parser.add_argument('--feature-dim', default=512, type=int)
parser.add_argument('--data-dir', default='/home/blackfoot/datasets/habitat/preprocessed_habitat_data', type=str)
parser.add_argument('--model-name', default='rpf_nuri', type=str)
parser.add_argument('--test-step', default=200, type=int)
parser.add_argument('--print-step', default=20, type=int)
parser.add_argument('--gpu-num', default=1, type=int)
parser.add_argument('--is_training', default=False, type=bool)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

load_name = 'outputs/rpf_nuri/best.pth'
valid_list = [os.path.join(args.data_dir, 'valid', x) for x in os.listdir(os.path.join(args.data_dir, 'valid'))]
valid_num = len(valid_list)
valid_dataset = HabitatDemoDataset(args, valid_list)


def make_settings():
    settings = dr.default_sim_settings.copy()
    settings["max_frames"] = 30
    settings["width"] = 256
    settings["height"] = 256
    settings["scene"] = '/home/blackfoot/datasets/habitat/scene_datasets/mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.glb'
    settings["save_png"] = True  # args.save_png
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


rpf = build_net('test', args)
rpf.load_state_dict(torch.load(load_name))
rpf.cuda()
rpf.eval()
ex = 100

with torch.no_grad():
    data = load_data_demo(valid_list[ex], args.action_dim)
    [_demo_rgb, _demo_action, _init_position, _init_rotation, _end_position, _end_rotation] = data
    demo_rgb, demo_action = np.expand_dims(_demo_rgb, 0), np.expand_dims(_demo_action, 0)
    demo_rgb, demo_action = torch.from_numpy(demo_rgb).cuda(), torch.from_numpy(demo_action).cuda()
    settings = make_settings()
    follower_sim = dr.DemoRunner(settings, dr.DemoRunnerType.EXAMPLE)
    perf = follower_sim.init_common(_init_position, _init_rotation, _end_position, _end_rotation)
    record, success = rpf(demo_rgb, demo_action, sim=follower_sim)
    ## 'eta', 'h_ts', 'attention_t', 'in_rgb_feats_t', 'mu_t', 'out_pred_t' ###
    out_gif_1 = []
    out_gif_2 = []
    for t in range(len(record['eta'])):
        memory_img = _demo_rgb[int(record['eta'][t])] * 255
        current_img = record['trial_rgb'][t] #* 255
        if record['trial_action'][t] == 1:
            f_angle = 0
        elif record['trial_action'][t] == 2:
            f_angle = 1
        elif record['trial_action'][t] == 3:
            f_angle = -1
        else:
            f_angle = 0
        cv2.line(current_img, (128, 256), (int(128 - 40 * f_angle), 256 - 40), (0, 255, 0), 3)
        out_gif_1.append(memory_img)
        out_gif_2.append(current_img)

    out_file_name_1 = './outputs/memory_%04d.gif' % (ex)
    imageio.mimsave(out_file_name_1, out_gif_1)
    out_file_name_2 = './outputs/vision_%04d.gif' % (ex)
    imageio.mimsave(out_file_name_2, out_gif_2)
    print("Save ", out_file_name_2)

print("done")

