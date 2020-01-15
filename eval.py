import torch
import os
import argparse
import numpy as np
from utils import runner as dr
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from utils.load_data import HabitatDemoDataset
from models.model_rpf import build_net
from utils.augmentations import BasicAugmentation
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch RPF Training')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--GRU_size', default=512, type=int)
parser.add_argument('--action_dim', default=3, type=int)
parser.add_argument('--demo_length', default=30, type=int)
parser.add_argument('--max_follow_length', default=40, type=int)
parser.add_argument('--memory_dim', default=256, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--feature_dim', default=512, type=int)
parser.add_argument('--data_dir', default='./data/habitat_nav_data_processed/test', type=str)
parser.add_argument('--data_split', default='demo', type=str)
parser.add_argument('--model_name', default='rpf_nuri', type=str)
parser.add_argument('--is_training', default=False, type=bool)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args('')


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

load_name = './outputs/rpf_nuri/best.pth'
DATA_SPLIT = args.data_split
CONTENT_PATH = './data/datasets/pointnav/mp3d/v1/{}/'.format(DATA_SPLIT)
valid_list = np.sort([os.path.join(args.data_dir, DATA_SPLIT, x) for x in os.listdir(os.path.join(args.data_dir, DATA_SPLIT))])
rpf = build_net(args)
statistic = []
rpf = nn.DataParallel(rpf)

rpf.load_state_dict(torch.load(load_name))

if args.cuda:
    rpf.cuda()
    cudnn.benchmark = True

rpf = rpf.eval()


def make_settings():
    settings = dr.default_sim_settings.copy()
    settings["max_frames"] = args.max_follow_length
    settings["width"] = args.img_size
    settings["height"] = args.img_size
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
    statistic = {'success': [], 'spl': []}
    past_house = ''
    valid_dataset = HabitatDemoDataset(args, valid_list, transform=BasicAugmentation(size=args.img_size))
    batch_iterator = iter(data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size))
    for episode in range(len(valid_dataset)):
        demo_im, demo_action_onehot, start_position, start_rotation, end_position, end_rotation, cur_house, data_index, demo_position = next(batch_iterator)
        start_position = start_position.view(-1).detach().numpy()
        start_rotation = start_rotation.view(-1).detach().numpy()
        end_position = end_position.view(-1).detach().numpy()
        end_rotation = end_rotation.view(-1).detach().numpy()
        cur_house = cur_house[0]
        if args.cuda:
            demo_im = Variable(demo_im.cuda(async=True))
            demo_action_onehot = Variable(demo_action_onehot.cuda(async=True))

        settings["scene"] = './data/scene_datasets/mp3d/{}/{}.glb'.format(cur_house, cur_house)
        if past_house != cur_house:
            if past_house != '':
                print('INTERMEDIATE SUCCESS RATE: %3f' % np.mean(statistic['success']))
                print('INTERMEDIATE SPL : %3f' % np.mean(statistic['spl']))
            try:
                demo_runner._sim.close()
            except:
                pass
            demo_runner = dr.DemoRunner(settings, dr.DemoRunnerType.EXAMPLE)
        demo_runner.init_episode(start_position, start_rotation, end_position, end_rotation)
        demo_runner.init_common()
        record, result = rpf(demo_im, demo_action_onehot, start_position=start_position, start_rotation=start_rotation, sim=demo_runner)
        statistic['success'].append(result)
        statistic['spl'].append(result * record['start_end_episode_distance']/max(record['agent_episode_distance'], record['start_end_episode_distance']))
        past_house = cur_house

    print('SUCCESS RATE: %3f' % np.mean(statistic['success']))
    print('SPL : %3f' % np.mean(statistic['spl']))
    np.save('./outputs/{}/stats'.format(args.model_name), statistic)


if __name__ == "__main__":
    settings = make_settings()
    main(settings)