import torch.utils.data as data
import numpy as np
import joblib
import torch


class HabitatDemoDataset(data.Dataset):
    def __init__(self, args, data_list, transform=None):
        self.data_list = data_list
        self.img_size = args.img_size
        self.action_dim = args.action_dim
        self.demo_length = args.demo_length
        self.max_follow_length = args.max_follow_length
        self.transform = transform

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def get_dist(self, demo_position):
        return np.linalg.norm(demo_position[-1] - demo_position[0], ord=2)

    def pull_image(self, index):
        demo_data = joblib.load(self.data_list[index])
        data_index = "".join(self.data_list[index].split('_')[4:-1])
        demo_image = demo_data['rgb']
        demo_position = demo_data['position']
        demo_rotation = demo_data['rotation']
        demo_action = demo_data['action']

        scene = self.data_list[index].split('/')[-1].split('_')[0]
        ## Normalize img data
        demo_image = np.array(demo_image, dtype=np.float32)
        if self.transform is not None:
            demo_image = self.transform(demo_image)
        demo_image = demo_image.transpose(0, 3, 1, 2)

        ## Regularizing action data
        demo_action = np.array(demo_action, dtype=np.int8) - 1
        demo_act = np.eye(self.action_dim)[demo_action].astype(np.float32)
        start_position = demo_position[0]
        start_rotation = demo_rotation[0]
        end_position = demo_position[-1]
        end_rotation = demo_rotation[-1]

        return torch.from_numpy(demo_image).float(), torch.from_numpy(demo_act).float(), \
               torch.from_numpy(start_position).float(), torch.from_numpy(start_rotation).float(), \
               torch.from_numpy(end_position).float(), torch.from_numpy(end_rotation).float(), scene, data_index, \
               torch.from_numpy(np.stack(demo_position)).float()


class HabitatDataset(data.Dataset):
    def __init__(self, cfg, data_list, transform=None):
        self.data_list = data_list
        self.img_size = cfg.img_size
        self.action_dim = cfg.action_dim
        self.max_follow_length = cfg.max_follow_length
        self.transform = transform

    def __getitem__(self, index):
        demo_im, demo_act, follower_im, follower_act = self.pull_image(index)
        return demo_im, demo_act, follower_im, follower_act

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        """
        :param args:
        action label : 6 = forward, 7 = rot_left, 8 = rot_right
            --> 1 = forward, 2 = rot_left, 3 = rot_right
        :return: [demo_im, demo_act, follower_im, follower_act, act_mask]
        """
        follower_data = joblib.load(self.data_list[index])

        ## Read matching demon data
        mode = 'train' if 'train' in str(self.data_list[index]) else 'valid'
        demo_data_name = str(self.data_list[index]).replace(mode, 'DEMON')
        demo_data_name = demo_data_name[:demo_data_name.find('.dat.gz') - 1] + '0.dat.gz'
        demo_data = joblib.load(demo_data_name)

        ## Normalize img data
        follower_im = np.array(follower_data['rgb'], dtype=np.float32)
        demo_im = np.array(demo_data['rgb'], dtype=np.float32)
        if self.transform is not None:
            follower_im = self.transform(follower_im)
            demo_im = self.transform(demo_im)
        follower_im = follower_im.transpose(0, 3, 1, 2)
        demo_im = demo_im.transpose(0, 3, 1, 2)

        ## Regularizing action data
        follower_data['action'] = np.array(follower_data['action'], dtype=np.int8) - 1
        demo_data['action'] = np.array(demo_data['action'], dtype=np.int8) - 1
        follower_act = follower_data['action'].astype(np.float32)
        demo_act = np.eye(self.action_dim)[demo_data['action']].astype(np.float32)

        ## fix trial data size to max_follow_length
        follower_im_out = np.zeros([self.max_follow_length, 3, self.img_size, self.img_size])
        follower_act_out = np.ones([self.max_follow_length]) * (-100)
        follower_length = np.minimum(len(follower_im), self.max_follow_length)
        follower_im_out[:follower_length] = follower_im[:follower_length]
        follower_act_out[:follower_length] = follower_act[:follower_length]

        return torch.from_numpy(demo_im).float(), torch.from_numpy(demo_act).float(), \
               torch.from_numpy(follower_im_out).float(), torch.from_numpy(follower_act_out).float()

