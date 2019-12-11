import torch
import torch.nn as nn
import numpy as np
import runner as dr
import quaternion as q
from torch.autograd import Variable
from utils.augmentations import BasicAugmentation


class FCN_2layer(nn.Module):
    def __init__(self, input_ch, hidden_ch, output_ch, activation=None):
        super(FCN_2layer, self).__init__()
        self.layer1 = nn.Linear(input_ch, hidden_ch, bias=True)
        self.layer2 = nn.Linear(hidden_ch, output_ch, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == None:
            self.activation = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        if self.activation == None:
            return x
        else:
            return self.activation(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = BatchNorm(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        self.img_size = args.img_size
        self.out_size = int(self.img_size / 2 ** 5)

        self.conv1 = BasicConv(3, 32, kernel_size=3, padding=1)
        self.conv2 = BasicConv(32, 64, kernel_size=3, padding=1)
        self.conv3 = BasicConv(64, 128, kernel_size=3, padding=1)
        self.conv4 = BasicConv(128, 256, kernel_size=3, padding=1)
        self.conv5 = BasicConv(256, 512, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * self.out_size * self.out_size, args.feature_dim * 2, bias=True)
        self.fc2 = nn.Linear(args.feature_dim * 2, args.feature_dim, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(GRUNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x, h):
        out, h = self.gru(x.view(-1, self.n_layers, self.input_dim), h)
        return h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden


class RPF(nn.Module):
    def __init__(self, args):
        super(RPF, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.record = {'actions_t': [], 'eta': [], 'h_ts': [], 'attention_t': [], 'inp_feats_t': [], 'mu_t': []}
        self.follow_length = args.max_follow_length
        self.GRU_size = args.GRU_size
        self.demo_length = args.demo_length
        self.img_size = args.img_size
        self.action_dim = args.action_dim
        self.memory_dim = args.memory_dim
        self.feature_dim = args.feature_dim
        self.is_training = args.is_training
        self.curr_h_t = None
        self.curr_eta = None
        self.curr_memories = None
        self.max_eta = torch.mul(torch.ones([self.batch_size, 1], requires_grad=False), 29).cuda()

        self.encode_im = CNN(self.args)
        self.visual_memory_fc = FCN_2layer((self.feature_dim + self.action_dim), 512, self.memory_dim, activation='relu') ## M
        # self.GRU = nn.GRUCell(self.memory_dim+self.feature_dim, self.GRU_size)
        self.GRU = GRUNet(self.memory_dim+self.feature_dim, self.GRU_size, 1)
        self.GRU_attention_fc = FCN_2layer(self.GRU_size, 512, 1, activation='tanh')
        self.GRU_out_fc = FCN_2layer(self.GRU_size, 512, self.action_dim)
        self.transform = BasicAugmentation()
        print('Building RPF networks for training...')
        if self.is_training:
            self.initialize()

    # def init_GRU_hidden(self, batch_size):
    #     with torch.no_grad():
    #         h = Variable(torch.zeros(batch_size, self.GRU_size))
    #     return h

    def encode_visual_memory(self, demo_im, demo_act):
        memory_list = []
        for t in range(self.demo_length):
            demo_im_t = demo_im[:, t]  # B X 3 X img X img
            demo_act_t = demo_act[:, t]  # B X action_dim
            demo_im_feats_t = self.encode_im(demo_im_t)
            memory_input = torch.cat([demo_im_feats_t, demo_act_t], -1)
            memory_t = self.visual_memory_fc(memory_input)
            memory_list.append(memory_t)

        return torch.stack(memory_list, 1)

    def forward(self, demo_im, demo_act, sim=None, start_position=None, start_rotation=None, follower_im=None):
        record = {'eta': [], 'h_ts': [], 'attention_t': [], 'follower_im_feats_t': [], 'mu_t': [],
                  'action_pred_t': [], 'follower_im': [], 'follower_action': [], 'dist2goal': [], 'rot2goal': []}
        memories_list = self.encode_visual_memory(demo_im, demo_act)
        batch_size = len(demo_im)
        h_t = self.GRU.init_hidden(batch_size)
        eta = torch.zeros([batch_size, 1]).cuda()

        if self.is_training:
            for t in range(self.follow_length):
                record['h_ts'].append(h_t)
                record['eta'].append(eta)
                attention_t = torch.stack([torch.exp(-torch.abs(eta - j)) for j in range(self.demo_length)], 1)
                record['attention_t'].append(attention_t)
                mu_t = torch.sum(torch.mul(memories_list, attention_t), dim=1)
                record['mu_t'].append(mu_t)

                follower_im_feats_t = self.encode_im(follower_im[:, t])
                record['follower_im_feats_t'].append(follower_im_feats_t)
                gru_in_t = torch.cat([follower_im_feats_t, mu_t], -1).view(batch_size, self.memory_dim + self.feature_dim)
                h_t = self.GRU(gru_in_t, h_t)
                b = 1 + self.GRU_attention_fc(h_t.view(batch_size, self.GRU_size))
                eta = torch.min(eta + b, self.max_eta)
                action_pred_t = self.GRU_out_fc(h_t.view(batch_size, self.GRU_size))
                record['action_pred_t'].append(action_pred_t)
            record['action_pred_t'] = torch.stack(record['action_pred_t'], 1)
        else:
            for t in range(self.follow_length):
                record['h_ts'].append(h_t)
                record['eta'].append(eta)
                attention_t = torch.stack([torch.exp(-torch.abs(eta - j)) for j in range(self.demo_length)], 1)
                record['attention_t'].append(attention_t)
                mu_t = torch.sum(torch.mul(memories_list, attention_t), dim=1)
                record['mu_t'].append(mu_t)

                if t == 0:
                    follower_im_t = demo_im[:, 0]
                    record['follower_im'].append(follower_im_t.view(3, self.img_size, self.img_size).permute(1, 2, 0).detach().cpu().numpy())
                    done = False
                else:
                    cur_rgb, done = sim.step(action)
                    record['follower_im'].append(cur_rgb)
                    cur_rgb = np.reshape(cur_rgb, [1, self.img_size, self.img_size, 3])
                    follower_im_t = torch.from_numpy(self.transform(cur_rgb)).float().cuda().permute(0, 3, 1, 2)

                follower_im_feats_t = self.encode_im(follower_im_t)
                record['follower_im_feats_t'].append(follower_im_feats_t)
                gru_in_t = torch.cat([follower_im_feats_t, mu_t], -1).view(batch_size, self.memory_dim + self.feature_dim)
                h_t = self.GRU(gru_in_t, h_t)
                b = 1 + self.GRU_attention_fc(h_t.view(batch_size, self.GRU_size))
                eta = torch.min(eta + b, self.max_eta)
                action_pred_t = self.GRU_out_fc(h_t.view(batch_size, self.GRU_size))
                record['action_pred_t'].append(action_pred_t)
                action = int(torch.argmax(action_pred_t).cpu().numpy())
                record['follower_action'].append(action)
                record['dist2goal'].append(sim.dist_to_goal)

                if done:
                    return record, 1

        return record, 0

    def action_pred_for_run(self, follower_im, h, eta, memories_list):
        attention = torch.stack([torch.exp(-abs(eta - float(j))) for j in range(self.demo_length)], 1)
        mu = torch.sum(torch.mul(memories_list, attention), dim=1)
        follower_im_feats = self.encode_im(follower_im)
        gru_in = torch.reshape(torch.cat([follower_im_feats, mu], -1), [self.batch_size, self.memory_dim + self.feature_dim])
        gru_out, h = self.GRU(gru_in, h)
        eta = eta + (1 + self.GRU_attention_fc(gru_out))
        eta = torch.min(eta, self.demo_length-1)

        return gru_out, mu, eta

    def initialize(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 and classname.find("BasicConv") == -1:
                m.weight.data.normal_(0.0, 0.02)
                try:
                    m.bias.data.fill_(0.001)
                except:
                    pass
                print("Initialized {}".format(m))
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                print("Initialized {}".format(m))
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0.001)
                print("Initialized {}".format(m))
            elif classname.find('GRU') != -1 and classname.find('GRUNet') == -1:
                m.weight_hh_l0.data.normal_(0.0, 0.02)
                m.weight_ih_l0.data.normal_(0.0, 0.02)
                try:
                    m.bias_hh_l0.data.fill_(0.001)
                    m.bias_ih_l0.data.fill_(0.001)
                except:
                    pass
            # elif classname.find('GRU') != -1:
            #     m.weight_hh.data.normal_(0.0, 0.02)
            #     m.weight_ih.data.normal_(0.0, 0.02)
            #     try:
            #         m.bias_hh.data.fill_(0.001)
            #         m.bias_ih.data.fill_(0.001)
            #     except:
            #         pass
                print("Initialized {}".format(m))

        self.encode_im.apply(weights_init)
        self.visual_memory_fc.apply(weights_init)
        self.GRU.apply(weights_init)
        # init.xavier_normal(self.GRU)
        self.GRU_attention_fc.apply(weights_init)
        self.GRU_out_fc.apply(weights_init)


def build_net(phase, args):
    # if phase == 'train':
    return RPF(args)