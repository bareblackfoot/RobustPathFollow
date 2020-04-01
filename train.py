import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils.load_data import HabitatDataset
from torch.autograd import Variable
from models.model_rpf import build_net
from utils.augmentations import RPFAugmentation

parser = argparse.ArgumentParser(description='PyTorch RPF Training')
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--GRU_size', default=64, type=int)
parser.add_argument('--action_dim', default=3, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_decay', default=0.5, type=float)
parser.add_argument('--wd', default=1e-5, type=float)
parser.add_argument('--demo_length', default=30, type=int)
parser.add_argument('--max_follow_length', default=40, type=int)
parser.add_argument('--memory_dim', default=128, type=int)
parser.add_argument('--max_iter', default=120000, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--feature_dim', default=128, type=int)
parser.add_argument('--data_dir', default='./data/pathfollow', type=str)
parser.add_argument('--model_name', default='default', type=str)
parser.add_argument('--test_step', default=5000, type=int)
parser.add_argument('--print_step', default=20, type=int)
parser.add_argument('--is_training', default=True, type=bool)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--cuda', default=True, type=bool)
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_dir = 'outputs/' + args.model_name
log_dir = 'experiments/tb_logs/' + args.model_name
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir)

train_list = [os.path.join(args.data_dir, 'train', x) for x in os.listdir(os.path.join(args.data_dir, 'train'))]
valid_list = [os.path.join(args.data_dir, 'valid', x) for x in os.listdir(os.path.join(args.data_dir, 'valid'))]

train_num, valid_num = len(train_list), len(valid_list)

num_batches = int(train_num / args.batch_size)
valid_num_batches = int(valid_num / args.batch_size)

train_dataset = HabitatDataset(args, train_list, transform=RPFAugmentation(size=args.img_size))
valid_dataset = HabitatDataset(args, valid_list, transform=RPFAugmentation(size=args.img_size))

rpf = build_net('train', args)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs")
    rpf = nn.DataParallel(rpf)

if args.cuda:
    rpf.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
logsoftmax = nn.LogSoftmax()
optimizer = optim.Adam(rpf.parameters(), lr=args.lr, weight_decay=args.wd)


def train():
    rpf.train()
    lr = args.lr
    epoch = disp_loss = 0
    eval_loss = 10000.
    start_time = time.time()
    epoch_size = len(train_dataset) // args.batch_size
    max_epoch = int(args.max_iter/epoch_size)
    step_values = [10000, 50000, 100000]
    step_index = 0
    batch_iterator = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    for iteration in range(args.max_iter):
        if iteration % epoch_size == 0:
            if epoch != 0:
                print("Saving state, epoch:", epoch)
                torch.save(rpf.state_dict(), save_dir + '/epoch_{}.pth'.format(epoch))
            batch_iterator = iter(data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers))
            epoch += 1
        demo_im, demo_action_onehot, follower_im, follower_action = next(batch_iterator)

        if args.cuda:
            demo_im = Variable(demo_im.cuda(async=True))
            demo_action_onehot = Variable(demo_action_onehot.cuda(async=True))
            follower_im = Variable(follower_im.cuda(async=True))
            with torch.no_grad():
                follower_action = Variable(follower_action.cuda(async=True).long())

        record, done = rpf(demo_im, demo_action_onehot, follower_im=follower_im) #Record: 'eta', 'h_ts', 'attention_t', 'in_rgb_feats_t', 'mu_t', 'action_pred_t'
        optimizer.zero_grad()
        loss = criterion(record['action_pred_t'].view(-1, args.action_dim), follower_action.view(-1))
        loss.backward()
        optimizer.step()
        disp_loss += loss.item()

        end_time = time.time()
        if iteration % args.print_step == 0 and iteration > 0:
            disp_loss = disp_loss / args.print_step
            print('Epoch [%d/%d] Iter [%d/%d] Loss: %.6f, lr: %.6f, Iter time: %.5fs' %
                  (epoch, max_epoch, iteration, args.max_iter, disp_loss, lr, (end_time - start_time)))
            writer.add_scalar('train_loss', disp_loss, iteration)
            disp_loss = 0.
            start_time = time.time()

        if iteration in step_values:
            step_index += 1
            lr = adjust_learning_rate(optimizer, step_index, args.lr_decay)

        if iteration % args.test_step == 0:# and iteration > 1:
            rpf.eval()
            tot_start_time = time.time()
            disp_loss = total_disp_loss = 0.
            print("Evaluating RPF networks...")
            valid_iterator = iter(data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers))
            with torch.no_grad():
                for valid_iteration in range(valid_num_batches):
                    demo_im, demo_action_onehot, follower_im, follower_action = next(valid_iterator)
                    if args.cuda:
                        with torch.no_grad():
                            demo_im = Variable(demo_im.cuda(async=True))
                            demo_action_onehot = Variable(demo_action_onehot.cuda(async=True))
                            follower_im = Variable(follower_im.cuda(async=True))
                            follower_action = Variable(follower_action.cuda(async=True).long())
                    record, done = rpf(demo_im, demo_action_onehot, follower_im=follower_im)
                    loss = criterion(record['action_pred_t'].view(-1, args.action_dim), follower_action.view(-1))
                    disp_loss += loss.item()
                    total_disp_loss += loss.item()

                    if valid_iteration % args.print_step == 0:
                        disp_loss = disp_loss / args.print_step
                        end_time = time.time()
                        print('Iter [%d/%d] Eval loss: %.6f, Time: %.5fs' % (valid_iteration, valid_num_batches, disp_loss, (end_time - start_time)))
                        disp_loss = 0.
                        start_time = time.time()

                total_disp_loss = total_disp_loss / (valid_num_batches - 1)
                tot_end_time = time.time()
                print('[Epoch : %d] Loss : %.6f, Eval_time : %.5fs' % (epoch, total_disp_loss, (tot_end_time - tot_start_time)))
                writer.add_scalar('valid_loss', total_disp_loss, epoch)

            if eval_loss > total_disp_loss:
                eval_loss = total_disp_loss
                torch.save(rpf.state_dict(), save_dir + '/best.pth')
                print('Updated the best model')
            rpf.train()


def adjust_learning_rate(optimizer, step_index, lr_decay):
    lr = args.lr * (lr_decay ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()