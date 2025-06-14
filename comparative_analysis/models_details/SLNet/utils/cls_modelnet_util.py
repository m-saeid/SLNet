import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/utils', BASE_DIR)[0]
sys.path.append(BASE_DIR)


# import copy
# import torch.nn.parallel
# import torch.utils.data
# import torch.utils.data.distributed

import os
import torch
import shutil
import logging
import argparse
import datetime
import numpy as np
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from encoder.Encoder import Encoder
from data.modelnet import ModelNet40
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from utils.util import Classifier, log_experiment, configuration_exists

# Import AveragedModel for SWA (Exponential Moving Average)
from torch.optim.swa_utils import AveragedModel, update_bn

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Classification(nn.Module):
    def __init__(self,
                 n=1024,
                 embed=[3, 16, 'no', 'yes', 0.4],
                 res_dim_ratio=0.25,
                 bias=False,
                 use_xyz=True,
                 norm_mode="anchor",
                 std_mode="BN1D",
                 dim_ratio=[2, 2, 2, 1],
                 num_blocks1=[1, 1, 2, 1],
                 transfer_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 block1_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 num_blocks2=[1, 1, 2, 1],
                 block2_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 k_neighbors=[32, 32, 32, 32],
                 sampling_mode=['fps', 'fps', 'fps', 'fps'],
                 sampling_ratio=[2, 2, 2, 2],
                 classifier_mode='mlp_very_large',
                 fps_method='pointops2',
                 knn_method='pytorch3d',
                 ):
        super(Classification, self).__init__()
        self.encoder = Encoder(
            n=n,
            embed=embed,
            res_dim_ratio=res_dim_ratio,
            bias=bias,
            use_xyz=use_xyz,
            norm_mode=norm_mode,
            std_mode=std_mode,
            dim_ratio=dim_ratio,
            num_blocks1=num_blocks1,
            transfer_mode=transfer_mode,
            block1_mode=block1_mode,
            num_blocks2=num_blocks2,
            block2_mode=block2_mode,
            k_neighbors=k_neighbors,
            sampling_mode=sampling_mode,
            sampling_ratio=sampling_ratio,
            fps_method=fps_method,
            knn_method=knn_method,
        )
        
        last_dim = embed[1]
        for d in dim_ratio:
            last_dim *= d
        
        self.classifier = Classifier(last_dim, classifier_mode, 40)
    
    def forward(self, xyz, feature):
        _, f_list = self.encoder(xyz, feature)
        x = F.adaptive_max_pool1d(f_list[-1], 1).squeeze(dim=-1)
        return self.classifier(x)
             
def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser('cls_modelnet_training')
    parser.add_argument('--n', type=int, default=1024, help='Point Number') 
    parser.add_argument('--embed', type=str, default='no', help='[mlp, no, adaptive]')
    parser.add_argument('--initial_dim', type=int, default=3, help='initial_dim')
    parser.add_argument('--embed_dim', type=int, default=16, help='embed_dim')
    parser.add_argument('--alpha_beta', type=str, default='yes', help='alpha_beta')
    parser.add_argument('--sigma', type=float, default=0.4, help='sigma')

    parser.add_argument('--res_dim_ratio', type=float, default=0.25,
                        help='Residual block dimension ratio.')
    parser.add_argument('--bias', type=bool, default=False, help='bias')
    parser.add_argument('--use_xyz', type=bool, default=True, help='Include xyz coordinates')
    parser.add_argument('--norm_mode', type=str, default='anchor', help='[center, anchor]')
    parser.add_argument('--std_mode', type=str, default='BN1D', help='[1111, B111, BN11, BN1D]')
    
    parser.add_argument('--dim_ratio', type=str, default='2-2-2-1', help='dim_ratio')
    
    parser.add_argument('--num_blocks1', type=str, default='1-1-2-1', help='num_blocks1')
    parser.add_argument('--transfer_mode', type=str, default='mlp-mlp-mlp-mlp', help='transfer_mode')
    parser.add_argument('--block1_mode', type=str, default='mlp-mlp-mlp-mlp', help='block1_mode')
    
    parser.add_argument('--num_blocks2', type=str, default='1-1-2-1', help='num_blocks2')
    parser.add_argument('--block2_mode', type=str, default='mlp-mlp-mlp-mlp', help='block2_mode')

    parser.add_argument('--k_neighbors', type=str, default='32-32-32-32', help='k_neighbors')
    parser.add_argument('--sampling_mode', type=str, default='fps-fps-fps-fps', help='sampling_mode')
    parser.add_argument('--sampling_ratio', type=str, default='2-2-2-2', help='sampling_ratio')

    parser.add_argument('--classifier_mode', type=str, default='mlp_very_large', help='classifier_mode')

    parser.add_argument('--fps_method', type=str, default='pointops2', help='fps_method: [pytorch3d, pointops2, pytorch]')
    parser.add_argument('--knn_method', type=str, default='pytorch3d', help='knn_method: [pytorch3d, pytorch]')

    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--epoch', default=300, type=int, help='number of epochs in training')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='minimum lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=6, type=int, help='number of data loader workers')
    parser.add_argument('--ema', default='yes', type=str, help='set "yes" to enable EMA using AveragedModel')
    
    return parser.parse_args()


def train(net, trainloader, optimizer, criterion, device, ema_net=None):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, feature, label) in enumerate(trainloader):
        data, feature, label = data.to(device), feature.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        logits = net(data, feature)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # Update the averaged model parameters
        if ema_net is not None:
            ema_net.update_parameters(net)

        train_loss += loss.item()
        preds = logits.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        total += label.size(0)
        correct += preds.eq(label).sum().item()
        progress_bar(batch_idx, len(trainloader), 
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, feature, label) in enumerate(testloader):
            data, feature, label = data.to(device), feature.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data, feature)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }