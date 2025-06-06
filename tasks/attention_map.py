import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]
sys.path.append(BASE_DIR)


import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data.modelnet import ModelNet40
from utils.helper import cal_loss

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.attention_map_util import plot_attention_map_points_only, Classification # plot_attention_map_to_file, plot_attention_map_interactive
from checkpoints.cls_dgcnn_modelnet.model import DGCNN

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--embed_dim', type=int, default=32,                    # SLNet
                        help='embed_dim')
    
    parser.add_argument('--batch_size', type=int, default=128,                  # test_loader
                        help='batch size in training')
    parser.add_argument('--num_points', type=int, default=1024,                 # test_loader
                        help='Point Number')

    parser.add_argument('--dropout', type=float, default=0.5,                   # DGCNN
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',      # DGCNN
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',               # DGCNN
                        help='Num of nearest neighbors to use')
    parser.add_argument('-c', '--slnet_path', default='checkpoints/cls_modelnet', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--dgcnn_path', type=str, default='checkpoints/cls_dgcnn_modelnet/model.1024.t7', metavar='N',
                        help='Pretrained model path')
    return parser.parse_args()


def main():
    args = parse_args()
    args.slnet_path = f'{args.slnet_path}/pretrained_embedDim{str(args.embed_dim)}/best_checkpoint.pth'

    # print(f"args: {args}")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, out_d=args.embed_dim), num_workers=4,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = 'cuda'

    SLNet = Classification( embed=[3, args.embed_dim, 'no', 'yes', 0.4]).to(device)
    dgcnn = DGCNN(args).to(device)

    checkpoint = torch.load(args.slnet_path, map_location=torch.device('cpu'))
    if device == 'cuda':
        SLNet = torch.nn.DataParallel(SLNet)
        cudnn.benchmark = True
    SLNet.load_state_dict(checkpoint['net'])

    dgcnn = DGCNN(args).to(device)
    dgcnn = nn.DataParallel(dgcnn)
    dgcnn.load_state_dict(torch.load(args.dgcnn_path))
    #model = model.eval()

    SLNet.eval()
    dgcnn.eval()

    att_map(SLNet, dgcnn, test_loader, device, args)


def att_map(SLNet, dgcnn, testloader, device, args):
    with torch.no_grad():
        for batch_idx, (data, feature, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)

            # SLNet
            SLNet(data, feature, args.embed_dim)

            # DGCNN
            dgcnn(data)

            # just NAPE
            for i in [1,2,5,6,7,10,11,13,19,21,24,47,59]:
                plot_attention_map_points_only(data.permute(0,2,1), feature, i, dir=f"attention/NAPE_embedDim{args.embed_dim}")

            print("DONE")
            return 1


main()
