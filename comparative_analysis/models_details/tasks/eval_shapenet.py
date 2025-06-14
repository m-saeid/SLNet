import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]

sys.path.append(BASE_DIR)
sys.path.append(f'{BASE_DIR}/Pointnet_Pointnet2')
sys.path.append(f'{BASE_DIR}/Pointnet_Pointnet2/models')
sys.path.append(f'{BASE_DIR}/dgcnn')
sys.path.append(f'{BASE_DIR}/CurveNet')
sys.path.append(f'{BASE_DIR}/pointMLP')
sys.path.append(f'{BASE_DIR}/APES')
sys.path.append(f'{BASE_DIR}/SLNet')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import csv
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data.shapenet import PartNormalDataset
from time import time

from tasks.measuring_tools import *


def main(model, test_loader, data, model_name):

    print(f'\n\n  ****************************************\n  >>>>>>>>>>>>>>>>>>>> {model_name} \n  ****************************************')

    Dataset = "Shapenet"

    Time = inference_time_shapenet(model, test_loader, args.batch_size, model_name)

    GFLOPs = gflops(model, data)
    Mem = memory(model, data)  # ideally a single float in GB or MB
    Params = params(model)
    # Time = inference_time_shapenet(model, test_loader, args.batch_size, model_name)
    NumPoints = args.num_points
    Batch_size = args.batch_size
    Workers = args.workers

    # GPU Info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name
    gpu_capability = f"{props.major}.{props.minor}"
    gpu_mem_total = round(props.total_memory / 1e9, 2)  # in GB

    # Prepare row in your desired order
    row = {
        "Model": model_name,
        "Dataset": Dataset,
        "GFLOPs": GFLOPs,
        "GPU_Memory_Used": Mem,
        "Parameters": Params,
        "InferenceTime_ms": Time,
        "NumPoints": NumPoints,
        "BatchSize": Batch_size,
        "Worker": Workers,
        "GPU_Name": gpu_name,
        "GPU_TotalMem_GB": gpu_mem_total,
        "GPU_ComputeCapability": gpu_capability
    }

    # CSV file path
    csv_file = "results.csv"
    fieldnames = list(row.keys())

    # Write to CSV (append if exists, write header if not)
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("✅ Results saved to", csv_file)




def one_batch_prep(shapenet_data, model_name):

    device = 'cuda'

    data = shapenet_data.__getitem__(0)
    data = (torch.from_numpy(d).unsqueeze(0) for d in data)


    if model_name[:5] == 'slnet':
        points, feature, label, target, norm_plt = data
        points, feature, label, target, norm_plt = Variable(points.float()), Variable(feature.float()), Variable(label.long()), \
                                            Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        feature = feature.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, feature, label, target, norm_plt = points.cuda(non_blocking=True), feature.cuda(non_blocking=True), \
                                                label.squeeze(1).cuda(non_blocking=True), \
                                                target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        return points, feature, norm_plt, to_categorical(label, 16)


    elif model_name == 'pointmlp':
        points, label, target, norm_plt = data
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                        Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                        target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        return points, norm_plt, to_categorical(label, 16)


    elif model_name in ['dgcnn', 'curvenet']:
        data, label, seg, _ = data
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        return data, label_one_hot


    elif model_name in ['pointnet', 'pointnet2_ssg', 'pointnet2_msg']:
        points, label, target, normals = data
        # points = torch.cat([points, normals], 2)    # with normals
        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
        points = points.transpose(2, 1)
        return points, to_categorical(label, 16)

    elif model_name in ['apes_local', 'apes_global']:
        data, label, seg, _ = data
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32)).unsqueeze(-1)
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        return data, label_one_hot



if __name__ == '__main__':
    args = parse_args()

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    device = 'cuda'
    cudnn.benchmark = True       

    if args.model == 'pointnet':
        # pointnet
        from Pointnet_Pointnet2.models.pointnet_part_seg import get_model as pointnet_model
        pointnet = pointnet_model(part_num=50, normal_channel=False)
        model = pointnet

    elif args.model == 'pointnet2_ssg':
        # pointnet++ - ssg
        from Pointnet_Pointnet2.models.pointnet2_part_seg_ssg import get_model as pointnet2_ssg_model
        pointnet2_ssg = pointnet2_ssg_model(num_classes=16, normal_channel=False)
        model = pointnet2_ssg

    elif args.model == 'pointnet2_msg':
        # pointnet++ - msg
        from Pointnet_Pointnet2.models.pointnet2_part_seg_msg import get_model as pointnet2_msg_model
        pointnet2_msg = pointnet2_msg_model(num_classes=16, normal_channel=False)
        model = pointnet2_msg

    elif args.model == 'dgcnn':
        # dgcnn
        from dgcnn2.model import DGCNN_partseg as dgcnn_model
        class args_dgcnn:
            k = 40
            emb_dims = 1024
            dropout = 0.5
        dgcnn = dgcnn_model(args_dgcnn, seg_num_all=50)
        model = dgcnn

    elif args.model == 'curvenet':
        # curvenet
        from CurveNet.core.models.curvenet_seg import CurveNet as curvenet_moel
        curvenet = curvenet_moel()
        model = curvenet

    elif args.model[:8] == 'pointmlp':
        if args.model == 'pointmlp_elite':
            raise NotImplementedError
        elif args.model == 'pointmlp':
            # pointmlp
            from pointMLP.part_segmentation.model.pointMLP import pointMLP as pointmlp_model
            pointmlp = pointmlp_model()
            model = pointmlp
        else:
            raise Exception("POINTMLP: {args.model}")
    
    elif args.model[:4] == 'apes':
        # apes
        from APES.apes.models.backbones.apes_seg_backbone import APESSegBackbone as APESSegBackbone_model
        from APES.apes.models.heads.apes_seg_head import APESSegHead as APESSegHead_model

        class APESCls_model(nn.Module):
            def __init__(self, which_ds):   # global/local
                super(APESCls_model, self).__init__()
                self.backbone = APESSegBackbone_model(which_ds=which_ds)
                self.head = APESSegHead_model()
            def forward(self, x, l):
                features = self.backbone(x, l)
                output = self.head(features)
                return output
        if args.model == 'apes_global':
            apes_global = APESCls_model(which_ds="global")
            model = apes_global
        elif args.model == 'apes_local':
            apes_local = APESCls_model(which_ds="local")
            model = apes_local
        else:
            raise Exception("APES: {args.model}")

    elif args.model[:5] == 'slnet':
        #SLNet
        from SLNet.decoder.Decoder import Decoder as slnet_model

        if args.model == 'slnet_embed_dim_16':
            slnet_embed_dim_16 = slnet_model(embed=[6, 16, 'no', 'yes', 0.4])
            model = slnet_embed_dim_16
        elif args.model == 'slnet_embed_dim_32':
            slnet_embed_dim_32 = slnet_model(embed=[6, 32, 'no', 'yes', 0.4])
            model = slnet_embed_dim_32
        else:
            raise Exception("SLNet: {args.model}")      
    
    else:
        raise Exception("MODEL NAME: {args.model}")

    '''
    all_models = {'pointnet':pointnet, 'pointnet2_ssg':pointnet2_ssg, 'pointnet2_ssg':pointnet2_ssg,
                  'dgcnn':dgcnn, 'curvenet':curvenet,
                  'pointmlp_elite':pointmlp_elite, 'pointmlp':pointmlp,
                  'apes_global':apes_global, 'apes_local':apes_local,
                  'slnet_embed_dim_16':slnet_embed_dim_16, 'slnet_embed_dim_32':slnet_embed_dim_32}
    '''

    # model = all_models[args.model]
    model = torch.nn.DataParallel(model)

    embd_dim = int(args.model[-2:]) if args.model[:5] == 'slnet' else None

    shapenet_data = PartNormalDataset(split='test', npoints=args.num_points, normalize=False, out_d=embd_dim)
    test_loader = DataLoader(shapenet_data, num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    data = one_batch_prep(shapenet_data, args.model)
        
    main(model.to(device), test_loader, data ,args.model)





"""
Models:

pointnet(point_cloud, label)
pointnet2(xyz, cls_label)
* dgcnn(x, l)
* curvenet(xyz, l=None)
* pointmlp(x, norm_plt, cls_label)
* apes(x, shape_class)
* slnet(points, feature, norm_plt, to_categorical(label, num_classes))

"""