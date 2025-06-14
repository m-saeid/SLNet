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
from data.scanobject import ScanObjectNN
from time import time

from tasks.measuring_tools import *


def main(model, test_loader, data, model_name):

    print(f'\n\n  ****************************************\n  >>>>>>>>>>>>>>>>>>>> {model_name} \n  ****************************************')

    Dataset = "ScanObjectNN"
    Time = inference_time(model, test_loader, args.batch_size, model_name,)
    GFLOPs = gflops(model, data)
    Mem = memory(model, data)  # ideally a single float in GB or MB
    Params = params(model)
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


    
    


if __name__ == '__main__':
    args = parse_args()

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    device = 'cuda'
    cudnn.benchmark = True       

    if args.model == 'pointnet':
        # pointnet
        from Pointnet_Pointnet2.models.pointnet_cls import get_model as pointnet_model
        pointnet = pointnet_model(k=15, normal_channel=False)
        model = pointnet

    elif args.model == 'pointnet2_ssg':
        # pointnet++ - ssg
        from Pointnet_Pointnet2.models.pointnet2_cls_ssg import get_model as pointnet2_ssg_model
        pointnet2_ssg = pointnet2_ssg_model(num_class=15, normal_channel=False)
        model = pointnet2_ssg

    elif args.model == 'pointnet2_msg':
        # pointnet++ - msg
        from Pointnet_Pointnet2.models.pointnet2_cls_msg import get_model as pointnet2_msg_model
        pointnet2_msg = pointnet2_msg_model(num_class=15, normal_channel=False)
        model = pointnet2_msg

    elif args.model == 'dgcnn':
        # dgcnn
        from dgcnn.pytorch.model import DGCNN as dgcnn_model
        class args_dgcnn:
            k = 20
            emb_dims = 1024
            dropout = 0.5
        dgcnn = dgcnn_model(args_dgcnn, output_channels=15)
        model = dgcnn

    elif args.model == 'curvenet':
        # curvenet
        from CurveNet.core.models.curvenet_cls import CurveNet as curvenet_moel
        curvenet = curvenet_moel(num_classes=15, k=20, setting='default')
        model = curvenet

    elif args.model[:8] == 'pointmlp':
        if args.model == 'pointmlp_elite':
            # pointmlp-elite
            from pointMLP.classification_ScanObjectNN.models.pointmlp import pointMLPElite as pointmlp_elite_model
            pointmlp_elite = pointmlp_elite_model()
            model = pointmlp_elite
        elif args.model == 'pointmlp':
            # pointmlp
            from pointMLP.classification_ScanObjectNN.models.pointmlp import pointMLP as pointmlp_model
            pointmlp = pointmlp_model()
            model = pointmlp
        else:
            raise Exception("POINTMLP: {args.model}")
    
    elif args.model[:4] == 'apes':
        # apes
        from APES.apes.models.backbones.apes_cls_backbone import APESClsBackbone as APESClsBackbone_model
        from APES.apes.models.heads.apes_cls_head import APESClsHead as APESClsHead_model

        class APESCls_model(nn.Module):
            def __init__(self, which_ds):   # global/local
                super(APESCls_model, self).__init__()
                self.backbone = APESClsBackbone_model(which_ds=which_ds)
                self.head = APESClsHead_model(num_classes=15)
            def forward(self, x):
                features = self.backbone(x)
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
        from SLNet.utils.cls_scanobject_util import Classification as slnet_model

        if args.model == 'slnet_embed_dim_16':
            slnet_embed_dim_16 = slnet_model(embed=[3, 16, 'no', 'no', 0.4])
            model = slnet_embed_dim_16
        elif args.model == 'slnet_embed_dim_32':
            slnet_embed_dim_32 = slnet_model(embed=[3, 32, 'no', 'no', 0.4])
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
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, in_d=3, out_d=embd_dim), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    data = torch.randn(1, 3, 1024).cuda().contiguous()
    if args.model[:5] == 'slnet':
        feat = torch.randn(1, embd_dim, 1024).cuda().contiguous()
        data = (data, feat)
        
    main(model.to(device), test_loader, data ,args.model)
