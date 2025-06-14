import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import index_points, farthest_point_sample
except:
    from encoder_util import index_points, farthest_point_sample


try:
    from pointnet2_ops import pointnet2_utils
except:
    print("pointnet2_ops library has not been installed")

try:
    from pytorch3d.ops import sample_farthest_points
except:
    print("pytorch3d library has not been installed")


class Sampling(nn.Module):
    def __init__(self, mode='fps', s=512, fps_method='pointops2', **kwargs):
        super(Sampling, self).__init__()
        self.mode = mode
        self.s = s
        self.fps_method = fps_method

        if fps_method =='pytorch3d':
            from pytorch3d.ops import sample_farthest_points
            #print('fps_method; pytorch3d')
        elif fps_method == 'pointops2':
            from pointnet2_ops import pointnet2_utils
            #print('fps_method; pointnet2_ops')
        elif fps_method == 'pytorch':
            pass
            #print('fps_method; pytorch')
        else:
            raise Exception(f'fps_method cant be {fps_method}! it must be [pointops2, pytorch3d, pytorch]')

    def forward(self, xyz, f):
        if self.mode == "fps":
            if self.fps_method =='pytorch3d':
                xyz_sampled, idx = sample_farthest_points(xyz, K=self.s)#.long()
                #idx = idx.long()
            elif self.fps_method == 'pointops2':
                xyz = xyz.contiguous()
                idx = pointnet2_utils.furthest_point_sample(xyz, self.s).long()  # [B, npoint]
                xyz_sampled = index_points(xyz, idx)
            elif self.fps_method == 'pytorch':
                idx = farthest_point_sample(xyz, self.s).long()
                xyz_sampled = index_points(xyz, idx)

            f_sampled = index_points(f, idx) if f is not None else None
            return xyz_sampled, f_sampled