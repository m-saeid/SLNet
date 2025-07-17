import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import index_points, knn_point, query_ball_point
except:
    from encoder_util import index_points, knn_point, query_ball_point


try:
    from pytorch3d.ops import knn_points
except:
    print("pytorch3d library has not been installed")

class Grouping(nn.Module):
    def __init__(self, k, use_xyz=True, knn_method='pytorch3d', **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param s: s number
        :param k: k-nerighbors
        :param kwargs: others
        """
        super(Grouping, self).__init__()
        self.k = k
        self.use_xyz = use_xyz
        self.knn_method = knn_method

        if knn_method == 'pytorch3d':
            from pytorch3d.ops import knn_points
            #print('knn_method; pytorch3d')
        elif knn_method == 'pytorch':
            pass
            #print('knn_method; pytorch')
        else:
            raise Exception(f'fps_method cant be {knn_method}! it must be [pytorch3d, pytorch]')


    def forward(self, xyz, f, xyz_sampled, f_sampled): # 2,1024,3  2,1024,16  2,512,3  2,512,16
        B, N, C = xyz.shape         # 2,1024,3
        xyz = xyz.contiguous()  # 2,1024,3    xyz [btach, n, xyz]

        # GROPPING
        gr = grouping(self.k, 0, xyz, xyz_sampled, mode="knn", knn_method=self.knn_method)  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
        if gr is not tuple:
            idx = gr
            xyz_grouped = index_points(xyz, idx)        # [b, s, k, c]  (2,512,24,3)
        else:
            idx = gr[0]
            xyz_grouped = gr[1]
        f_grouped = index_points(f, idx)  # [b, s, k, c]  (2,512,24,16)

        return xyz_grouped, f_grouped


def grouping(k, radius, xyz, new_xyz, mode="knn", knn_method='pytorch3d'):
    if mode == "knn":
        if knn_method == 'pytorch3d':
            _, idx, grouped_xyz = knn_points(new_xyz, xyz, K=k, return_nn=True)  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
            return idx, grouped_xyz
        elif knn_method == 'pytorch':
            idx = knn_point(k, xyz, new_xyz)  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
            return idx
    elif mode == "ball":
        idx = query_ball_point(radius, k, xyz, new_xyz)
    return idx, grouped_xyz
