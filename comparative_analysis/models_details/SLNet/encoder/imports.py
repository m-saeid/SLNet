try:
    from encoder.m0_embedding import Embedding
    from encoder.m1_sampling import Sampling
    from encoder.m2_grouping import Grouping
    from encoder.m3_normalization import Normalization

    from encoder.m4_block1 import Block1
    from encoder.m5_aggregation import Agg
    from encoder.m6_block2 import Block2
except:
    from m0_embedding import Embedding
    from m1_sampling import Sampling
    from m2_grouping import Grouping
    from m3_normalization import Normalization

    from m4_block1 import Block1
    from m5_aggregation import Agg
    from m6_block2 import Block2

try:
    from pointnet2_ops import pointnet2_utils
except:
    print("pointnet2_ops library has not been installed")

try:
    from pytorch3d.ops import knn_points
except:
    print("pytorch3d library has not been installed")