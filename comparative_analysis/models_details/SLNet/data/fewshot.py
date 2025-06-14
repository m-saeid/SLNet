import os
import sys
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
try:
    from nape import NAPE
except:
    from data.nape import NAPE

# Path setup
project_path = os.path.abspath(".")
sys.path.append(project_path)

# import data.dataset_utils as dutils

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def load_data_modelnet40fewshot(partition, n_way, k_shot, fold):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset/modelnet_fewshot')

    with open(
        os.path.join(
            DATA_DIR, f"{n_way}way_{k_shot}shot", f"{fold}.pkl"
        ),
        "rb",
    ) as f:
        pkl_data = pickle.load(f)[partition]

    fold_data = np.array([inner[0][:, :3] for inner in pkl_data])
    fold_normals = np.array([inner[0][:, 3:] for inner in pkl_data])
    fold_label = np.array([inner[1] if len(inner) > 1 else None for inner in pkl_data])

    return fold_data, fold_label


class ModelNet40FewShot(Dataset):
    def __init__(
        self,
        num_points=1024,
        partition="train",
        n_way=5,
        k_shot=10,
        augment_type=None,
        in_d=3,
        out_d=16
    ):

        self.num_points = num_points
        self.partition = partition
        self.augment_type = augment_type
        self.nape = NAPE(in_d, out_d)

        self.fold_num = 0
        all_fold_data = []
        all_fold_label = []
        for fold in range(10):
            fold_data, fold_label = load_data_modelnet40fewshot(
                partition, n_way, k_shot, fold)
            all_fold_data.append(fold_data)
            all_fold_label.append(fold_label)
        self.all_data = np.stack(all_fold_data, axis=0)
        self.all_label = np.stack(all_fold_label, axis=0)

        self.set_fold(self.fold_num)

    def set_augmentation(self, augment_type):
        self.augment_type = augment_type

    def set_fold(self, fold_num):
        self.fold_num = fold_num
        self.data = self.all_data[self.fold_num]
        self.label = self.all_label[self.fold_num]

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points]
        label = self.label[item]
        if self.partition == "train" and self.augment_type:
            pointcloud[:, 0:3] = pc_normalize(pointcloud[:, 0:3])
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        feature = torch.tensor(pointcloud).unsqueeze(0)  # 1,n,3     1,1024,3
        feature = self.nape(feature).squeeze()        #   n,out_d    1024,16
        feature = feature.numpy()
        return pointcloud, feature, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":

    train = ModelNet40FewShot(num_points=1024)
    test = ModelNet40FewShot(num_points=1024, partition="test")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        ModelNet40FewShot(partition="train", num_points=1024),
        num_workers=4,
        batch_size=32,
        shuffle=False,
        drop_last=True,
    )
    train_loader.dataset.set_fold(2)
    for batch_idx, (data, feature, label) in enumerate(train_loader):
        print(data[0, 0])
        print(
            f"batch_idx: {batch_idx}  | data shape: {data.shape} | feautre shape: {feature.shape} | ;lable shape: {label.shape}"
        )
        break

    train_set = ModelNet40FewShot(partition="train", num_points=1024, in_d=3, out_d=16)
    test_set = ModelNet40FewShot(partition="test", num_points=1024, in_d=3, out_d=16)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
