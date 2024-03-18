# Some functions are adapted from TP6

import numpy as np
import random
import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import write_ply, read_ply
import sys


class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

     
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ToTensor()])


def test_transforms():
    return transforms.Compose([ToTensor()])


class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}


def data_loaders(ROOT_DIR = "./data/ModelNet40_PLY", transform=default_transforms(), batch_size=32,verbose=True):
    print("current dir ", os.getcwd())
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=transform)
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=transform)
    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    if verbose:
        print("Classes: ", inv_classes)
        print('Train dataset size: ', len(train_ds))
        print('Test dataset size: ', len(test_ds))
        print('Number of classes: ', len(train_ds.classes))
        print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)
    return(train_loader,test_loader)


if __name__=="__main__":
    print("current dir ", os.getcwd())
    ROOT_DIR = "../data/ModelNet40_PLY"
    train_loader,test_loader = data_loaders(ROOT_DIR=ROOT_DIR)
