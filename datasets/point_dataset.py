import random

import torch
import torchvision.transforms
from PIL import Image
import os
import pandas as pd
import pickle as pkl
from configs.config import cfg
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomVerticalFlip, \
    RandomHorizontalFlip, RandomPerspective
from configs.config import project_root, data_root
from datasets.get_metadata import get_metadata
from timm.data import create_transform

path_train = os.path.join(data_root + '/训练集数据/csv文件/')
path_testA = os.path.join(data_root + '/A榜测试集数据/csv文件/')
label_train = os.path.join(data_root + '/训练集数据/文件标签汇总数据.csv')

class_to_ind = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
}


# for i in enumerate(os.listdir('.' + path_train)):
#     data = pd.read_csv(os.path.join(os.path.join('.' + path_train + i[1])))
#     data = np.array(data)
#     data = torch.tensor(data)
#     print(data)
#     label_list = pd.read_csv('.' + label_train)
#     label_list = np.array(label_list)
#     label_list = label_list[:, [1, 0]]
#     print(label_list)
#     label_list = dict(label_list)
#     print(label_list)
#     exit(0)


class Dataset_point(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()

        if split not in ('trainval', 'train', 'test'):
            raise ValueError("split must be in trainval, train, val, or test. Supplied {}".format(split))
        self.split = split
        self.cfg = cfg
        self.is_test: bool
        self.class_to_ind = class_to_ind
        if self.split == 'trainval' or 'train':
            self.is_test = False
            self.path = path_train

            self.label_train = label_train
            label_list = pd.read_csv(self.label_train)
            label_list = np.array(label_list)
            self.label_names = label_list[:, 0]
            self.img_names = label_list[:, 1]
            label_list = label_list[:, [1, 0]]
            self.label_list = dict(label_list)

            self.img_paths = os.listdir(path_train)
            self.label_idxs = []
            for _, i in enumerate(self.label_names):
                self.label_idxs.append(self.class_to_ind[i])
        else:
            self.is_test = True
            self.path = path_testA
            self.img_paths = os.listdir(path_testA)

        self.datas = []
        for _, i in enumerate(self.img_names):
            data = pd.read_csv(os.path.join(self.path + i))
            data = np.array(data)
            data = torch.tensor(data)

            N, C = data.shape
            index = torch.LongTensor(random.sample(range(N), cfg.npoints))
            data = torch.index_select(data, 0, index)

            self.datas.append(data)

    def __getitem__(self, index):

        if self.split != 'test':
            entry = {
                'img': self.datas[index],
                'img_name': self.img_names[index],
                'label_names': self.label_names[index],
                'label_idxs': self.label_idxs[index],
                'img_path': self.img_paths[index]}
        else:
            entry = {
                'img': self.datas[index],
                'img_name': self.img_names[index],
                'img_path': self.img_paths[index]}
        return entry

    def __len__(self):
        return len(self.datas)
