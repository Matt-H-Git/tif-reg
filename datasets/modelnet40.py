#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source: Originally from the DGCNN repo, modified to work as a registration dataset
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

from se_math.transformer import SE3Transformer
from utils import preprocess_data_single, add_gaussian_noise
from huggingface_hub import hf_hub_download


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        hf_hub_download(repo_id="Msun/modelnet40", filename="modelnet40_ply_hdf5_2048.zip", local_dir=DATA_DIR, repo_type="dataset")
        os.system("unzip %s -d %s" % (DATA_DIR + "/modelnet40_ply_hdf5_2048.zip", DATA_DIR))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, args, partition='train'): # Options are 'train' or 'test'
        """
        Dataset for ModelNet40. Uses category files to determine which items are placed in the train or test partitions.
        Uses a randomly generated transform to obtain the source for each target.

        :param args: Arguments from cli
        :param partition: Whether to return the train or test items
        """
        self.data, self.label = load_data(partition)
        self.num_points = args.num_points
        self.partition = partition
        self.transformer = SE3Transformer(args)
        self.add_noise = args.add_noise


    def __getitem__(self, item):
        target = self.data[item][:self.num_points]
        target = torch.from_numpy(target)
        target = preprocess_data_single(target)
        source, gt = self.transformer(target)
        if self.add_noise:
            source = add_gaussian_noise(source, scale=30)
        gt = gt.squeeze(dim=0)
        return target, source, gt

    def __len__(self):
        if self.partition == "test":
            return 400
        return self.data.shape[0]