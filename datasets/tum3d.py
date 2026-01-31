import os
from pathlib import Path
import glob
from torch.utils.data import Dataset
import torch

from se_math.transformer import SE3Transformer
from utils import tensor_from_ply, downsample_tensor, preprocess_data_single


def train_files():
    return ["ganesha_final.ply", "david2_final.ply", "dragon_final.ply", "wolf2_final.ply", "horse7_final.ply",
            "chicken_final.ply", "trex_final.ply", "rhino_final.ply", "gorilla0_final.ply", "para_final.ply",
            "lioness13_final.ply", "face_final.ply", "gun0026_final.ply", "centaur1_final.ply", "cat1_final.ply",
            "bunny_final.ply"]

def test_files():
    return ["armadillo_final.ply", "cheff_final.ply", "victoria3_final.ply", "dog7_final.ply"]


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    DESTINATION_DIR = os.path.join(DATA_DIR, 'clutter-models')
    path = Path(DESTINATION_DIR)
    if not os.path.exists(DESTINATION_DIR):
        path.mkdir(parents=True)
        www = 'https://cvg.cit.tum.de/_media/data/datasets/clutter/clutter-models.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition, num_points=2048):
    files = train_files() if partition == 'train' else test_files()
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    all_data = []
    for ply in glob.glob(os.path.join(DATA_DIR, 'clutter-models/models', '*.ply')):
        if os.path.basename(ply) not in files:
            continue
        ply_object = tensor_from_ply(ply)
        downsampled_object = downsample_tensor(ply_object, num_points)
        all_data.append(downsampled_object)

    all_data = torch.stack(all_data, dim=0)
    return all_data

class TUM3D(Dataset):
    def __init__(self, args, partition='train'):  # Options are 'train' or 'test'
        """
        Same as ModelNet40 in all regards.

        :param args: Arguments from cli
        :param partition: Whether to return train or test items
        """
        self.data = load_data(partition, args.num_points)
        self.transformer = SE3Transformer(args)

    def __getitem__(self, item):
        target = self.data[item]
        target = preprocess_data_single(target)
        source, gt = (self.transformer(target))
        gt = gt.squeeze(dim=0)
        return target, source, gt

    def __len__(self):
        return self.data.shape[0]