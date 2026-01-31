from torch.utils.data import Dataset
from se_math.transformer import SE3Transformer
from utils import tensor_from_ply, downsample_tensor, preprocess_data_single, add_gaussian_noise


class StanfordBunny(Dataset):
    def __init__(self, args, target_path, n_elements=50, n_points=2048):
        """
        Dataset for the Stanford Bunny. Takes in a single ply as the target and generates a new random transform
        to obtain each source.

        :param args: Arguments from cli
        :param target_path: Path to the Stanford Bunny .ply
        :param n_elements: Number of Bunny's to generate in the dataset
        :param n_points: Number of points to downsample the bunny to
        :param add_noise: Whether to add noise or not
        """
        self.num_points = n_points
        self.target = preprocess_data_single(downsample_tensor(tensor_from_ply(target_path), self.num_points))
        self.noisy_target = add_gaussian_noise(self.target.clone(), scale=50) if args.add_noise else None
        self.transformer = SE3Transformer(args)
        self.n_elements = n_elements

    def __getitem__(self, item):
        source, gt = (self.transformer(self.target))
        gt = gt.squeeze(dim=0)
        if self.noisy_target is not None:
            return self.noisy_target, source, gt
        return self.target, source, gt

    def __len__(self):
        return self.n_elements