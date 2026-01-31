import random
from math import radians
import torch
from utils import matrix_from_R_and_t
from . import se3, so3


class SE3Transformer:
    """ rigid motion """

    def __init__(self, args):
        self.max_rotation = args.max_rot # Degrees
        self.max_translation = args.max_trans
        self.min_rotation =70
        self.min_translation = 0.1

        self.gt = None
        self.igt = None

    def get_skewed_random_rotation(self):
        """
        Use weibull variate distribution to generate skewed random values
        The skew is towards the maximum value, since this is preferred for the present use case
            alpha -> How tall / flat the probability peak is. Higher alpha returns sharper peak
            beta -> The centre of the probability generation.
        Use a slightly lower value than the max rotation as the beta
        """
        # Generate random value
        x = random.weibullvariate(10.0, radians(self.max_rotation) * 0.85)

        # Randomly flip the sign
        flip_sign = random.randint(0, 1)
        if flip_sign:
            x = -x

        return x


    def generate_transform(self):
        # return: a se3 transform matrix [4x4]
        theta = torch.tensor(self.get_skewed_random_rotation())

        # Rotation
        r = torch.randn(1, 3)
        r = r / r.norm(p=2, dim=1, keepdim=True)
        K = so3.mat(r)
        R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.bmm(K, K)

        # Translation
        t = torch.randn(1, 3) * self.max_translation

        T = matrix_from_R_and_t(R, t)
        T_inv = torch.linalg.inv(T)

        self.gt = T_inv.squeeze(0)  # gt: source -> target
        self.igt = T.squeeze(0)  # igt: target -> source

    def apply_transform(self, target):
        # p0: [N, 3]
        # g: [1, 4, 4]
        g = self.igt
        source = target @ g[0:3, 0:3] + g[3, 0:3]
        return source, self.gt

    def __call__(self, tensor):
        self.generate_transform()
        return self.apply_transform(tensor)