import torch.nn as nn
import torch


class CorrespondingPointGeneration(nn.Module):
    """
        [N, 320], [N, 320], [N, 3] -> [N, 3]
    """

    def __init__(self, args):
        super(CorrespondingPointGeneration, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.device = args.device

    def forward(self, Fx, Fy, y):
        W = self.softmax(Fx @ Fy.mT) #.to(self.device)
        Z = W @ y
        return Z