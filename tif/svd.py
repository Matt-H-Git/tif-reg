import torch
import torch.nn as nn

class SingularValueDecomposition(nn.Module):
    """
        [N, 320], [N, 320], [N, 3] -> [N, 3]
    """

    def __init__(self, args):
        super(SingularValueDecomposition, self).__init__()
        self.loss = 0.0
        self.U = None
        self.S = None
        self.Vt = None

    def forward(self, X, Z):
        X = X.mT
        Z = Z.mT

        # Mean
        x_mu = torch.mean(X, dim=2).unsqueeze(-1)
        z_mu = torch.mean(Z, dim=2).unsqueeze(-1)

        # Centralised data objects
        x_cen = X - x_mu
        z_cen = Z - z_mu

        # Cross covariance matrix
        H = x_cen @  z_cen.transpose(-1, -2)

        # SVD
        U, S, Vt = torch.linalg.svd(H, full_matrices=True)
        self.U = U
        self.S = S
        self.Vt = Vt

        # Estimated transformation parameters
        R = Vt.mT @ U.mT
        T = (-R @ x_mu) + z_mu
        self.loss = torch.linalg.norm(R @ X + T - Z)
        return R, T