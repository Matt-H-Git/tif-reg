import torch
from .tif_extract import TIFExtract
from .deep_feature_embedding import DGCNN
from .corresponding_point_generation import CorrespondingPointGeneration
from .svd import SingularValueDecomposition

"""
TIF-REG architecture

4 stages:
Nx3 -> NxKx4 -> Nx320 -> Nx3 -> Output
1) Transform invariant feature extraction
    Runs KNN for each point, builds up a set of neighbourhoods and translates them in 4 different ways to end up with
    shape NxKx4 where K is the number of points in each neighbourhood
    
2) Deep feature embedding
    Runs an MLP taken from another paper which ends in shape Nx320

3) Corresponding point generation
    Use the attention mechanism to generate a point cloud that is as similar as possible to the target with a mapping
    to correspond each point in the source to the output point cloud

4) Decoupled Singular Value Decomposition
    Finds a rotation and translation transformation between the source and target. Find a cross covariance matrix,
    calculate its decomposition and form the transformations by solving the dual optimisation problem

"""



# Overall pipeline
class TIFReg(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.tif_extract = TIFExtract(args)
        self.feature_embedding = DGCNN(args)
        self.corresponding_point_generation = CorrespondingPointGeneration(args)
        self.svd = SingularValueDecomposition(args)

    def forward(self, x, y):
        # x, y: [B, N, 3]
        # NB: The input data should be mean centered and normalised to a unit sphere

        # Stage 1: Extract TIFs
        x_tifs = self.tif_extract(x)
        y_tifs = self.tif_extract(y)

        # Stage 2: Extract deep learning features
        x_tifs = torch.permute(x_tifs, (0, 3, 1, 2))
        y_tifs = torch.permute(y_tifs, (0, 3, 1, 2))
        Fx = self.feature_embedding(x_tifs)
        Fy = self.feature_embedding(y_tifs)

        # Stage 3: Corresponding point generation
        Z = self.corresponding_point_generation(Fx, Fy, y)

        # Stage 4: Decoupled Singular Value Decomposition
        # Not doing decoupled for now, this is just regular SVD
        R, T = self.svd(x, Z)
        T = T.squeeze(dim=-1)

        return R.mT, T # Result is column major so transpose R


"""
This class implements the simplified version of TIF-Reg. 
See the README for an explanation on the shortcut.
"""
class TIFRegSimplified(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.tif_extract = TIFExtract(args)
        self.feature_embedding = DGCNN(args)
        self.corresponding_point_generation = CorrespondingPointGeneration(args)
        self.svd = SingularValueDecomposition(args)

    def forward(self, x, y):
        # x, y: [B, N, 3]
        R, T = self.svd(x, y)
        return R.mT, T.squeeze(dim=-1) # Result is column major so transpose R