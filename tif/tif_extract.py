import torch
from sklearn.neighbors import KNeighborsTransformer


# Stage 1
class TIFExtract:
    """
    Transform Invariant Feature Extraction
    [N, 3] -> [N, K, 4]
    """
    def __init__(self, args):
        self.k = args.k
        self.device = args.device

    def construct_neighbourhoods(self, x: torch.Tensor):
        x_cpu = x.cpu() # Sklearn uses numpy, which requires tensors to be on CPU for the conversion
        knn_transformer = KNeighborsTransformer(n_neighbors=self.k)
        knn_transformer.fit(x_cpu)
        neighbour_idxs = knn_transformer.kneighbors(x_cpu, n_neighbors=self.k, return_distance=False) # [N, K]
        # Use the indexes to retrieve the actual coordinates
        neighbour_coords = x[neighbour_idxs] # [N, K] -> [N, K, 3]
        return neighbour_coords

    def __call__(self, batch):
        tif_arr = []
        for x in batch: # TODO: Vectorise
            X = self.construct_neighbourhoods(x) # [N, 3] -> [N, K, 3]
            x_mu = torch.mean(X, dim=1)  # Centre of each neighbourhood
            x_ik = X[:, -1, :]  # x_ik is the last point in U(xi)

            # l1, l2, l3, l4 -> [N, K]
            l1 = torch.norm(X - x_mu.unsqueeze(1).expand(-1, X.shape[1], -1), dim=-1)  # || x_ib - x_mu ||
            l2 = torch.norm(X - x.unsqueeze(1).expand(-1, X.shape[1], -1), dim=-1)  # || x_ib - x_i ||
            l3 = torch.norm(x - x_mu, dim=-1).unsqueeze(-1).expand(-1, X.shape[1])  # || x_i - x_mu ||
            l4 = torch.norm(x_ik - x, dim=-1).unsqueeze(-1).expand(-1, X.shape[1])  # || x_ik - x_i ||
            tif = torch.stack((l1, l2, l3, l4), dim=-1)  # [N, K, 4]
            tif_arr.append(tif)

        tifs = torch.stack(tif_arr, dim=0)
        return tifs
