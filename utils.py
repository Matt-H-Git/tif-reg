import os

import open3d
import torch
import copy
import numpy as np
from math import floor

def tensor_from_ply(path, voxel_downsample_scale=0):
    """
    :param path:
    :param n_points: Number of points to downsample the object to
    :return:
    """
    pcd = open3d.io.read_point_cloud(path)
    if voxel_downsample_scale != 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample_scale)
    x = np.asarray(pcd.points)
    return torch.from_numpy(x).type(torch.float32)

def tensor_to_pcd(x: torch.Tensor):
    pcd = open3d.geometry.PointCloud()
    x = x.squeeze(0)
    pcd.points = open3d.utility.Vector3dVector(x.cpu().numpy())
    return pcd

def downsample_tensor(x: torch.Tensor, n_points):
    # Construct a range of indexes which will uniformly downsample the model
    """step_size = floor(x.shape[0] / n_points)
    idx_range = torch.arange(0, x.shape[0], step=step_size)[:n_points]
    return x[idx_range]"""
    # Use torch interpolate
    mode = 'nearest'
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    if len(x.shape) == 3:
        mode = 'linear'
    return torch.nn.functional.interpolate(x.mT, n_points, mode=mode).squeeze(0).T

def matrix_from_R_and_t(R, t):
    mat = torch.eye(4)
    mat[0:3, 0:3] = R
    mat[3, 0:3] = t
    return mat

def batch_matrix_from_R_and_t(R, t):
    mat = torch.eye(4).unsqueeze(0)
    mat = mat.repeat(R.shape[0], 1, 1)
    mat[:, 0:3, 0:3] = R
    mat[:, 3, 0:3] = t
    return mat

def torch_to_numpy(x: torch.Tensor) -> np.ndarray: # Converts column major torch tensor to row major numpy tensor
    result = np.identity(4)
    result[0:3, 0:3] = x[0:3, 0:3].t()
    result[0:3, 3] = x[3, 0:3]
    result[3, 0:3] = x[0:3, 3]
    return result

def preprocess_data(source: torch.Tensor, target: torch.Tensor):
    # Preprocessing:
    assert source.ndim == 3 and target.ndim == 3
    ## Mean centre separately
    source -= torch.mean(source, dim=1, keepdim=True)
    target -= torch.mean(target, dim=1, keepdim=True)

    ## Normalise the target then apply the same transform to the source
    furthest_distance_target = torch.norm(target, dim=2).max(dim=1, keepdim=True)[0]  # [B, 1]
    target = target / furthest_distance_target.unsqueeze(-1)  # Rescale for unit sphere [B, N, 3]

    furthest_distance_source = torch.norm(source, dim=2).max(dim=1, keepdim=True)[0]  # [B, 1]
    source = source / furthest_distance_source.unsqueeze(-1)  # Rescale for unit sphere [B, N, 3]
    return source, target

def preprocess_data_single(x: torch.Tensor):
    # Preprocessing:
    assert x.ndim == 2
    ## Mean centre separately
    x -= torch.mean(x, dim=0, keepdim=True)

    ## Normalise the target then apply the same transform to the source
    furthest_distance = torch.norm(x, dim=1).max(dim=0, keepdim=True)[0]  # [1]
    x = x / furthest_distance.unsqueeze(-1)  # Rescale for unit sphere [N, 3]

    return x

def add_gaussian_noise(x: torch.Tensor, scale: int):
    # Generate some Gaussian noise and add it to the target
    x = x + (torch.rand(x.shape[0]).unsqueeze(-1) / scale)
    return x

# visualize the point clouds
def draw_registration_result(source: torch.Tensor, target: torch.Tensor, transformation: torch.Tensor):
    # Convert to o3d
    source_o3d = tensor_to_pcd(source)
    target_o3d = tensor_to_pcd(target)
    transform_np = torch_to_numpy(transformation.cpu().detach())
    # Draw
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    open3d.visualization.draw_geometries([source_temp, target_temp]) # Uncomment this to draw the original meshes before registration
    source_temp.transform(transform_np)
    open3d.visualization.draw_geometries([source_temp, target_temp])

    open3d.io.write_point_cloud("./source_test.ply", source_temp)
    open3d.io.write_point_cloud("./target_test.ply", target_temp)