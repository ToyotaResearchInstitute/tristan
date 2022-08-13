import string
from collections import OrderedDict
from typing import Tuple

import torch
from torch.utils.data import Dataset


def compute_mean_variances(
    data: torch.Tensor, diagonal_term_coefficient: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean and covariance matrix of a set of samples.

    Parameters
    ----------
    data: torch.Tensor
        [batch, num_features, num_samples] tensor with samples of a vector per batch.

    Returns
    -------
    mean: torch.Tensor
        [batch,num_features] tensor of the mean of each feature
    covariance: torch.Tensor
        [batch, num_features, num_features] tensor with the covariance matrix for each batch.
    """
    # data should be [batch, num_features, num_samples]
    mean = torch.mean(data, -1)
    feature_dim = data.shape[-2]
    data_m = data - mean.unsqueeze(-1)

    covariance2 = torch.matmul(data_m, data_m.transpose(-1, -2)) / data_m.shape[-1]
    # Add a diagonal term to the covariance metrics
    covariance2 += torch.eye(n=covariance2.shape[-2], device=covariance2.device) * diagonal_term_coefficient
    return mean, covariance2


class ExtendedRandomDictDataset(Dataset):
    """Random input/output dictionary Dataset with support for multiple keys.
    Args:
        sizes: dict: str -> tuple
        num_samples: number of samples
    """

    def __init__(self, sizes: dict, num_samples: int = 250) -> None:
        self.len = num_samples
        self.data = OrderedDict()
        for key in sizes:
            self.data[key] = torch.randn(num_samples, sizes[key])

    def __getitem__(self, index: int) -> None:
        res = {}
        for i, key in enumerate(self.data):
            res[key] = self.data[key][index] + i
        return res

    def __len__(self) -> int:
        return self.len


def merge_dims(tensor: torch.Tensor, dims_to_merge: Tuple[int, int]) -> torch.Tensor:
    """Merge two adjacent dims into one.

    tensor:
        The tensor to operate on.
    dims_to_merge: Tuple[int, int]
        The index of the two dimension to merge.

    return:
        New tensor with one dim less
    """
    dim1, dim2 = dims_to_merge[0], dims_to_merge[1]
    assert abs(dim1 - dim2) == 1, "dims_to_merge must be adjacent dims"
    shape = list(tensor.shape)
    shape[dim1] = shape[dim1] * shape[dim2]
    shape.pop(dim2)
    # [batch, samples, agents*future_timesteps, attention_dim]
    tensor = tensor.reshape(*shape)
    return tensor


def split_dims(tensor: torch.Tensor, dims_to_split: int, size_of_dim: int):
    """Split one dimension into two.

    tensor: torch.Tensor
        The tensor to split.
    dims_to_split: int
        The index of the dimension to split.
        The new dimension will be at dims_to_split+1
    size_of_dim:
        The length of the first dimension at index dims_to_split.

    Return:
        tensor with an additional dimension
    """
    shape = list(tensor.shape)
    shape.insert(dims_to_split, 0)
    if dims_to_split < 0:  # shift negative index after inserting behind
        dims_to_split -= 1
    shape[dims_to_split] = size_of_dim
    shape[dims_to_split + 1] = -1
    # [batch, samples, agents, future_timesteps, attention_dim]
    tensor = tensor.reshape(*shape)
    return tensor


def vector_list_to_rotation_matrices(vector_list: torch.Tensor):
    v_x = vector_list[..., 0]
    v_y = vector_list[..., 1]
    vector_angles_rad = torch.atan2(v_y, v_x)  # torch atan2 expects the y coordinates first
    vector_cos = torch.cos(vector_angles_rad)
    vector_sin = torch.sin(vector_angles_rad)
    return torch.stack(
        (
            torch.stack((vector_cos, -vector_sin), dim=-1),
            torch.stack((vector_sin, vector_cos), dim=-1),
        ),
        dim=-2,
    )


def apply_2d_coordinate_rotation_transform(
    rotations_a_b: torch.Tensor,
    coordinates_b: torch.Tensor,
    result_einsum_prefix: str,
    rotation_einsum_prefix: str = None,
    coordinate_einsum_prefix: str = None,
):
    """Given a set of einsum prefixes, returns the vectors `coordinates_b` transformed by the matrices `rotations_a_b`"""
    if rotation_einsum_prefix is None:
        rotation_einsum_prefix = result_einsum_prefix
    if coordinate_einsum_prefix is None:
        coordinate_einsum_prefix = result_einsum_prefix

    assert (
        len(rotation_einsum_prefix) == len(rotations_a_b.shape) - 2
    ), f"Rotation einsum prefix length must match rotations_a_b's prefix dimension length"
    assert (
        len(coordinate_einsum_prefix) == len(coordinates_b.shape) - 1
    ), "Coordinate einsum prefix must match coordinate_b's prefix dimension length"

    einsum_equation = f"{rotation_einsum_prefix}ij,{coordinate_einsum_prefix}j->{result_einsum_prefix}i"
    return torch.einsum(einsum_equation, rotations_a_b, coordinates_b)
