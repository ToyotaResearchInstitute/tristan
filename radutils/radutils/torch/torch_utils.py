import os
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


# TODO(guy.rosman): consolidate with other statistics functions, e.g. KLD.
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

    covariance2 = torch.matmul(data_m, data_m.transpose(-1, -2)) / (data_m.shape[-1] - 1)
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
    rotation_einsum_prefix: Optional[str] = None,
    coordinate_einsum_prefix: Optional[str] = None,
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


def split_batch(batch: dict, num_sub_batch: int):
    """Split a batch into multiple smaller batches.
    Return:
        a list of dict of batch_item
    """
    batch_size = list(batch.values())[0].shape[0]
    num_sub_batch = min(batch_size, num_sub_batch)

    chunk_size = batch_size // num_sub_batch
    if batch_size % num_sub_batch != 0:
        chunk_size += 1
    ret = _split_tensor(batch, num_sub_batch, batch_size, chunk_size)
    return ret


def get_current_process_cuda_memory():
    """Return current process CUDA memory in MB"""

    if not torch.cuda.is_available():
        return 0

    import pynvml  # type: ignore[import]

    try:
        # Result example
        # process       5454 uses      184.188 MB GPU memory
        # process      20385 uses      943.000 MB GPU memory
        pynvml.nvmlInit()
        cuda_usage_lines = []
        for device_idx in range(torch.cuda.device_count()):
            cuda_usage_lines.extend(torch.cuda.list_gpu_processes(device_idx).split("\n"))

    except pynvml.NVMLError as err:
        print(err)
        return 0
    finally:
        pynvml.nvmlShutdown()

    pid = str(os.getpid())
    ret = 0
    for line in cuda_usage_lines:
        if line[:7] != "process" or pid not in line:
            continue
        memory = line.split("MB")[0]
        ret = float(memory.split("uses")[-1])
    return ret


def _split_tensor(item: Union[dict, torch.Tensor], num_sub_batch, batch_size, chunk_size):
    if isinstance(item, dict):
        ret = [{} for _ in range(num_sub_batch)]
        for k, v in item.items():
            sub_tensors = _split_tensor(v, num_sub_batch, batch_size, chunk_size)
            for i in range(num_sub_batch):
                ret[i][k] = sub_tensors[i]
    elif isinstance(item, (torch.Tensor, list)):
        ret = []
        for i in range(0, batch_size, chunk_size):
            ret.append(item[i : i + chunk_size])
    else:
        raise RuntimeError("Unsupported item type: ", item)
    return ret


class CudaMemoryLimit:
    """This class is used to limit the CUDA memory usage.
    The memory allocated in
    """

    def __init__(self, device=None):
        if device is None:
            device = torch.cuda.current_device()
        if device is not None:
            if isinstance(device, int):
                cuda_device_index = device
            elif isinstance(device, torch.device):
                if device.type != "cpu":
                    cuda_device_index = device.index
                else:
                    cuda_device_index = None
            else:
                cuda_device_index = None
        else:
            cuda_device_index = None
        # All memory variables are in MB
        self.cuda_init_memory = 0
        self.cuda_total_memory = 0
        self.cuda_memory_limit = None
        self.cuda_device_index = cuda_device_index
        self.set_init_cuda_memory()

    def set_init_cuda_memory(self):
        self.cuda_init_memory = get_current_process_cuda_memory()

    def limit_memory(self, limit_percentage: float):
        """Set a limit to the max CUDA memory usage. It's a percentage of the total CUDA memory."""

        if self.cuda_device_index is None:
            print(
                f"WARNING: unable to set CUDA memory limit to {limit_percentage}, invalid device {self.cuda_device_index}"
            )
            return

        torch.cuda.set_per_process_memory_fraction(limit_percentage)
        import pynvml  # type: ignore[import]

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.cuda_device_index)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except pynvml.NVMLError_LibraryNotFound:
            return
        # In MB
        total_memory = mem_info.total / 1024 / 1024
        self.cuda_total_memory = total_memory
        self.cuda_memory_limit = limit_percentage * self.cuda_total_memory
        print("CUDA total memory", self.cuda_total_memory)

    def assert_cuda_memory_limit(self):
        """Assert the current CUDA memory is under the limit.
        Assert fails if CUDA memory exceeded limit.
        Return max CUDA memeory used, in GB.
        """
        if self.cuda_device_index is None or self.cuda_memory_limit is None:
            return 0.0
        # Two ways to calculate the max CUDA usage
        # 1. Get current process CUDA memory
        process_cuda = get_current_process_cuda_memory()
        # 2. add init CUDA memory and max_allocated memory
        max_allocated_MB = torch.cuda.max_memory_allocated(self.cuda_device_index) / 1024 / 1024
        mem_used = max(process_cuda, self.cuda_init_memory + max_allocated_MB)
        assert (
            mem_used < self.cuda_memory_limit
        ), f"CUDA out of memory. CUDA memory limit is {self.cuda_memory_limit}MB, but process used {mem_used}MB"
        # Return memory used in GB
        return mem_used / 1024
