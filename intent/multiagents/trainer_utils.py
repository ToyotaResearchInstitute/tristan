import torch

from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset


def reformat_tensor(original: torch.Tensor, index_tensor: torch.Tensor, num_dims: tuple) -> torch.Tensor:
    """Prepend dimensions to tensor and position data according to given indices

    An index tensor is used to inform the location in the newly added dimensions while
    num_dims informs the size of the newly added dimensions.

    Parameters
    ----------
    original : torch.Tensor
        Original data tensor of shape (batch_size, max_num_datapoints, b_1, ..., b_m)
    index_tensor : torch.Tensor
        Tensor of indices accessing original data of shape (batch_size, max_num_datapoints, n) or
        (batch_size, max_num_datapoints). In this tensor, the value -1 is used to indicate
        that the corresponding datapoint in original is assumed to be empty.
    num_dims : tuple
        Integer tuple of size n containing maximum number of elements per indexed dimension.

    Returns
    -------
    torch.Tensor
        Reformatted variant of the original data of final shape
        (batch_size, num_dims[0], ... num_dims[n-1], b_1, ..., b_m). This will be 0
        for everything that does not have any datapoint in the original data. The
        dtype and device of this tensor will be the same as of the original tensor.
    """
    batch_size, max_num_datapoints = index_tensor.shape[0:2]

    # Shape: (batch_size, num_dims[0], ... num_dims[n-1], b_1, ..., b_m)
    result = original.new_zeros((batch_size,) + num_dims + original.shape[2:])

    index_tensor_corrected = index_tensor
    if len(index_tensor.shape) == 2:
        index_tensor_corrected = index_tensor_corrected.unsqueeze(-1)

    # Shape: (batch_size, max_num_datapoints, 1)
    nonempty_datapoints_mask = (index_tensor_corrected != -1)[:, :, 0]

    # Shape: (num_nonempty_datapoints, b_1, ..., b_m)
    nonempty_datapoints = original[nonempty_datapoints_mask]

    # Shape: (batch_size, max_num_datapoints, 1)
    batch_idxs = (
        torch.arange(batch_size, dtype=index_tensor.dtype, device=index_tensor.device)
        .unsqueeze(-1)
        .repeat(1, max_num_datapoints)
        .unsqueeze(-1)
    )

    # Append batch indices to index tensor.
    # Shape: (batch_size, max_num_datapoints, n+1)
    index_tensor_with_batch = torch.cat((batch_idxs, index_tensor_corrected), dim=-1)

    # Shape:  (num_nonempty_datapoints, n+1)
    res_idxs = index_tensor_with_batch[nonempty_datapoints_mask]

    result[[res_idxs[:, i] for i in range(len(num_dims) + 1)]] = nonempty_datapoints

    return result


def get_prediction_item_sizes(batch_itm):
    """Extract sizes from the batch item.

    Parameters
    ----------
    batch_itm: dict
        A batch item dictionary.

    Returns
    -------
    batch_size: int
        The batch size
    num_agents: int
        The number of agents
    num_timestamps: int
        The number of time points
    num_past_timepoints: int
        The number of past time points
    num_future_timepoints: int
        The number of future time points
    """
    # Get relevant parameters.
    batch_size, num_agents, num_timestamps, _ = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape
    # Obtain num past and num future steps.
    num_past_timepoints = batch_itm[ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS][0].detach().cpu().item()
    num_future_timepoints = num_timestamps - num_past_timepoints
    return batch_size, num_agents, num_timestamps, num_past_timepoints, num_future_timepoints


def args_to_str(args_list):
    """
    Converts an argument list (as sys.argv) into a command line string. May not completely match the original
    commandline string.
    :param args_list: A list of the executable + arguments.
    :return:
    """
    commandstring = ""

    for arg in args_list:  # skip sys.argv[0] since the question didn't ask for it
        if " " in arg:
            commandstring += '"{}"  '.format(arg)  # Put the quotes back in
        else:
            commandstring += "{}  ".format(arg)  # Assume no space => no quotes

    return commandstring
