from abc import ABC, abstractmethod
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import torch
from torch import nn

from intent.multiagents.trainer_logging import TrainingLogger


def range_tensor(tensor, range_dim):
    """
    Computes a tensor with a linear range value [0,...shape[range_dim]] according to one of its axes.
    :param tensor: A tensor with the right device type and shape.
    :param range_dim: Which dimension to use for the value.
    :return:
    """
    grid_tensors = []
    for dim_i, dim in enumerate(tensor.shape):
        if dim_i == range_dim:
            grid_tensors.append(tensor.float().new_tensor(range(dim)))
        else:
            grid_tensors.append(tensor.float().new_ones(dim))
    res = torch.meshgrid(grid_tensors)
    return res[range_dim]


def random_rotation_perturbation(rot_mat, dispersion=0.01):
    """Randomly perturbs a set of given rotations.

    Parameters
    ----------
    rot_mat : torch.Tensor
        A set of rotation matrices of shape (*, 2, 2), where * is a non-zero arbitrary
        number of extra dimensions.
    dispersion : float
        Dispersion to apply when computing the rotations

    Returns
    -------
    torch.Tensor
        The perturbed rotation matrices of same shape as rot_mat.
    """

    rot_angles = torch.normal(
        torch.zeros(rot_mat.shape[:-2], dtype=rot_mat.dtype, device=rot_mat.device),
        dispersion * torch.ones(rot_mat.shape[:-2], dtype=rot_mat.dtype, device=rot_mat.device),
    )

    angles_sin = torch.sin(rot_angles)
    angles_cos = torch.cos(rot_angles)
    noise_mat = torch.stack((angles_cos, -angles_sin, angles_sin, angles_cos), dim=-1)
    noise_mat = noise_mat.reshape(rot_mat.shape)
    return torch.matmul(noise_mat, rot_mat)


class PredictionModelInterface(nn.Module, ABC):
    """
    A general class for prediction models that generate trajectory predictions.
    This class includes a 2DOF normalizing rotation to match each agent's current time coordinate frame.
    """

    logger: TrainingLogger

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.should_normalize = params["predictor_normalize"]
        self.predictor_local_rotation = params["predictor_local_rotation"]

    def set_logger(self, logger: TrainingLogger) -> None:
        """
        Set a logger inside the predictor, for instrumentation and data collection during training.

        Parameters
        ----------
        logger: TrainingLogger
            A logger object.

        Returns
        -------
        """
        self.logger = logger

    @abstractmethod
    def fuse_scene_tensor(self, additional_inputs: List, batch_size: int):
        """
        Fuse additional inputs, after encoding them. Does child dropout if it is set.


        Parameters
        ----------
        additional_inputs: List
            List of input tensors.
        batch_size: int
            Batch size

        Returns
        -------
        scene_inputs_tensor: Tensor
            Output tensor of size [batch_size,N_agents, total dimensionality of encoded inputs].
        """
        return None

    def compute_normalizing_transforms(
        self, trajectory_data, is_valid, timestamps, prediction_timestamp, time_weights_epsilon=1e-8
    ) -> torch.Tensor:
        """
        Compute 2DOF transformations to normalize the trajectory data such that that the trajectory starts at (0,0) and
        ends at y=0

        :param trajectory_data: [B x N_agents x N_timepoints x 3] tensor.
        :param is_valid: [B x N_agents x N_timepoints] tensor.
        :param prediction_timestamp: [B] tensor.
        :param time_weights_epsilon: epsilon adding to time weights. A small value puts more weights to recent time.
        :return: transforms_local_scene: [B x N_agents x 3 x 2] tensor, affine transform
        """
        dt = (timestamps - prediction_timestamp.unsqueeze(1)).abs()
        dt_relative = dt - dt.min(dim=1)[0].unsqueeze(1)
        time_weights = (dt_relative + time_weights_epsilon) ** (-1)
        tensored_range = range_tensor(is_valid, 2)
        # Compute local shifting vectors.
        x = (trajectory_data[:, :, :, 0] * is_valid.float() * time_weights.unsqueeze(1)).sum(dim=2) / (
            (is_valid.float() * time_weights.unsqueeze(1)).sum(dim=2) + 1e-10
        )
        y = (trajectory_data[:, :, :, 1] * is_valid.float() * time_weights.unsqueeze(1)).sum(dim=2) / (
            (is_valid.float() * time_weights.unsqueeze(1)).sum(dim=2) + 1e-10
        )
        if not self.should_normalize:
            x *= 0
            y *= 0

        # Compute local rotation matrix.
        if self.should_normalize and self.predictor_local_rotation:

            is_valid_nan = is_valid.clone().float()
            is_valid_nan[is_valid.logical_not()] = np.NaN

            range_nan = is_valid_nan * tensored_range
            range_min = range_nan.clone()
            range_min[range_min.isnan()] = 1e4
            idx_start, _ = (range_min).min(2, keepdim=True)
            mask_start = idx_start == tensored_range
            idx_end, _ = (is_valid * tensored_range).max(2, keepdim=True)
            mask_end = idx_end == tensored_range
            pos_start = (trajectory_data[:, :, :, :2] * mask_start.unsqueeze(3)).sum(2)
            pos_end = (trajectory_data[:, :, :, :2] * mask_end.unsqueeze(3)).sum(2)
            dt_agents = dt.repeat(trajectory_data.shape[1], 1, 1).permute(1, 0, 2)
            time_start = (dt_agents * mask_start).sum(2)
            time_end = (dt_agents * mask_end).sum(2)
            velocity = (pos_end - pos_start) / (time_end - time_start).abs().repeat(2, 1, 1).permute(1, 2, 0)
            velocity_v = pos_end - pos_start
            velocity_v[..., 0] += 1e-9  # Add epsilon to velocity x to avoid divide by 0 in computing R.
            v1 = (
                velocity_v
                / torch.norm(velocity_v, dim=-1).unsqueeze(dim=-1)
                * torch.tensor((1.0, -1.0)).to(velocity_v.device)
            )
            v2 = torch.stack((-v1[:, :, 1], v1[:, :, 0]), dim=-1)
            R = torch.stack((v1, v2), dim=-1)

            if self.params["local_rotation_noise"] is not None:
                R = random_rotation_perturbation(R, self.params["local_rotation_noise"])
        else:
            # Create an identity matrix, with shape [num_batch, num_agents, 2, 2].
            R = [
                torch.cat([torch.ones_like(x).unsqueeze(2), torch.zeros_like(x).unsqueeze(2)], 2).unsqueeze(2),
                torch.cat([torch.zeros_like(x).unsqueeze(2), torch.ones_like(x).unsqueeze(2)], 2).unsqueeze(2),
            ]
            R = torch.cat(R, 2)

        # Combine rotation matrix and transformation matrix.
        transforms_local_scene = torch.cat(
            [
                R[:, :, 0, 0].unsqueeze(2),
                R[:, :, 1, 0].unsqueeze(2),
                -x.unsqueeze(2),
                R[:, :, 0, 1].unsqueeze(2),
                R[:, :, 1, 1].unsqueeze(2),
                -y.unsqueeze(2),
            ],
            dim=2,
        )
        transforms_local_scene = transforms_local_scene.view(
            [transforms_local_scene.shape[0], transforms_local_scene.shape[1], -1, 3]
        ).transpose(2, 3)
        return transforms_local_scene

    @abstractmethod
    def normalize_trajectories(self, trajectory_data, is_valid, transforms, invert=False) -> torch.Tensor:
        """
        Normalize the trajectories
        :param trajectory_data: [B x N_agents x N_timepoints x 3] tensor.
        :param is_valid: [B x N_agents x N_timepoints] tensor.
        :param transforms: output from compute_normalizing_transforms, [B x N_agents x 3 x 2] tensor/
        :return: [B x N_agents x 3 x 2] tensor, affine transform.
        """
        return None

    @abstractmethod
    def forward(
        self,
        trajectory_data,
        additional_inputs,
        agent_additional_inputs,
        relevant_agents,
        agent_type,
        is_valid,
        timestamps,
        prediction_timestamp,
        additional_params=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """
        Main forward function -- generates predicted trajectories, plus additional outputs. Usually called by
        generate_trajectory.
        :param trajectory_data: a B x N_agents x N_timepoints x3 tensor
        :param additional_inputs: a dictionary of key -> additional input
        :param is_valid: [B x N_agents x N_timepoints] tensor.
        :return: A tuple of [B x N_agents x N_timepoints x2] tensor, [B x N_timepoints x joint decoding space] tensor,
            dictionary of auxiliary stats.
        """
        return None

    @abstractmethod
    def save_model(
        self,
        data,
        is_valid: bool,
        folder: str,
        checkpoint: dict = None,
        use_async: bool = False,
        save_to_s3: bool = False,
    ):
        """Save the model.

        Parameters:
        ---
        data : tensor
            The input data for the model
        is_valid :
            The validity data of the input data
        folder : str
            The folder to save the model
        checkpoint : dict
            The batch info and optimizer state
        """

    @abstractmethod
    def get_generator_parameters(self, require_grad=True) -> OrderedDict:
        return None

    @abstractmethod
    def get_discriminator_parameters(self, require_grad=True) -> OrderedDict:
        return None

    def get_latent_factors_parameters(self, require_grad=True) -> OrderedDict:
        return None

    @abstractmethod
    def generate_trajectory(
        self,
        input_trajectory,
        additional_inputs,
        agent_additional_inputs,
        relevant_agents,
        agent_type,
        is_valid,
        timestamps,
        prediction_timestamp,
        num_samples=1,
        additional_params: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """
        Generate a trajectory sample
        :param input_trajectory: a dictionary with key 'positions'
        :param is_valid: indicator of which agent/timestep is valid.
        :return:
        """
        return None

    @abstractmethod
    def discriminate_trajectory(
        self,
        past_trajectory,
        past_inputs,
        agent_additional_inputs,
        future_trajectory,
        expected,
        agent_type,
        is_past_valid,
        is_future_valid,
        timestamps,
        prediction_timestamp,
        relevant_agents,
        additional_param: dict,
    ):
        """
        :param input: a dictionary with key 'positions'
        :return:
        """
        return None

    @abstractmethod
    def set_num_agents(self, num_agents, num_past_points, num_future_timepoints):
        """
        Reset the graph structure to match the number of agents, timepoints
        :param num_agents:
        :param num_past_points:
        :return:
        """

    @abstractmethod
    def compute_generator_cost(
        self,
        past_trajectory,
        past_additional_inputs,
        agent_additional_inputs,
        predicted,
        expected,
        agent_type,
        is_valid,
        is_future_valid,
        timestamps,
        prediction_timestamp,
        semantic_labels,
        future_encoding,
        relevant_agents,
        label_weights,
        param,
        stats,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator training cost.

        Args:
        :param past_trajectory: Tensor of past trajectories (shape batch_size x num_agents x num_timesteps x 2)
        :param past_additional_inputs,
        :param agent_additional_inputs,
        :param predicted: Dictionary containing different predictions including the predicted "trajectories" (shape
            batch_size x num_agents x num_timesteps x 2 x num_samples)
        :param expected: Corresponding true label dictionary with entries having the same shapes as predicted except
            that there is no extra dimension for multiple predicted samples.
        :param agent_type,
        :param is_valid: Boolean tensor of indicating in which past timesteps an agent is visible.
            (shape batch_size x num_agents x num_timesteps)
        :param is_future_valid: Boolean tensor of indicating in which past timesteps an agent is visible.
            (shape batch_size x num_agents x num_timesteps)
        :param timestamps: Timestamps of the trajectory data of shape (batch_size x num_timesteps)
        :param prediction_timestamp: All timestamps larger than the timestamps given here are considered for ADE/FDE
            computation. Shape (batch_size,). This is necessary because the first entry of timestamps might not
            correspond to the timestamp of the first entry in predicted / observed.
        :param semantic_labels:
        :param future_encoding:
        :param relevant_agents:
        :param label_weights:
        :param param:
        :param stats:
        :param relevant_agents:
        :return:
        """
        return None

    @abstractmethod
    def get_semantic_keys(self) -> List[str]:
        return None

    @abstractmethod
    def compute_discriminator_cost(
        self,
        past_trajectory,
        additional_inputs,
        agent_additional_inputs,
        predicted_trajectories_scene,
        expected,
        agent_type,
        is_valid,
        is_future_valid,
        is_fake,
        timestamps,
        prediction_timestamp,
        relevant_agents,
        param,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Get discriminator cost for training.

        Parameters
        ----------
        past_trajectory: Tensor
            [B x N_agents x timesteps x 3] tensor, past positions.
        additional_inputs: Dict
            A dictionary of additional inputs. Each input should be of size [B x timesteps x..].
        agent_additional_inputs: Dict
            [B x N_agents x timesteps x 2 x sample set size], the set of predicted trajectories.
        predicted_trajectories_scene
        expected: Corresponding true label dictionary with entries having the same shapes as predicted except
            that there is no extra dimension for multiple predicted samples.
        agent_type: Tensor
            [B x N_agents x N_types], one-hot type representations
        is_valid: bool
            [B x N_agents x past_timesteps] tensor, are the past positions valid?
        is_future_valid: bool
            [B x N_agents x timesteps] tensor, are the future positions valid?
        is_fake: bool
            boolean, are these trajectories fake or not?
        timestamps
        prediction_timestamp
        relevant_agents
        param

        Returns
        -------

        """
        return None
