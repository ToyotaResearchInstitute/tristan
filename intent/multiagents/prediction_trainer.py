import collections
import copy
import datetime
import json
import os
import pickle
import pprint
import subprocess
import sys
import time
from collections import OrderedDict
from subprocess import check_output
from typing import Deque, Dict, Tuple, Union

import numpy as np
import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from typeguard import typechecked

import radutils.misc as rad_misc
import radutils.session_data as sess_data
import triceps.protobuf.proto_arguments as proto_args
from intent.multiagents.logging_handlers import LogWorstCases, SaveErrorStatistics
from intent.multiagents.trainer_logging import TerminalMessageLogger
from intent.multiagents.trainer_logging_tensorboard import TensorboardTrainingLogger
from intent.multiagents.trainer_logging_wandb import WandbTrainingLogger
from intent.multiagents.trainer_utils import args_to_str, get_prediction_item_sizes
from intent.multiagents.trainer_visualization import (
    visualize_histogram,
    visualize_itm,
    visualize_label_accuracy,
    visualize_sample_process,
)
from loaders.ado_key_names import AGENT_TYPE_NAME_MAP
from radutils.reproducibility import RunManifest
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_auxiliary import AuxiliaryStateIndices
from triceps.protobuf.prediction_dataset_map_handlers import MapDataIndices
from triceps.protobuf.snippet_pb_writer import convert_predictor_output_to_protobuf

ERROR_FILTER_TAPS = 10


def compute_global_normalization(input_tensor, is_valid):
    """
    Compute global normalization, through averaging all valid points
    :param input_tensor: normalized trajectory tensor,
        with shape [batch_size, num_agents, num_past_timepoints, trajectory_dimension].
    :param batch_positions_tensor: raw trajectory tensor,
        with shape [batch_size, num_agents, num_all_timepoints, trajectory_dimension].
    :param is_valid: validity of each time point,
        with shape [batch_size, num_agents, num_past_timepoints].
    :return: updated input_tensor, with normalization offsets (offset_x, offset_y)
    """
    num_agents = is_valid.shape[1]

    # Compute offsets as center of all valid positions.
    offset_x = (input_tensor[:, :, :, 0] * is_valid.float()).sum(axis=2).sum(axis=1) / (
        (1e-7 + is_valid.float()).sum(axis=2).sum(axis=1).detach()
    )
    offset_y = (input_tensor[:, :, :, 1] * is_valid.float()).sum(axis=2).sum(axis=1) / (
        (1e-7 + is_valid.float()).sum(axis=2).sum(axis=1).detach()
    )

    offset_x = offset_x.unsqueeze(1).repeat([1, num_agents])
    offset_y = offset_y.unsqueeze(1).repeat([1, num_agents])
    # Center trajectories.
    input_tensor[:, :, :, 0] -= offset_x.unsqueeze(2)
    input_tensor[:, :, :, 1] -= offset_y.unsqueeze(2)
    return input_tensor, offset_x, offset_y


def get_normalized_trajectories(
    params,
    batch_positions_tensor: torch.Tensor,
    is_valid: torch.Tensor,
    num_past_timepoints: int,
    num_future_timepoints: int,
):
    # Compute agent-wise normalized positions.
    if params["disable_position_input"]:
        shape = batch_positions_tensor.shape
        input_tensor = torch.zeros((shape[0], shape[1], num_past_timepoints, shape[3]))
        offset_x = torch.zeros(2)
        offset_y = torch.zeros(2)
    else:
        input_tensor = batch_positions_tensor[:, :, :num_past_timepoints, :].clone().detach()
        input_tensor, offset_x, offset_y = compute_global_normalization(input_tensor, is_valid)

    # Obtain centered observed future trajectory from data.
    expected_trajectories_scene = batch_positions_tensor[:, :, -num_future_timepoints:, :2].clone().detach()
    expected_trajectories_scene[:, :, :, 0] -= offset_x.unsqueeze(2)
    expected_trajectories_scene[:, :, :, 1] -= offset_y.unsqueeze(2)

    return input_tensor, expected_trajectories_scene, offset_x, offset_y


def get_inputs_from_batch_item(batch_item, num_past_timepoints):
    additional_inputs = {}
    if ProtobufPredictionDataset.DATASET_KEY_IMAGES in batch_item:
        additional_inputs[ProtobufPredictionDataset.DATASET_KEY_IMAGES] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_IMAGES
        ]
        additional_inputs[ProtobufPredictionDataset.DATASET_KEY_IMAGES_MAPPING] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_IMAGES_MAPPING
        ]

    if ProtobufPredictionDataset.DATASET_KEY_STUB in batch_item:
        additional_inputs[ProtobufPredictionDataset.DATASET_KEY_STUB] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_STUB
        ][:, :num_past_timepoints, ...]

    if ProtobufPredictionDataset.DATASET_KEY_MAP_STUB in batch_item:
        additional_inputs[ProtobufPredictionDataset.DATASET_KEY_MAP_STUB] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_MAP_STUB
        ]

    if ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE in batch_item:
        additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE
        ]

    # Load additional inputs of each agent.
    agent_additional_inputs = {}
    if ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES in batch_item:
        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES
        ]
        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_MAPPING] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_MAPPING
        ]
    if ProtobufPredictionDataset.DATASET_KEY_AGENT_STUB in batch_item:
        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AGENT_STUB] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_AGENT_STUB
        ][:, :, :num_past_timepoints, :]
    if ProtobufPredictionDataset.DATASET_KEY_AGENT_MAP_STUB in batch_item:
        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AGENT_MAP_STUB] = batch_item[
            ProtobufPredictionDataset.DATASET_KEY_AGENT_MAP_STUB
        ][:, :, :num_past_timepoints, :]
    if ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE in batch_item:
        # Auxliary state includes (length, width, yaw, vel_x, vel_y) per time step per agent.
        auxiliary_state = batch_item[ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE]
        auxiliary_state = auxiliary_state[:, :, :num_past_timepoints, :].float()
        # Add absolute velocity to auxiliary state.
        auxiliary_state_vel_abs = torch.sqrt(
            auxiliary_state[..., AuxiliaryStateIndices.AUX_STATE_IDX_VEL_X] ** 2
            + auxiliary_state[..., AuxiliaryStateIndices.AUX_STATE_IDX_VEL_Y] ** 2
        )
        auxiliary_state = torch.cat([auxiliary_state, auxiliary_state_vel_abs.unsqueeze(-1)], -1)
        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE] = auxiliary_state

    return additional_inputs, agent_additional_inputs


def preprocess_batch_item(batch_item, params):
    scale = params["predictor_normalization_scale"]
    # Read semantic labels
    if ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_LABELS in batch_item:
        semantic_labels = batch_item[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_LABELS].float()
    else:
        semantic_labels = None
    # Read positions, validity, timestamps.
    # Scale down positions.
    batch_positions_tensor = batch_item[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, :].float() * scale
    # TODO(guy.rosman): GENERALIZATION - encapsulate reading of batch data items (inputs/outputs)
    batch_is_valid = batch_item[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 2]
    batch_timestamps_tensor = batch_item[ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS][:, :].float()

    # Read DOT keys, agent types.
    batch_is_rel = batch_item[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT]  # one-hot is_relevant_pedestrian
    batch_agent_type = batch_item[ProtobufPredictionDataset.DATASET_KEY_AGENT_TYPE].float()

    # Add ego car to relevant agent.
    if not params["ignore_ego"]:
        batch_is_ego = batch_item[ProtobufPredictionDataset.DATASET_KEY_IS_EGO_VEHICLE]  # one-hot is_ego_vehicle
        batch_is_rel = batch_is_ego + batch_is_rel

    prediction_timestamp = batch_item[ProtobufPredictionDataset.DATASET_KEY_PREDICTION_TIMESTAMP].float()

    return (
        batch_positions_tensor,
        batch_is_valid,
        batch_is_rel,
        batch_agent_type,
        batch_timestamps_tensor,
        prediction_timestamp,
        semantic_labels,
    )


def scale_maps(map_input_type, scale, batch_itm, agent_additional_inputs, offset_x, offset_y):
    """Normalizes the scales and offsets of the maps to get the "global normalization".

    Parameters
    ----------
    scale : float
        The correction scale.
    batch_itm : dict
        The dataloader dictionary
    agent_additional_inputs: dict
        The additional input tensors dictionary.
    offset_x : torch.Tensor,
        The x offset for instances.
    offset_y : torch.Tensor,
        The y offset for instances.

    Returns:
    -------
    batch_map_positions_tensor : torch.Tensor
        The positions of the map elements of shape (batch_size, max_point_num, 2)
    batch_map_validity_tensor : torch.Tensor
        The positions of the map elements of shape (batch_size, max_point_num)
    batch_map_others_tensor : torch.Tensor
        Point type, tangent information, normal information, point id.
        (batch_size, max_point_num, 6). The elements in the last 6 dimensions are
        structured, refer to DATASET_KEY_MAP,
        where point type is corresponding to the integer in MapPointType.
    """
    if offset_x.shape[1] > 1:
        assert offset_x.std(1).sum() < 1e-15
        assert offset_y.std(1).sum() < 1e-15
    num_past_timepoints = batch_itm[ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS][0].detach().cpu().item()
    batch_map_positions_tensor = None
    batch_map_validity_tensor = None
    batch_map_others_tensor = None
    if ProtobufPredictionDataset.DATASET_KEY_MAP in batch_itm:
        if map_input_type == "point":
            batch_map_positions_tensor = batch_itm[ProtobufPredictionDataset.DATASET_KEY_MAP][..., :2].float() * scale
            batch_map_validity_tensor = batch_itm[ProtobufPredictionDataset.DATASET_KEY_MAP][
                ..., MapDataIndices.MAP_IDX_VALIDITY
            ].float()
            batch_map_others_tensor = batch_itm[ProtobufPredictionDataset.DATASET_KEY_MAP][
                ..., MapDataIndices.MAP_IDX_TYPE :
            ].float()
        else:
            agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_MAP] = batch_itm[
                ProtobufPredictionDataset.DATASET_KEY_MAP
            ][:, :num_past_timepoints, :, :, :]

        # Normalize map points in global frame (not per agent).
        if map_input_type == "point":
            agent_additional_inputs[
                ProtobufPredictionDataset.DATASET_KEY_MAP
            ] = update_global_normalized_map_point_tensors(
                batch_map_positions_tensor,
                batch_map_validity_tensor,
                batch_map_others_tensor,
                offset_x,
                offset_y,
            )

    return batch_map_positions_tensor, batch_map_validity_tensor, batch_map_others_tensor


def update_global_normalized_map_point_tensors(
    batch_map_positions_tensor: torch.Tensor,
    batch_map_validity_tensor: torch.Tensor,
    batch_map_others_tensor: torch.Tensor,
    offset_x,
    offset_y,
):
    """
    Compute global map tensors through shifting.
    :param batch_map_positions_tensor: original map positions,
        with shape [batch_size, num_max_points, point_dimension].
    :param batch_map_validity_tensor: map validity,
        with shape [batch_size, num_max_points].
    :param batch_map_others_tensor: additional map information,
        with shape [batch_size, num_max_points, additional_info_dimension].
    :param offset_x: delta x for global map centering, with shape [batch_size, num_agents].
    :param offset_y: delta y for global map centering, with shape [batch_size, num_agents].
    :return: Updated map inputs tensor.
    """
    # Read raw inputs.
    input_map_tensor = batch_map_positions_tensor.clone().detach()
    input_map_tensor_validity = batch_map_validity_tensor.clone().detach()
    input_map_tensor_others = batch_map_others_tensor.clone().detach()

    map_x = batch_map_positions_tensor[..., 0] - offset_x[:, :1]
    map_y = batch_map_positions_tensor[..., 1] - offset_y[:, :1]
    map_xy = torch.stack((map_x, map_y), dim=-1)

    input_map_tensor[..., 0] = map_xy[..., 0] * input_map_tensor_validity
    input_map_tensor[..., 1] = map_xy[..., 1] * input_map_tensor_validity
    input_map_tensor = torch.cat(
        (input_map_tensor, input_map_tensor_validity.unsqueeze(-1), input_map_tensor_others), axis=-1
    )
    # Get map output, with shape [num_batch, num_max_points, point_dim].
    return input_map_tensor


def _initialize_session(params: Dict) -> Tuple[int, str]:
    # Session name for model, log saving.
    param_level = rad_misc.get_param_level_from_session_id(params["resume_session_name"])
    # Always create a new session name
    tf_session_name = rad_misc.create_or_resume_session(
        param_level=param_level,
        session_name=params["log_name_string"],
        resume_session_name=params["resume_session_name"],
        current_session_name=params["current_session_name"],
        path_to_model_directory=params["model_load_folder"],
    )
    sess_data.save_session_values(tf_session_name, params, proto_args.path_keys_list())
    return param_level, tf_session_name


class PredictionProtobufTrainer:
    def __init__(
        self, datasets: dict, prediction_model: nn.Module, trainer_param: dict, device: torch.device, debugger=None
    ):
        """
        :param datasets: pytorch datasets with examples.
        :param prediction_model: The model to train.
        :param trainer_param: a dictionary of parameters.
        :param device: model device.
        :param debugger: TrainerDebugger for testing
        """
        self.device = device
        self.profiler = None
        self.prediction_model = prediction_model
        # Set device.
        self.prediction_model.to(device)
        self.inference_mode = trainer_param.get("inference_mode", False)
        self.message_logger = TerminalMessageLogger()

        # TODO(guy.rosman): make discriminator training optional - optimizers, schedulers.
        # TODO(guy.rosman): Save model for tensorRT.
        assert "train" in datasets.keys()
        assert "validation" in datasets.keys()
        self.trainer_param = trainer_param
        self.message_logger = TerminalMessageLogger()
        self.datasets = datasets
        self.distributed_training = self.trainer_param["multigpu"]
        if self.distributed_training:
            self.message_logger.log_message("{} GPUs.".format(torch.cuda.device_count()))
            self.prediction_model = nn.DataParallel(self.prediction_model)

        # Flags for model configurations.
        # TODO(mark.flanagan.ctr): handle this logic while parsing parameters
        self.use_discriminator = trainer_param["use_discriminator"] and not trainer_param.get("use_linear_model")
        self.use_semantics = trainer_param["use_semantics"]
        self.use_latent_factors = trainer_param["use_latent_factors"]
        self.predictor_local_rotation = trainer_param["predictor_local_rotation"]

        # Distributed training.
        if self.distributed_training:
            self.nondistributed_prediction_model = self.prediction_model.module
        else:
            self.nondistributed_prediction_model = self.prediction_model

        # Define parameters.
        self.generator_params = self.nondistributed_prediction_model.get_generator_parameters(require_grad=True)
        if self.generator_params is not None:
            self.generator_params = list(self.generator_params.values())

            # Optimizers for generator/discriminator.
            self.g_optimizer = optim.Adam(
                self.generator_params,
                trainer_param["learning_rate"],
                weight_decay=trainer_param["weight_decay"],
                eps=trainer_param["learning_epsilon"],
                betas=(0.5, 0.999),
            )
        else:
            self.g_optimizer = None
        if "resume_optimizer_state" in trainer_param and "g_optimizer_state" in trainer_param["resume_optimizer_state"]:
            try:
                self.g_optimizer.load_state_dict(trainer_param["resume_optimizer_state"]["g_optimizer_state"])
            except ValueError:
                self.message_logger.log_message("Failed to load g_optimizer state")
        self.g_optimizer_steps = 0
        if self.use_discriminator:
            self.discriminator_params = list(
                self.nondistributed_prediction_model.get_discriminator_parameters(require_grad=True).values()
            )
            self.d_optimizer = optim.Adam(
                self.discriminator_params,
                trainer_param["learning_rate"] * trainer_param["discriminator_learning_ratio"],
                weight_decay=trainer_param["weight_decay"],
                eps=trainer_param["learning_epsilon"],
                betas=(0.5, 0.999),
            )
            if (
                "resume_optimizer_state" in trainer_param
                and "d_optimizer_state" in trainer_param["resume_optimizer_state"]
            ):
                try:
                    self.d_optimizer.load_state_dict(trainer_param["resume_optimizer_state"]["d_optimizer_state"])
                except ValueError:
                    self.message_logger.log_message("Failed to load optimizer state")

            self.d_optimizer_steps = 0

        # Schedulers for generator/discriminator.
        self.g_scheduler = None
        if self.use_discriminator:
            self.d_scheduler = None
        if self.trainer_param["learning_rate_milestones"] is not None:
            milestones = [int(x) for x in self.trainer_param["learning_rate_milestones"]]
            self.g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.g_optimizer, milestones, gamma=self.trainer_param["learning_rate_gamma"]
            )
            if self.use_discriminator:
                self.d_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.d_optimizer, milestones, gamma=self.trainer_param["learning_rate_gamma"]
                )

        _, self.tf_session_name = _initialize_session(self.trainer_param)
        self.run_manifest = RunManifest(self.tf_session_name, self.trainer_param)

        self.start_epoch = 0
        self.start_global_batch_cnt = 0
        # Loading model for resuming
        if self.trainer_param["resume_session_name"]:
            self.message_logger.log_message(f"Loading model for session {self.trainer_param['resume_session_name']}")
            self.load_models()  # Must create session to download models before this
        self.total_epoch = self.start_epoch + self.trainer_param["num_epochs"]
        self.residual_computation = self.trainer_param["log_residuals"]
        # Parameter copy for residuals.
        if self.residual_computation:
            self.p_g_copy = None
            # For individual input encoders
            self.p_g_inputs_encoders_copy = {}
            self.p_g_agent_inputs_encoders_copy = {}
            if self.use_discriminator:
                self.p_d_copy = None
                # For individual input encoders
                self.p_d_inputs_encoders_copy = {}
                self.p_d_agent_inputs_encoders_copy = {}
            if self.use_latent_factors:
                self.p_latent_factors_copy = {}

        self.log_folder = os.path.expanduser(os.path.join(self.trainer_param["logs_dir"], self.tf_session_name))
        trainer_param["artifacts_folder"] = os.path.join(trainer_param["artifacts_folder"], self.tf_session_name)
        trainer_param["runner_output_folder"] = os.path.join(
            trainer_param["artifacts_folder"], trainer_param["runner_output_folder"]
        )
        os.makedirs(trainer_param["artifacts_folder"], exist_ok=True)
        os.makedirs(trainer_param["runner_output_folder"], exist_ok=True)

        if not self.trainer_param["logger_type"] == "none":
            if self.trainer_param["logger_type"] == "tensorboard":
                self.logger = TensorboardTrainingLogger(
                    self.trainer_param, self.device.type, session_name=self.tf_session_name
                )
            elif self.trainer_param["logger_type"] == "weights_and_biases":
                self.logger = WandbTrainingLogger(
                    self.trainer_param, self.device.type, session_name=self.tf_session_name
                )

            # TODO(igor.gilitschenski): This is basically a hack to determine whether we are in evaluation or training
            # TODO  mode.
            self.worst_cases_folder = None
        else:
            self.worst_cases_folder = self.trainer_param["runner_output_folder"]
            self.logger = None
        self.message_logger.log_message("Session: " + self.tf_session_name)

        current_commit_hash = rad_misc.get_current_commit_hash()
        current_commit_comment = self.get_current_commit_comment()
        commandline_args = str(sys.argv)
        commandline = args_to_str(sys.argv)
        if self.logger:
            if hasattr(self.prediction_model, "set_logger"):
                self.prediction_model.set_logger(self.logger)

            self.logger.add_text("commit", f"{current_commit_hash}\n{current_commit_comment}")
            self.logger.add_text("sys.argv", commandline_args)
            self.logger.add_text("sys.argv_string", commandline)
            if "params_key" in self.trainer_param:
                self.logger.add_text("training_schedule_key", self.trainer_param["params_key"])
            params_to_save = copy.copy(self.trainer_param)
            del params_to_save["datasets_name_lists"]
            self.logger.add_text("params", pprint.pformat(params_to_save, indent=2).replace("\n", "  \n"))
            self.logger.add_text(
                "datasets_name_lists",
                pprint.pformat(self.trainer_param["datasets_name_lists"], indent=2).replace("\n", "  \n"),
            )
            for key in self.datasets.keys():
                self.logger.add_text(f"{key} dataset_length", str(len(self.datasets[key])))
        self.visualize_items = self.trainer_param["visualize_items"]

        # Define semantics weights.
        if (
            not self.inference_mode
            and "latent_factors_trainer_callback" in self.trainer_param
            and not self.trainer_param["disable_label_weights"]
        ):
            self.latent_factor_callback = self.trainer_param["latent_factors_trainer_callback"]
            self.latent_factor_callback.trainer_init(datasets, self.logger)
        else:
            self.latent_factor_callback = None

        self.additional_trainer_callbacks = self.trainer_param.get("additional_trainer_callbacks", [])

        self._dataloader_pin_memory = trainer_param.get("dataloader_pin_memory", True)
        self._dataloader_pre_fetch_factor = trainer_param.get("dataloader_pre_fetch_factor")

        # Regression Test early stop
        self._early_stop = trainer_param.get("regression_test_early_stop", False)
        self._early_stop_error = trainer_param.get("regression_test_early_stop_test_error", "")
        self._early_stop_error_type = trainer_param.get("regression_test_early_stop_test_error_type", None)

        self.dataloaders_info = self.init_dataloaders()
        self.training_start_time = None
        self.last_saved_model_hour = 0
        self.debugger = debugger

    def init_dataloaders(self):
        # TODO(guy.rosman): GENERALIZATION - allow adding other dataloaders into the OrderedDict(), such as test set,
        # TODO  datasets for additional cost terms, etc.
        dataloaders_info = OrderedDict()

        evaluation_dataset = self.trainer_param.get("evaluation_dataset", None)
        train_only = evaluation_dataset == "train"
        validation_only = evaluation_dataset == "validation"

        # TODO(guy.rosman): GENERALIZATION - think how to encapsulate inference mode -- also maybe encapsulate data loop
        # TODO  for inference, training, etc.
        if self.inference_mode:
            assert not self.trainer_param["disable_validation"], "Cannot disable validation in inference mode"

            msg = "Evaluation Dataset must be set to 'validation' if inference mode is enabled"
            assert validation_only or not evaluation_dataset, msg

        if self.inference_mode:
            use_validation = True

            use_train = use_viz = False
        else:
            use_train = not validation_only
            use_validation = not (train_only or self.trainer_param["disable_validation"])
            use_viz = not (evaluation_dataset or self.trainer_param["disable_visualizations"])

        if use_train:
            dataloaders_info["train"] = {
                "dataset_key": "train",
                "dataset": self.datasets["train"],
                "epoch_size": self.trainer_param["epoch_size"],
                "num_workers": self.trainer_param["num_workers"],
                "batch_size": self.trainer_param["batch_size"],
            }

        if use_validation:
            dataloaders_info["validation"] = {
                "dataset_key": "validation",
                "dataset": self.datasets["validation"],
                "epoch_size": self.trainer_param["val_epoch_size"],
                "num_workers": self.trainer_param["val_num_workers"],
                "batch_size": self.trainer_param["val_batch_size"],
            }

        if use_viz:
            dataloaders_info["vis"] = {
                "dataset_key": "validation",
                "dataset": self.datasets["validation"],
                "epoch_size": self.trainer_param["vis_epoch_size"],
                "num_workers": self.trainer_param["vis_num_workers"],
                "batch_size": self.trainer_param["vis_batch_size"],
            }

        return dataloaders_info

    @staticmethod
    def smoothed_error(stats: Deque) -> float:
        # Just use a moving average
        error = sum(stats) / len(stats)
        return error

    def should_early_stop(self, stats: Deque):
        if self._early_stop and stats:
            error = self.smoothed_error(stats)
            if error < self._early_stop_error:
                print(f"error {self._early_stop_error_type} achieved goal: {self._early_stop_error}. Stop training")
                return True
        return False

    @typechecked
    def save_models(
        self,
        cur_iter: int,
        global_batch_cnt: int,
        input_tensor: torch.Tensor,
        is_valid: torch.Tensor,
        folder_postfix: str = "",
    ) -> None:
        """Save both torch models for training/resuming/testing, and TensorRT models for deployment.

        Parameters
        ----------
        cur_iter: int
          The training epoch.
        global_batch_cnt: int
          The global training batch count.
        input_tensor: tensor
          Data used for model saving. Currently a tensor, but not used, we may need a dictionary for different inputs.
        is_valid: tensor
          Validity data, similar to input tensor.
        folder_postfix: str
          A string to dictate different folder names to save ongoing models vs. best so far. Defaults to ''.

        """
        if self.trainer_param["disable_model_saving"]:
            return
        save_folder = os.path.join(self.trainer_param["model_save_folder"], self.tf_session_name + folder_postfix)
        os.makedirs(save_folder, exist_ok=True)
        # Save models
        # Save optimizers
        fname_checkpoint = os.path.join(save_folder, "checkpoint.tar")
        checkpoint = {
            "epoch": cur_iter + 1,
            "global_batch_cnt": global_batch_cnt,
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "g_optimizer_steps": self.g_optimizer_steps,
        }
        if self.use_discriminator:
            checkpoint.update(
                {
                    "d_optimizer_state_dict": self.d_optimizer.state_dict(),
                    "d_optimizer_steps": self.d_optimizer_steps,
                }
            )
        # Save learning rate schedulers
        if self.trainer_param["learning_rate_milestones"] is not None:
            checkpoint.update({"g_scheduler_state_dict": self.g_scheduler.state_dict()})
            if self.use_discriminator:
                checkpoint.update({"d_scheduler_state_dict": self.d_scheduler.state_dict()})
        self.message_logger.log_message(
            f"Saving checkpoint to {fname_checkpoint}, at iteration {cur_iter}, {folder_postfix}"
        )
        self.nondistributed_prediction_model.save_model(
            input_tensor,
            is_valid.bool(),
            save_folder,
            checkpoint=checkpoint,
            use_async=self.trainer_param["async_save"],
            save_to_s3=self.trainer_param["save_to_s3"],
        )

        params_filename = os.path.join(save_folder, "params.pkl")
        # Create a copy without modules, for saving
        saved_params = copy.copy(self.trainer_param)
        if "latent_factors" in saved_params:
            del saved_params["latent_factors"]
        if "latent_factors_trainer_callback" in saved_params:
            del saved_params["latent_factors_trainer_callback"]
        if "additional_trainer_callbacks" in saved_params:
            del saved_params["additional_trainer_callbacks"]
        with open(params_filename, "wb") as fp:
            pickle.dump(saved_params, fp)
        self.save_run_manifest(save_folder)
        file_lists_file = os.path.join(save_folder, "data_files.json")
        self.message_logger.log_message(f"Saving split to {file_lists_file}, at epoch {cur_iter}")
        if "datasets_name_lists" in saved_params and saved_params["datasets_name_lists"] != "none":
            file_lists = saved_params["datasets_name_lists"]
            with open(file_lists_file, "w") as fp:
                json.dump(file_lists, fp, indent=2)

    def load_models(self):
        """Load torch models for training/resuming/testing, and TensorRT models for deployment.
        :return:
        """
        load_folder = os.path.join(self.trainer_param["model_load_folder"], self.trainer_param["resume_session_name"])
        # Load models
        self.nondistributed_prediction_model.load_model(load_folder)
        # Load checkpoint (epoch, optimizers and learning rate schedulers)
        fname_checkpoint = os.path.join(load_folder, "checkpoint.tar")
        self.message_logger.log_message(f"Loading checkpoint from {fname_checkpoint}")
        checkpoint = torch.load(fname_checkpoint, map_location=self.device)
        self.start_epoch = checkpoint["epoch"]
        self.start_global_batch_cnt = checkpoint["global_batch_cnt"]
        if self.trainer_param["resume_optimizer"]:
            # Note: optimizer.load_state_dict() doesn't have "strict" flag to load partial state
            # Besides, the loaded state dict should match the size of optimizer's group
            self.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            self.g_optimizer_steps = checkpoint["g_optimizer_steps"]
            if self.use_discriminator:
                self.d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
                self.d_optimizer_steps = checkpoint["d_optimizer_steps"]
            # Load learning rate schedulers
            if self.trainer_param["learning_rate_milestones"] is not None:
                self.g_scheduler.load_state_dict(checkpoint["g_scheduler_state_dict"])
                if self.use_discriminator:
                    self.d_scheduler.load_state_dict(checkpoint["d_scheduler_state_dict"])

    def save_run_manifest(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        filename = os.path.join(save_folder, "run_manifest.json")
        with open(filename, "w") as f:
            json.dump(self.run_manifest.serialize(), f, indent=4, default=str)

    @staticmethod
    def augment_data(trainer_param, batch_itm: dict, encoder_keys: list) -> None:
        """Do data augmentation on batch_itm, rotation/translating/adding noise.

        The augmentation happens "in-place". Thus, no value is returend.

        Parameters
        ----------
        batch_itm : dict
            The dataloader dictionary
        """
        # TODO(guy.rosman): Fix augmentation for map inputs
        _, num_agents, num_timestamps, _ = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape

        # Augment trajectories by rotation.
        # TODO(guy.rosman): move to function.
        augmentation_angles = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, 0, 0, 0].clone()
        augmentation_angles.uniform_()
        augmentation_angles -= 0.5
        augmentation_angles *= trainer_param["augmentation_angle_scale"] * 2.0

        sn = torch.sin(augmentation_angles).float()
        cs = torch.cos(augmentation_angles).float()

        augmentation_offset_x = (
            batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, 0, 0, 0].clone().normal_()
            * trainer_param["augmentation_translation_scale"]
        )
        augmentation_offset_y = (
            batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, 0, 0, 0].clone().normal_()
            * trainer_param["augmentation_translation_scale"]
        )
        # A noise source for every timestep
        random_walk = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, :2].float().clone().normal_()

        # Add a random walk with spring force, to prevent a big difference from the source.
        spring_force = trainer_param["augmentation_process_noise_spring"]
        for t in range(batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape[2]):
            # Set an additional force -- either spring force to bound the process noise, or 0 (first time step)
            if t > 0:
                dprocess_noise_t = -random_walk[:, :, t - 1, :] * spring_force

                # accumulate random steps
                random_walk[:, :, t, :] += random_walk[:, :, t - 1, :]
            else:
                dprocess_noise_t = 0

            random_walk[:, :, t, :] += dprocess_noise_t
        random_walk *= trainer_param["augmentation_process_noise_scale"]

        if (
            ProtobufPredictionDataset.DATASET_KEY_MAP in batch_itm
            and ProtobufPredictionDataset.DATASET_KEY_MAP in encoder_keys
        ):
            # Currently map handlers do not transform their content. Hence, we prevent augmentation.
            cs = torch.ones_like(cs)
            sn = torch.zeros_like(sn)
            augmentation_offset_x *= 0
            augmentation_offset_y *= 0
        dx = (
            batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 0].float().clone().normal_()
            * trainer_param["augmentation_noise_scale"]
        )
        dy = (
            batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 0].float().clone().normal_()
            * trainer_param["augmentation_noise_scale"]
        )
        nx = (
            batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 0].float()
            * cs.unsqueeze(-1).unsqueeze(-1).repeat(1, num_agents, num_timestamps).float()
            - batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 1].float()
            * sn.unsqueeze(-1).unsqueeze(-1).repeat(1, num_agents, num_timestamps).float()
            + augmentation_offset_x.unsqueeze(-1).unsqueeze(-1).repeat(1, num_agents, num_timestamps).float()
            + dx
            + random_walk[:, :, :, 0]
        )
        ny = (
            batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 0].float()
            * sn.unsqueeze(-1).unsqueeze(-1).repeat(1, num_agents, num_timestamps).float()
            + batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 1].float()
            * cs.unsqueeze(-1).unsqueeze(-1).repeat(1, num_agents, num_timestamps).float()
            + augmentation_offset_y.unsqueeze(-1).unsqueeze(-1).repeat(1, num_agents, num_timestamps).float()
            + dy
            + random_walk[:, :, :, 1]
        )

        batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 0] = nx
        batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][:, :, :, 1] = ny

    def save_example_outputs(self, batch_itm, predicted_trajectories_scene, decoding, stats):
        """Save example protobuf outputs (to mimic deployment).

        Parameters
        ----------
        batch_itm : dict
            The dataloader dictionary.
        predicted_trajectories_scene: torch.Tensor
            The emitted trajectory outputs.
        stats: dict
            A dictionary of additional statistics (such as labels).

        """
        batch_size, _, _, _ = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape
        predicted_trajectories_examples = convert_predictor_output_to_protobuf(
            predicted_trajectories_scene,
            batch_itm["dot_keys"],
            decoding,
            stats,
            timestep=self.trainer_param["future_timestep_size"],
            t0=0.0,
        )
        examples_save_folder = os.path.join(self.trainer_param["model_save_folder"], self.tf_session_name)
        os.makedirs(examples_save_folder, exist_ok=True)
        for save_sample_i, save_sample in enumerate(predicted_trajectories_examples):
            samples_save_fname = os.path.join(
                examples_save_folder, "output_example_protobufs_{:08d}.pb".format(save_sample_i + iter * batch_size)
            )
            with open(samples_save_fname, "wb") as pbfile:
                pbfile.write(save_sample.SerializeToString())

        # TODO(guy.rosman): Verify the format before removing this .embed()
        import IPython

        IPython.embed(header="save results -- this is a temporary specification, should verify")

    def optimize_generator(self, g_cost, averages):
        """Perform the generator optimization, gradient accumulation and residual computation.

        Parameters
        ----------
        g_cost: torch.Tensor scalar
            Generator cost.
        averages: dict, including:
            avg_g_input_encoders_residual: dict
                Residuals of the input encoders
            avg_g_agent_input_encoders_residual: dict
                Residuals of the agent input encoders
            avg_g_residual: scalar
                Accumulator for g residuals, averaged over the epoch.
            avg_g_cost: scalar
                Accumulator of g costs, averaged over the epoch.
            avg_latent_factors_residual: scalar
                Accumulator of latent factors residuls, averaged over the epoch.
        """
        assert not torch.isnan(g_cost).any()
        # Save parameters for residual computation later on.
        if self.residual_computation and self.p_g_copy is None:
            self.p_g_copy = [
                x.cpu().detach().numpy()
                for x in self.nondistributed_prediction_model.get_generator_parameters().values()
            ]
            for key, module in self.nondistributed_prediction_model.get_agent_input_encoders().items():
                param_vec = np.concatenate(
                    [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                )
                self.p_g_agent_inputs_encoders_copy[key] = param_vec

            for key, module in self.nondistributed_prediction_model.get_input_encoders().items():
                param_vec = np.concatenate(
                    [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                )
                self.p_g_inputs_encoders_copy[key] = param_vec
            if self.use_latent_factors:
                self.p_latent_factors_copy = [
                    x.cpu().detach().numpy()
                    for x in self.nondistributed_prediction_model.get_latent_factors_parameters().values()
                ]

        if not self.trainer_param["disable_optimization"]:
            g_cost.backward()

            if self.trainer_param["clip_gradients"]:
                threshold = self.trainer_param["clip_gradients_threshold"]
                torch.nn.utils.clip_grad_norm_(
                    list(self.nondistributed_prediction_model.get_generator_parameters().values()), threshold
                )

        self.g_optimizer_steps += 1
        if (
            self.g_optimizer_steps % self.trainer_param["optimizer_step_batch"] == 0
            and not self.trainer_param["disable_optimization"]
        ):
            self.g_optimizer.step()
            # set_to_none - Based on https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            self.g_optimizer.zero_grad(set_to_none=True)

        if torch.isinf(g_cost).sum() or torch.isnan(g_cost).sum() > 0:
            import IPython

            IPython.embed(header="inf generator cost")
        if self.residual_computation:
            new_p_g_copy = [
                x.cpu().detach().numpy()
                for x in self.nondistributed_prediction_model.get_generator_parameters().values()
            ]
            input_residuals = {}
            agent_input_residuals = {}
            for key, module in self.nondistributed_prediction_model.get_agent_input_encoders().items():
                param_vec = np.concatenate(
                    [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                )
                agent_input_residuals[key] = ((self.p_g_agent_inputs_encoders_copy[key] - param_vec) ** 2).sum()
                self.p_g_agent_inputs_encoders_copy[key] = param_vec
                if key not in averages["avg_g_agent_input_encoders_residual"]:
                    averages["avg_g_agent_input_encoders_residual"][key] = 0
                averages["avg_g_agent_input_encoders_residual"][key] += agent_input_residuals[key]

            for key, module in self.nondistributed_prediction_model.get_input_encoders().items():
                param_vec = np.concatenate(
                    [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                )
                input_residuals[key] = ((self.p_g_inputs_encoders_copy[key] - param_vec) ** 2).sum()
                self.p_g_inputs_encoders_copy[key] = param_vec
                if key not in averages["avg_g_input_encoders_residual"]:
                    averages["avg_g_input_encoders_residual"][key] = 0
                averages["avg_g_input_encoders_residual"][key] += input_residuals[key]

            residual = np.sum([((x - y) ** 2).sum() for x, y in zip(new_p_g_copy, self.p_g_copy)])
            self.p_g_copy = new_p_g_copy
            averages["avg_g_residual"] += residual

            # Compute residuals for latent factors if using it.
            if self.use_latent_factors:
                new_p_latent_factors_copy = [
                    x.cpu().detach().numpy()
                    for x in self.nondistributed_prediction_model.get_latent_factors_parameters().values()
                ]
                latent_factors_residual = np.sum(
                    [((x - y) ** 2).sum() for x, y in zip(new_p_latent_factors_copy, self.p_latent_factors_copy)]
                )
                self.p_latent_factors_copy = new_p_latent_factors_copy
                averages["avg_latent_factors_residual"] += latent_factors_residual

        # TODO(guy.rosman): add input encoder multitask costs logging
        averages["avg_g_cost"] += g_cost.detach().cpu().item()

    def optimize_discriminator(self, d_cost, averages):
        """Perform the discriminator optimization, gradient accumulation and residual computation.

        Parameters
        ----------
        d_cost torch.Tensor scalar
            Discriminator cost.
        averages: dict, including:
            avg_d_input_encoders_residual: dict
                Residuals of the input encoders.
            avg_d_agent_input_encoders_residual: dict
                Residuals of the agent input encoders.
            avg_d_residual: scalar
                Accumulator for d residuals, averaged over the epoch.
            avg_d_cost: scalar
                Accumulator of d costs, averaged over the epoch.
        """
        if not self.trainer_param["disable_discriminator_update"]:
            if self.residual_computation:
                self.p_d_copy = [
                    x.cpu().detach().numpy()
                    for x in self.nondistributed_prediction_model.get_discriminator_parameters().values()
                ]

                for (
                    key,
                    module,
                ) in self.nondistributed_prediction_model.get_discriminator_agent_input_encoders().items():
                    param_vec = np.concatenate(
                        [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                    )
                    self.p_d_agent_inputs_encoders_copy[key] = param_vec

                for key, module in self.nondistributed_prediction_model.get_discriminator_input_encoders().items():
                    param_vec = np.concatenate(
                        [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                    )
                    self.p_d_inputs_encoders_copy[key] = param_vec
            # TODO(guy.rosman): GENERALIZATION - encapsulate optimization into a method
            if not self.trainer_param["disable_optimization"]:
                d_cost.backward()
                if self.trainer_param["clip_gradients"]:
                    threshold = self.trainer_param["clip_gradients_threshold"]
                    torch.nn.utils.clip_grad_norm_(
                        list(self.nondistributed_prediction_model.get_discriminator_parameters().values()), threshold
                    )
            if torch.isinf(d_cost).sum() or torch.isnan(d_cost).sum() > 0:
                import IPython

                IPython.embed(header="inf discriminator cost")
            self.d_optimizer_steps += 1
            if (
                self.d_optimizer_steps % self.trainer_param["optimizer_step_batch"] == 0
                and not self.trainer_param["disable_optimization"]
            ):
                self.d_optimizer.step()
                self.d_optimizer.zero_grad(set_to_none=True)
            if self.residual_computation:
                new_p_d_copy = [
                    x.cpu().detach().numpy()
                    for x in self.nondistributed_prediction_model.get_discriminator_parameters().values()
                ]
                residual = np.sum([((x - y) ** 2).sum() for x, y in zip(new_p_d_copy, self.p_d_copy)])
                self.p_d_copy = new_p_d_copy
                averages["avg_d_residual"] += residual

                input_residuals = {}
                agent_input_residuals = {}
                for (
                    key,
                    module,
                ) in self.nondistributed_prediction_model.get_discriminator_agent_input_encoders().items():
                    param_vec = np.concatenate(
                        [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                    )
                    agent_input_residuals[key] = ((self.p_d_agent_inputs_encoders_copy[key] - param_vec) ** 2).sum()
                    self.p_d_agent_inputs_encoders_copy[key] = param_vec
                    if key not in averages["avg_d_agent_input_encoders_residual"]:
                        averages["avg_d_agent_input_encoders_residual"][key] = 0
                    averages["avg_d_agent_input_encoders_residual"][key] += agent_input_residuals[key]

                for key, module in self.nondistributed_prediction_model.get_discriminator_input_encoders().items():
                    param_vec = np.concatenate(
                        [p.view(-1).unsqueeze(0).detach().cpu().numpy() for p in module.parameters()], 1
                    )
                    input_residuals[key] = ((self.p_d_inputs_encoders_copy[key] - param_vec) ** 2).sum()
                    self.p_d_inputs_encoders_copy[key] = param_vec
                    if key not in averages["avg_d_input_encoders_residual"]:
                        averages["avg_d_input_encoders_residual"][key] = 0
                    averages["avg_d_input_encoders_residual"][key] += input_residuals[key]

        averages["avg_d_cost"] += d_cost.detach().cpu().item()

    def get_statistics_keys(self):
        # Define what stats to use.
        statistic_keys = [
            "l2_error",
            "robust_error",
            "MoN_error",
            "l2_cost",
            "data_cost",
            "ade_error",
            "fde_error",
            "MoN_ade_error",
            "MoN_fde_error",
            "MoN_fde_marginal",
            "MoN_ade_marginal",
            "decoder/decoder_x_mean",
            "decoder/decoder_y_mean",
            "decoder/decoder_x_std",
            "decoder/decoder_y_std",
        ]
        if self.use_semantics:
            statistic_keys.append("semantic_cost")
            for skey in self.prediction_model.get_semantic_keys():
                statistic_keys.append("semantic_costs/" + skey)
        if self.trainer_param["trajectory_regularization_cost"] > 0.0:
            statistic_keys.append("acceleration_cost")
        if self.use_discriminator:
            statistic_keys.append("discrimination_cost")
        if not self.trainer_param["l2_error_only"]:
            statistic_keys.append("MoN_cost")
            if self.trainer_param["use_marginal_error"]:
                statistic_keys.append("MoN_cost_marginal")
        for fde_timepoint in self.trainer_param["err_horizons_timepoints"]:
            statistic_keys.append("fde_error/{}_sec".format(fde_timepoint))
            statistic_keys.append("fde_error_full/{}_sec".format(fde_timepoint))
            statistic_keys.append("ade_error/{}_sec".format(fde_timepoint))
            statistic_keys.append("MoN_fde_error/{}_sec".format(fde_timepoint))
        if self.trainer_param["report_agent_type_metrics"]:
            assert (
                "agent_types" in self.trainer_param
            ), "Agent types need to be provided if report_agent_type_metrics is set."
            for agent_type_id in self.trainer_param["agent_types"]:
                agent_type_name = AGENT_TYPE_NAME_MAP[agent_type_id]
                for fde_point in self.trainer_param["err_horizons_timepoints"]:
                    statistic_keys.append("MoN_fde_marginal/{}/{}_sec".format(agent_type_name, fde_point))
                    statistic_keys.append("MoN_ade_marginal/{}/{}_sec".format(agent_type_name, fde_point))
        if self.trainer_param["report_sample_metrics"]:
            for i in range(self.trainer_param["MoN_number_samples"]):
                statistic_keys.append("MoN_ade_sample/sample_{}".format(i))
        if self.trainer_param["map_input_type"] == "point" and self.trainer_param["map_encoder_type"] == "gnn":
            statistic_keys.append("map_point_encoder/position")
            statistic_keys.append("map_point_encoder/tangent")
            statistic_keys.append("map_point_encoder/poly")
            statistic_keys.append("map_point_encoder/normal")
            statistic_keys.append("map_point_encoder/type")

        # Statistic key callbacks for non-standard keys.
        for cb in self.additional_trainer_callbacks:
            cb.update_statistic_keys(statistic_keys)

        return statistic_keys

    def get_balanced_indices(self, dataset, dataset_key, epoch_size, batch_size):
        # Balance classes - create the data item indices for each class
        if self.latent_factor_callback is not None:
            balanced_indices = self.latent_factor_callback.get_balanced_indices(
                dataset,
                dataset_key,
                epoch_size,
                batch_size,
                None,
            )
        else:
            indices_range = np.array(range(len(dataset)))
            rebalance_class_indices = {"all": indices_range}

            # Balance classes - create the indices with each class,
            # concatenate to create an index set to sample from.
            corrected_epoch_size = int(np.ceil(max(epoch_size, batch_size * 2) // len(rebalance_class_indices)))
            if not self.trainer_param["full_dataset_epochs"]:
                balanced_indices = []
                for cls_id, class_idxs in rebalance_class_indices.items():
                    if self.trainer_param["dataloader_shuffle"]:
                        balanced_indices.append(
                            np.random.choice(
                                a=class_idxs,
                                size=corrected_epoch_size,
                                replace=True,
                            )
                        )
                    else:
                        # tile indices to corrected epoch size
                        class_idxs = class_idxs * (corrected_epoch_size // len(class_idxs) + 1)
                        balanced_indices.append(class_idxs[:corrected_epoch_size])
                balanced_indices = np.concatenate(balanced_indices)
            else:
                self.message_logger.log_message("Skip rebalancing as we are just going over the whole dataset.")
                balanced_indices = list(range(len(dataset)))

        return balanced_indices

    def get_epochs(self):
        if self.inference_mode:
            return [self.start_epoch]
        else:
            return range(self.start_epoch, self.total_epoch)

    @typechecked
    def _has_trained_for_this_many_hours(self, hours: Union[int, float]) -> bool:
        """Returns True if the number of hours has passed"""
        now = datetime.datetime.now()
        delta = now - self.training_start_time
        passed_hours = delta.total_seconds() / 3600
        return passed_hours > hours

    @typechecked
    def _max_training_time_reached(self) -> bool:
        """Returns True if max training time has elapsed"""
        max_hours = self.trainer_param["max_training_time"]
        if max_hours > 0.0 and self._has_trained_for_this_many_hours(max_hours):
            self.message_logger.log_message(f"Stop training, hit max training time, {max_hours}")
            return True
        return False

    @typechecked
    def _save_periodic_models(
        self, epoch: int, global_batch_cnt: int, input_tensor: torch.Tensor, is_valid: torch.Tensor
    ) -> None:
        """Possibly save both the latest model every N epochs and a snapshot model every M hours"""
        if not self.trainer_param["disable_model_saving"]:
            offset_epoch = epoch + self.trainer_param["save_iteration_interval_offset"]
            if offset_epoch % self.trainer_param["save_iteration_interval"] == 0 or epoch == self.total_epoch - 1:
                self.save_models(epoch, global_batch_cnt, input_tensor, is_valid)

            if self.trainer_param["periodic_save_hours"]:
                next_save_hour = self.last_saved_model_hour + self.trainer_param["periodic_save_hours"]
                if self._has_trained_for_this_many_hours(next_save_hour):
                    self.save_models(
                        epoch, global_batch_cnt, input_tensor, is_valid, folder_postfix=f"_hour_{next_save_hour}"
                    )
                    self.last_saved_model_hour = next_save_hour

    def train(self, additional_logging_handlers=None):
        """Runs the main training loop for a model.

        Parameters
        ----------
        additional_logging_handlers : list
            Instances of LoggingHandler that are called at beginning / end of
            each epoch and after each iteration.
        """
        if additional_logging_handlers is None:
            additional_logging_handlers = []

        # Get epoch tqdm iterator.
        tqdm_iter = tqdm.tqdm(self.get_epochs(), desc="Epoch", disable=self.trainer_param["disable_tqdm"])
        global_batch_cnt = self.start_global_batch_cnt

        statistic_keys = self.get_statistics_keys()

        # TODO(guy.rosman): GENERALIZATION - encapsulate GAN functionality
        if self.g_optimizer is not None:
            self.g_optimizer.zero_grad(set_to_none=True)
        if self.use_discriminator and self.d_optimizer:
            self.d_optimizer.zero_grad(set_to_none=True)

        # TODO(guy.rosman): GENERALIZATION - encapsulate logging handlers, unify the 2 types of logging handlers.
        for logging_handler in additional_logging_handlers:
            logging_handler.initialize_training(self.logger)
        # Create dictionary of LogWorstCases handlers for ease of access during training.
        worst_case_loggers = {
            logger.logger_key: logger for logger in additional_logging_handlers if isinstance(logger, LogWorstCases)
        }
        miss_rate_stats = {}

        # Used to decide when to save a model
        save_validation_criterion = np.inf

        if self.debugger is not None:
            self.debugger.record_value("model_state", self.nondistributed_prediction_model.state_dict())
        self.training_start_time = datetime.datetime.now()
        for epoch in tqdm_iter:
            # TODO(guy.rosman): GENERALIZATION - encapsulate residual computation
            # TODO(guy.rosman): GENERALIZATION - encapsulate statistics
            if self.trainer_param["disable_tqdm"]:
                print("epoch: ", epoch)

            if self._max_training_time_reached():
                break

            self.trainer_param["writer_global_step"] = global_batch_cnt

            offset_epoch = epoch + self.trainer_param["vis_interval_offset"]
            skip_visualization = (
                self.trainer_param["disable_visualizations"]
                or not offset_epoch % self.trainer_param["vis_interval"] == 0
            )

            offset_epoch = epoch + self.trainer_param["val_interval_offset"]
            skip_validation = (
                self.trainer_param["disable_validation"] or not offset_epoch % self.trainer_param["val_interval"] == 0
            )

            regression_statistics = collections.deque(maxlen=ERROR_FILTER_TAPS)

            for cb in self.additional_trainer_callbacks:
                cb.epoch_update(epoch, self.trainer_param)

            for dataloader_type, dataloader_info in self.dataloaders_info.items():
                if (skip_visualization and dataloader_type == "vis") or (
                    skip_validation and dataloader_type == "validation"
                ):
                    continue

                statistics = {key: [] for key in statistic_keys}
                statistics_sums = {key: 0 for key in statistic_keys}

                torch.set_grad_enabled(dataloader_type == "train" and not self.trainer_param["disable_optimization"])

                # Obtain dataloader.
                dataset_key = dataloader_info["dataset_key"]
                dataset = dataloader_info["dataset"]
                epoch_size = dataloader_info["epoch_size"]
                batch_size = dataloader_info["batch_size"]
                num_workers = dataloader_info["num_workers"]

                self.profiler.step("before_encoder")

                if len(dataset) < batch_size * 2 and not self.inference_mode:
                    warning_str = "Warning: len(dataset)<batch_size*2, {},{}.".format(len(dataset), batch_size)
                    self.message_logger.log_message(warning_str)

                sampler = None
                if not self.inference_mode and dataloader_type in ("train", "validation"):
                    balanced_indices = self.get_balanced_indices(dataset, dataset_key, epoch_size, batch_size)
                    if self.trainer_param["dataloader_shuffle"]:
                        sampler = SubsetRandomSampler(indices=balanced_indices)
                    else:
                        dataset = Subset(dataset, balanced_indices)

                # Set it to pytorch's default 2 when num_workers == 0, otherwise it will throw.
                if num_workers == 0:
                    persistent_workers = False
                    pre_fetch = 2
                else:
                    persistent_workers = True
                    pre_fetch = self._dataloader_pre_fetch_factor

                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    sampler=sampler,
                    shuffle=False,
                    pin_memory=self._dataloader_pin_memory,
                    prefetch_factor=pre_fetch,
                    persistent_workers=persistent_workers,
                )

                if dataloader_type == "train":
                    self.nondistributed_prediction_model.child_net_train()
                    self.nondistributed_prediction_model.train()
                else:
                    self.nondistributed_prediction_model.child_net_eval()
                    self.nondistributed_prediction_model.eval()

                # Create tqdm iterator for dataloader.
                p_i_tqdm = tqdm.tqdm(dataloader, desc=dataloader_type, disable=self.trainer_param["disable_tqdm"])

                g_costs = []
                if self.use_discriminator:
                    d_costs = []
                t1 = time.perf_counter()
                visual_size = self.trainer_param["num_visualization_worst_cases"]
                batch_stats = {}
                time_intervals = {}
                epoch_semantic_labels = None
                epoch_predicted_semantics = None
                if self.device.type != "cpu":
                    torch.cuda.empty_cache()

                for logging_handler in additional_logging_handlers:
                    if isinstance(logging_handler, LogWorstCases):
                        logging_handler.epoch_start(dataloader_type, visual_size)
                    else:
                        logging_handler.epoch_start(dataloader_type)

                for key in ["dt0", "dt1", "dt2", "dt3"]:
                    time_intervals[key] = 0

                self.profiler.step("before_dataloading")

                g_batchs_cnt = 0  # Used for normalizing costs.
                data_samples_cnt = 0  # Used for normalizing summary statistics.
                avg_g_cost = 0.0
                avg_g_residual = 0.0
                avg_g_input_encoders_residual = {}
                avg_g_agent_input_encoders_residual = {}
                if self.use_discriminator:
                    avg_d_residual = 0.0
                    avg_d_cost = 0.0
                    avg_d_input_encoders_residual = {}
                    avg_d_agent_input_encoders_residual = {}
                    d_batchs_cnt = 0  # Used for normalizing costs.
                if self.use_latent_factors:
                    avg_latent_factors_residual = 0.0
                for batch_itm_idx, batch_itm in enumerate(p_i_tqdm):
                    max_batch_item_visualized = self.trainer_param["num_map_visualization_batches"]
                    self.trainer_param["skip_visualization"] = (
                        skip_visualization or batch_itm_idx > max_batch_item_visualized
                    )
                    self.profiler.step("after_dataloading")

                    iter_t0 = time.perf_counter()
                    if self.visualize_items:
                        visualize_itm(batch_itm, batch_idx=0)

                    for key in batch_itm:
                        if not isinstance(batch_itm[key], list):
                            batch_itm[key] = batch_itm[key].to(self.device)

                    (
                        batch_size,
                        num_agents,
                        _,
                        num_past_timepoints,
                        num_future_timepoints,
                    ) = get_prediction_item_sizes(batch_itm)
                    # Compute cost
                    g_cost = 0
                    if self.use_discriminator:
                        d_cost = 0
                    # Global position scale.
                    scale = self.trainer_param["predictor_normalization_scale"]
                    # TODO(guy.rosman): GENERALIZATION - encapsulate all normalizations
                    if self.trainer_param["augment_trajectories"] and dataloader_type == "train":
                        # TODO(guy.rosman): GENERALIZATION - make augmentation modifiable / a functor
                        self.augment_data(
                            self.trainer_param, batch_itm, list(self.prediction_model.agent_input_encoders.keys())
                        )

                    # Preprocess batch item, by scaling down positions.
                    (
                        batch_positions_tensor,
                        batch_is_valid,
                        batch_is_rel,
                        batch_agent_type,
                        batch_timestamps_tensor,
                        prediction_timestamp,
                        semantic_labels,
                    ) = preprocess_batch_item(batch_itm, self.trainer_param)

                    # Load global additional inputs: images, maps, test stub.
                    additional_inputs, agent_additional_inputs = get_inputs_from_batch_item(
                        batch_itm, num_past_timepoints
                    )

                    for cb in self.additional_trainer_callbacks:
                        cb.update_additional_inputs(additional_inputs, batch_itm, num_past_timepoints)

                    self.nondistributed_prediction_model.set_num_agents(
                        num_agents, num_past_timepoints, num_future_timepoints
                    )
                    is_valid = batch_is_valid[:, :, :num_past_timepoints].detach()
                    is_future_valid = batch_is_valid[:, :, -num_future_timepoints:].detach()

                    input_tensor, expected_trajectories_scene, offset_x, offset_y = get_normalized_trajectories(
                        self.trainer_param,
                        batch_positions_tensor,
                        is_valid,
                        num_past_timepoints,
                        num_future_timepoints,
                    )

                    # Load map tensor, depending on different type.
                    if self.trainer_param["use_global_map"]:
                        map_inputs = additional_inputs
                    else:
                        map_inputs = agent_additional_inputs
                    (batch_map_positions_tensor, batch_map_validity_tensor, batch_map_others_tensor,) = scale_maps(
                        self.trainer_param["map_input_type"],
                        scale,
                        batch_itm,
                        map_inputs,
                        offset_x,
                        offset_y,
                    )

                    for cb in self.additional_trainer_callbacks:
                        cb.update_agent_additional_inputs(
                            agent_additional_inputs,
                            batch_positions_tensor,
                            batch_is_valid,
                            offset_x,
                            offset_y,
                            num_past_timepoints,
                        )

                    iter_t1 = time.perf_counter()
                    time_intervals["dt0"] += iter_t1 - iter_t0

                    # Obtain number of samples
                    if dataloader_type == "validation" and self.trainer_param["validate_multiple_samples"]:
                        num_samples = self.trainer_param["validate_sample_size"]
                    else:
                        num_samples = self.trainer_param["MoN_number_samples"]

                    expected = {"trajectories": expected_trajectories_scene[..., :2]}
                    additional_inputs["future_positions"] = expected_trajectories_scene

                    # Obtain expected trajectory and validity for both past and future.
                    expected_trajectories_full_scene = torch.cat(
                        [input_tensor[..., :2], expected_trajectories_scene[..., :2]], 2
                    )
                    is_valid_full = torch.cat([is_valid, is_future_valid], -1)

                    # Obtain predictions and other information from model.
                    (
                        predicted_trajectories_scene,
                        decoding,
                        stats_list,
                        traj_additional_costs,
                    ) = self.nondistributed_prediction_model.generate_trajectory(
                        input_trajectory=input_tensor,
                        additional_inputs=additional_inputs,
                        agent_additional_inputs=agent_additional_inputs,
                        relevant_agents=batch_is_rel,
                        agent_type=batch_agent_type,
                        is_valid=is_valid.bool(),
                        timestamps=batch_timestamps_tensor[:, :num_past_timepoints],
                        prediction_timestamp=prediction_timestamp,
                        num_samples=num_samples,
                        additional_params={"skip_visualization": skip_visualization},
                    )
                    self.profiler.step("after_generate_trajectory")

                    # Get the stats of the first sample, assuming the stats is consistent across all samples.
                    stats = stats_list[0]

                    # TODO(guy.rosman) Introduce a global un-normalizing of coordinates, to check for deployment.
                    for key in statistic_keys:
                        if key in statistics and key in stats:
                            if isinstance(stats[key], list):
                                addition = np.array(stats[key])
                            else:
                                addition = stats[key].detach().cpu().numpy()
                            statistics[key].append(addition)

                    predicted = {"trajectories": predicted_trajectories_scene}
                    for cb in self.additional_trainer_callbacks:
                        cb.update_decoding(predicted, stats_list)

                    if self.inference_mode:
                        # Change the offset back.
                        traj_shape = list(predicted_trajectories_scene.size())
                        predicted_trajectories_scene[:, :, :, 0] += (
                            offset_x.unsqueeze(2).unsqueeze(3).repeat(1, 1, traj_shape[2], traj_shape[4])
                        )
                        predicted_trajectories_scene[:, :, :, 1] += (
                            offset_y.unsqueeze(2).unsqueeze(3).repeat(1, 1, traj_shape[2], traj_shape[4])
                        )

                        # Change the scale back to original.
                        predicted_trajectories_scene /= scale
                        output_protobuf = convert_predictor_output_to_protobuf(
                            predicted_trajectories_scene,
                            batch_itm["dot_keys"],
                            decoding,
                            stats,
                            timestep=self.trainer_param["future_timestep_size"],
                            t0=0.0,
                        )
                        return output_protobuf

                    iter_t2 = time.perf_counter()
                    time_intervals["dt1"] += iter_t2 - iter_t1
                    if "cumulative_durations" in stats:
                        for key in stats["cumulative_durations"]:
                            stats_mean_key = dataloader_type + "_duration_mean_" + key
                            if stats_mean_key not in batch_stats:
                                batch_stats[stats_mean_key] = []
                            batch_stats[stats_mean_key].append(
                                stats["cumulative_durations"][key]["mean"].detach().cpu().numpy()
                            )

                    for cb in self.additional_trainer_callbacks:
                        cb.update_expected_results(expected, batch_itm, num_future_timepoints)

                    if self.trainer_param["disable_position_input"]:
                        expected["trajectories"] *= 0

                    if self.use_semantics and self.latent_factor_callback:
                        self.semantic_label_weights = self.latent_factor_callback.get_semantic_label_weights(
                            semantic_labels, dataloader_type, self.device
                        ).cpu()
                    else:
                        self.semantic_label_weights = None

                    # NOTE(igor.gilitschenski): This is intended for looping over the data and experimenting with it.
                    if self.trainer_param["disable_gan_training"]:
                        import IPython

                        IPython.embed(header="Look at different predictor models")
                        # TODO: visualize batch_itm['images'], others.
                        continue

                    self.profiler.step("before_compute_cost")
                    if dataloader_type in ("train", "validation"):
                        is_generator_update = batch_itm_idx % 2 == 0 or not self.use_discriminator
                        if self.trainer_param["save_cases_for_table"]:
                            is_generator_update = True
                        if_log_worst_case = not skip_visualization
                        if is_generator_update:
                            # TODO(guy.rosman): separate into a function of computer_generator_cost +
                            # TODO  all the statistics.

                            # Compute generator costs.
                            # TODO(cyrushx): Skip gt trajectory after computing the loss.
                            (g_cost1, g_stats) = self.nondistributed_prediction_model.compute_generator_cost(
                                past_trajectory=input_tensor[..., :2],
                                past_additional_inputs=additional_inputs,
                                agent_additional_inputs=agent_additional_inputs,
                                predicted=predicted,
                                expected=expected,
                                agent_type=batch_agent_type,
                                is_valid=is_valid,
                                is_future_valid=is_future_valid,
                                timestamps=batch_timestamps_tensor,
                                prediction_timestamp=prediction_timestamp,
                                semantic_labels=semantic_labels,
                                future_encoding=decoding,
                                relevant_agents=batch_is_rel,
                                label_weights=self.semantic_label_weights,
                                param=self.trainer_param,
                                stats=stats_list,
                            )

                            predicted_semantics = None if not self.use_semantics else g_stats["predicted_semantics"]
                            visual_size = min(visual_size, batch_size)

                            # TODO(guy.rosman): consider unifying with runner statistics loggers calls.
                            data_dict = {
                                "batch_itm": batch_itm,
                                "batch_cost": g_cost1,
                                "predicted_trajectories": predicted_trajectories_scene,
                                "offset_x": offset_x,
                                "offset_y": offset_y,
                                "is_future_valid": is_future_valid,
                                "semantic_labels": semantic_labels,
                                "predicted_semantics": predicted_semantics,
                                "map_coordinates": batch_map_positions_tensor,
                                "map_validity": batch_map_validity_tensor,
                                "map_others": batch_map_others_tensor,
                                "agent_transforms": stats_list[0]["agent_transforms"],
                            }
                            stats_dict = {"num_future_timepoints": num_future_timepoints}

                            if if_log_worst_case and "g_cost" in worst_case_loggers:
                                worst_case_loggers["g_cost"].iteration_update(
                                    data_dict=data_dict, stats_dict=stats_dict
                                )

                                if "distance_x0" in stats:
                                    data_dict["batch_cost"] = stats["distance_x0"]

                                    worst_case_loggers["distance_x0"].iteration_update(
                                        data_dict=data_dict, stats_dict=stats_dict
                                    )

                            # Update worse case for each agent index, according to valid agent MoN fde.
                            # TODO(cyrushx): Make this for each agent type instead.
                            for agent_idx in range(self.trainer_param["max_agents"]):
                                logger_key = f"fde_agent_{agent_idx}"
                                # rel_agent_mon_fde = g_stats["agent_mon_fde"][:, agent_idx]
                                rel_agent_mon_fde = (g_stats["agent_mon_fde"] * batch_is_rel)[:, agent_idx]
                                # Set nan losses to 0, which should be ignored in worst cases.
                                rel_agent_mon_fde[torch.isnan(rel_agent_mon_fde)] = 0

                                data_dict["batch_cost"] = rel_agent_mon_fde
                                if if_log_worst_case:
                                    worst_case_loggers[logger_key].iteration_update(
                                        data_dict=data_dict, stats_dict=stats_dict
                                    )

                            if self.use_semantics:
                                for skey in self.prediction_model.get_semantic_keys():
                                    key = "semantic_costs/" + skey
                                    if key in g_stats:
                                        # TODO(ThomasB): Can this be removed into training_utils? g_stats doesn't exist
                                        # TODO  until the training loop is run.
                                        if not if_log_worst_case:
                                            continue

                                        if key not in worst_case_loggers:
                                            worst_case_loggers[key] = LogWorstCases(
                                                key, self.trainer_param, self.worst_cases_folder
                                            )
                                            worst_case_loggers[key].epoch_start(
                                                dataloader_type,
                                                visual_size,
                                                num_agents,
                                            )

                                        try:
                                            data_dict["batch_cost"] = g_stats[key]
                                            worst_case_loggers[key].iteration_update(
                                                data_dict=data_dict, stats_dict=stats_dict
                                            )
                                        except:
                                            self.message_logger.log_message(
                                                "Failed to perform iteration_update on key {}, cost: {}".format(
                                                    key, g_stats[key]
                                                )
                                            )

                            if torch.isnan(g_cost1.mean()):
                                import IPython

                                IPython.embed(header="nan g_cost1")
                            g_cost = g_cost1.mean()
                            coeff_input_processors_costs = self.trainer_param["coeff_input_processors_costs"]
                            for key in traj_additional_costs:
                                if not key == "point_count":
                                    g_cost_additional = (
                                        traj_additional_costs[key]
                                        / (
                                            traj_additional_costs["point_count"] + 1e-20
                                        )  # Add small epsilon when counter is 0.
                                        * coeff_input_processors_costs
                                    )
                                    g_cost += g_cost_additional
                                    g_stat_key = "map_point_encoder/" + key
                                    g_stats[g_stat_key] = g_cost_additional
                            if self.debugger is not None:
                                self.debugger.record_value(f"{dataloader_type}_g_cost1", g_cost1)
                                self.debugger.record_value(f"{dataloader_type}_g_cost", g_cost)

                            g_costs.append(g_cost1.detach().cpu().numpy())
                            for key in statistic_keys:
                                if key in statistics and key in g_stats:
                                    if isinstance(g_stats[key], list):
                                        addition = np.array(g_stats[key])
                                    else:
                                        addition = g_stats[key].detach().cpu().numpy()
                                    statistics[key].append(addition)
                            if self.use_semantics:
                                for skey in self.prediction_model.get_semantic_keys():
                                    key = "semantic_costs/" + skey
                                    # TODO(guy.rosman): Clean up statistics_sums.
                                    if key in g_stats:
                                        curr_sum = statistics_sums.get(key, 0)
                                        if isinstance(g_stats[key], list):
                                            statistics_sums[key] = curr_sum + np.sum(g_stats[key])
                                        else:
                                            statistics_sums[key] = curr_sum + g_stats[key].sum().detach().cpu().item()

                                for key in self.nondistributed_prediction_model.get_semantic_keys():
                                    if key in g_stats and len(g_stats[key]) > 0:
                                        if key not in batch_stats:
                                            batch_stats[key] = []
                                        batch_stats[key].extend(g_stats[key].detach().cpu().numpy().tolist())

                            data_samples_cnt += len(g_stats["data_cost"])

                            # TODO(guy.rosman): move to function.
                            # Choose the first trajectory if multiple trajectories are predicted.
                            if dataloader_type == "validation" and self.trainer_param["validate_multiple_samples"]:
                                worst_predicted_traj_sample_size = worst_case_loggers[
                                    "g_cost"
                                ].worst_predicted_traj.shape[-1]
                                predicted_trajectories_scene = predicted_trajectories_scene[
                                    ..., :worst_predicted_traj_sample_size
                                ]

                            if (semantic_labels is not None) and (semantic_labels**2).sum() > 0:
                                if epoch_semantic_labels is None:
                                    epoch_semantic_labels = semantic_labels
                                else:
                                    epoch_semantic_labels = torch.cat((epoch_semantic_labels, semantic_labels), dim=0)
                                if epoch_predicted_semantics is None:
                                    epoch_predicted_semantics = g_stats["predicted_semantics"]
                                else:
                                    epoch_predicted_semantics = torch.cat(
                                        (epoch_predicted_semantics, g_stats["predicted_semantics"]), dim=0
                                    )
                            g_batchs_cnt += 1
                        else:
                            # Compute discriminator costs.
                            if len(additional_inputs) == 0:
                                import IPython

                                IPython.embed(header="len(additional_inputs) == 0")
                            # pylint: disable=W0641
                            d_cost1, _ = self.nondistributed_prediction_model.compute_discriminator_cost(
                                input_tensor,
                                additional_inputs,
                                agent_additional_inputs,
                                predicted_trajectories_scene,
                                expected_trajectories_scene,
                                batch_agent_type,
                                is_valid,
                                is_future_valid,
                                is_fake=True,
                                timestamps=batch_timestamps_tensor,
                                prediction_timestamp=prediction_timestamp,
                                relevant_agents=batch_is_rel,
                                param=self.trainer_param,
                            )
                            if self.debugger is not None:
                                self.debugger.record_value(f"{dataloader_type}_d1_cost", d_cost1)
                            d_cost = d_cost1.mean() / 2.0
                            if torch.isnan(d_cost1.mean()) or torch.isinf(d_cost1.mean()):
                                import IPython

                                IPython.embed(header="nan d_cost1")
                            d_cost2, _ = self.nondistributed_prediction_model.compute_discriminator_cost(
                                input_tensor,
                                additional_inputs,
                                agent_additional_inputs,
                                expected_trajectories_scene.unsqueeze(4),
                                expected_trajectories_scene,
                                batch_agent_type,
                                is_valid,
                                is_future_valid,
                                is_fake=False,
                                timestamps=batch_timestamps_tensor,
                                prediction_timestamp=prediction_timestamp,
                                relevant_agents=batch_is_rel,
                                param=self.trainer_param,
                            )
                            if self.debugger is not None:
                                self.debugger.record_value(f"{dataloader_type}_d2_cost", d_cost2)
                            # pylint: enable=W0641
                            if torch.isnan(d_cost2.mean()) or torch.isinf(d_cost2.mean()):
                                import IPython

                                IPython.embed(header="nan/inf d_cost2")
                            # TODO(ThomasB): See if we can remove accumulation of values from cost computation, or
                            # TODO  add division by "MoN number samples" inside cost computation to keep things consistent in
                            # TODO  the trainer.
                            d_cost += d_cost2.mean() / 2.0 * self.trainer_param["MoN_number_samples"]
                            if self.debugger is not None:
                                self.debugger.record_value(f"{dataloader_type}_d_cost", d_cost)
                            d_costs.append(d_cost2.detach().cpu().numpy())
                            d_batchs_cnt += 1

                    elif dataloader_type == "vis":
                        # Do visualization.
                        max_visualized_batches = np.ceil(
                            min(dataloader_info["epoch_size"], self.trainer_param["num_visualization_images"])
                            // batch_size
                        )
                        if batch_itm_idx < max_visualized_batches:
                            (g_cost1, g_stats) = self.nondistributed_prediction_model.compute_generator_cost(
                                input_tensor[..., :2],
                                additional_inputs,
                                agent_additional_inputs,
                                predicted,
                                expected,
                                batch_agent_type,
                                is_valid,
                                is_future_valid,
                                batch_timestamps_tensor,
                                prediction_timestamp,
                                semantic_labels,
                                decoding,
                                batch_is_rel,
                                self.semantic_label_weights,
                                self.trainer_param,
                                stats_list,
                            )

                            visualize_sample_process(
                                batch_itm_idx=batch_itm_idx,
                                batch_size=batch_size,
                                batch_itm=batch_itm,
                                g_stats=g_stats,
                                predicted=predicted,
                                predicted_trajectories_scene=predicted_trajectories_scene,
                                is_future_valid=is_future_valid,
                                scale=scale,
                                offset_x=offset_x,
                                offset_y=offset_y,
                                cost=g_cost1,
                                param=self.trainer_param,
                                summary_writer=self.logger,
                                iter=global_batch_cnt,
                                tag_prefix="vis",
                                num_past_timepoints=num_past_timepoints,
                                label_weights=self.semantic_label_weights,
                                semantic_labels=semantic_labels,
                                predicted_semantics=predicted_semantics,
                                map_coordinates=batch_map_positions_tensor,
                                map_validity=batch_map_validity_tensor,
                                map_others=batch_map_others_tensor,
                                visualization_callbacks=self.additional_trainer_callbacks,
                            )
                        else:
                            break
                    self.profiler.step("after_compute_cost")

                    iter_t3 = time.perf_counter()
                    time_intervals["dt2"] += iter_t3 - iter_t2
                    if dataloader_type == "train":
                        # Do the optimization step
                        if batch_itm_idx % 2 == 0 or not self.use_discriminator:
                            # Optimize the generator
                            averages = {
                                "avg_g_input_encoders_residual": avg_g_input_encoders_residual,
                                "avg_g_agent_input_encoders_residual": avg_g_agent_input_encoders_residual,
                                "avg_g_residual": avg_g_residual,
                                "avg_g_cost": avg_g_cost,
                            }
                            if self.use_latent_factors:
                                averages["avg_latent_factors_residual"] = avg_latent_factors_residual
                            self.optimize_generator(g_cost, averages)
                            if not self.trainer_param["disable_tqdm"]:
                                p_i_tqdm.set_description(str(g_cost.detach().cpu().item()))
                            else:
                                now = time.time()
                                if not hasattr(self, "last_loop_time"):
                                    self.last_loop_time = now
                                memory_gb = torch.cuda.max_memory_allocated() * 1e-9
                                print(
                                    f"g_cost: {g_cost.item()} time: {now - self.last_loop_time} memory: {memory_gb} gb"
                                )
                                self.last_loop_time = now
                            avg_g_cost = averages["avg_g_cost"]
                            avg_g_residual = averages["avg_g_residual"]
                            avg_g_agent_input_encoders_residual = averages["avg_g_agent_input_encoders_residual"]
                            avg_g_input_encoders_residual = averages["avg_g_input_encoders_residual"]
                            if self.use_latent_factors:
                                avg_latent_factors_residual = averages["avg_latent_factors_residual"]
                        else:
                            averages = {
                                "avg_d_input_encoders_residual": avg_d_input_encoders_residual,
                                "avg_d_agent_input_encoders_residual": avg_d_agent_input_encoders_residual,
                                "avg_d_residual": avg_d_residual,
                                "avg_d_cost": avg_d_cost,
                            }
                            self.optimize_discriminator(d_cost, averages)
                            avg_d_cost = averages["avg_d_cost"]
                            avg_d_residual = averages["avg_d_residual"]
                            avg_d_agent_input_encoders_residual = averages["avg_d_agent_input_encoders_residual"]
                            avg_d_input_encoders_residual = averages["avg_d_input_encoders_residual"]
                    elif dataloader_type == "validation":
                        # Aggregate statistics
                        if batch_itm_idx % 2 == 0 or not self.use_discriminator:
                            avg_g_cost += g_cost.detach().cpu().item()
                        else:
                            avg_d_cost += d_cost.detach().cpu().item()

                        if self.trainer_param["save_example_outputs"]:
                            self.save_example_outputs(batch_itm, predicted_trajectories_scene, decoding, stats)
                    self.profiler.step("after_optimization")

                    if epoch == 0 and batch_itm_idx == 0 and dataloader_type == "train":
                        if self.logger:
                            self.logger.add_text(
                                "parameters of the model", self.nondistributed_prediction_model.count_parameters()
                            )
                    torch.cuda.empty_cache()
                    for logging_handler in additional_logging_handlers:
                        data_dictionary = {
                            "input_tensor": input_tensor,
                            "additional_inputs": additional_inputs,
                            "agent_additional_inputs": agent_additional_inputs,
                            "predicted_trajectories": predicted_trajectories_scene,
                            "expected_trajectories": expected_trajectories_scene,
                            "expected_trajectories_full": expected_trajectories_full_scene,
                            "is_valid": is_valid,
                            "is_future_valid": is_future_valid,
                            "is_valid_full": is_valid_full,
                            "batch_cost": data_dict["batch_cost"],
                            "batch_itm": batch_itm,
                            "batch_itm_index": batch_itm_idx,
                            "batch_positions_tensor": batch_positions_tensor,
                            "batch_agent_type": batch_agent_type,
                            "relevant_agents": batch_is_rel,
                            "offset_x": offset_x,
                            "offset_y": offset_y,
                            "param": self.trainer_param,
                            "stats_list": stats_list,
                            "global_batch_cnt": global_batch_cnt,
                            "agent_transforms": stats_list[0]["agent_transforms"],
                        }
                        stats_dict["g_stats"] = None

                        if dataloader_type == "vis" or is_generator_update:
                            stats_dict["g_stats"] = g_stats

                        if semantic_labels is not None and "predicted_semantics" in g_stats:
                            data_dictionary["semantic_labels"] = semantic_labels
                            data_dictionary["predicted_semantics"] = g_stats["predicted_semantics"]
                        else:
                            data_dictionary["semantic_labels"] = None
                            data_dictionary["predicted_semantics"] = None

                        if self.trainer_param["map_input_type"] == "point" and batch_map_positions_tensor is not None:
                            data_dictionary["map_coordinates"] = batch_map_positions_tensor / scale
                            data_dictionary["map_validity"] = batch_map_validity_tensor
                            data_dictionary["map_others"] = batch_map_others_tensor
                        else:
                            data_dictionary["map_coordinates"] = None
                            data_dictionary["map_validity"] = None
                            data_dictionary["map_others"] = None

                        data_dictionary["dataloader_type"] = dataloader_type
                        if not isinstance(logging_handler, LogWorstCases):
                            logging_handler.iteration_update(data_dictionary, stats_dict)
                        if isinstance(logging_handler, SaveErrorStatistics):
                            miss_rate_stats = logging_handler.vis_logger_stats

                    iter_t4 = time.perf_counter()
                    torch.cuda.empty_cache()
                    time_intervals["dt3"] += iter_t4 - iter_t3
                    self.profiler.step("after_batch_loop")

                if self.use_discriminator:
                    d_cost = None

                self.profiler.step("before_log_stats")

                t2 = time.perf_counter()
                # Log statistics
                if dataloader_type in ("train", "validation"):
                    avg_g_residual /= data_samples_cnt
                    avg_g_cost /= g_batchs_cnt
                    if self.use_discriminator:
                        avg_d_residual /= data_samples_cnt
                        avg_d_cost /= d_batchs_cnt
                    if self.use_latent_factors:
                        avg_latent_factors_residual /= data_samples_cnt
                    time_per_sample = (t2 - t1) / data_samples_cnt

                    for cb in self.additional_trainer_callbacks:
                        cb.update_statistics(
                            statistic_keys,
                            statistics,
                            self.logger,
                            dataloader_type,
                            global_batch_cnt,
                        )

                    skip_keys = ["task_specific_prediction"]
                    for key in statistic_keys:
                        if key in skip_keys:
                            continue

                        if len(statistics[key]) > 0:
                            if len(statistics[key][0].shape) > 0:
                                aggregate = np.concatenate(statistics[key])
                            else:
                                aggregate = statistics[key]
                            if len(aggregate) > 0:
                                with np.errstate(all="raise"):
                                    try:
                                        scalar = np.mean(aggregate)
                                    except:
                                        import IPython

                                        IPython.embed(header="check")
                                    if self.logger:
                                        self.logger.add_scalar(
                                            dataloader_type + "_" + str(key),
                                            scalar,
                                            global_step=global_batch_cnt,
                                        )

                        # TODO(cyrushx): Fix the last condition.
                        if (
                            len(statistics[key]) > 0
                            and len(np.hstack(statistics[key])) > 1
                            and self.logger
                            and self.visualize_items
                        ):
                            try:
                                img = visualize_histogram(
                                    np.hstack(statistics[key]),
                                    key,
                                    image_format=self.trainer_param["visualization_image_format"],
                                )
                                self.logger.add_image(
                                    f"{dataloader_type}/hist_{key}", img.transpose(2, 0, 1), global_step=epoch
                                )
                            except:
                                self.message_logger.log_message("Failed to plot histogram.")

                    if self.logger:
                        self.logger.add_scalar(dataloader_type + "_g_cost", avg_g_cost, global_step=global_batch_cnt)
                    for key in avg_g_agent_input_encoders_residual:
                        avg_g_agent_input_encoders_residual[key] /= data_samples_cnt
                        if self.logger:
                            self.logger.add_scalar(
                                f"{dataloader_type}_g_residual_agent_{key}",
                                np.log(avg_g_agent_input_encoders_residual[key] + 1e-10),
                                global_step=global_batch_cnt,
                            )

                    for key in avg_g_input_encoders_residual:
                        avg_g_input_encoders_residual[key] /= data_samples_cnt
                        if self.logger:
                            self.logger.add_scalar(
                                f"{dataloader_type}_g_residual_{key}",
                                np.log(avg_g_input_encoders_residual[key] + 1e-10),
                                global_step=global_batch_cnt,
                            )

                    if self.use_discriminator:
                        if self.logger:
                            self.logger.add_scalar(
                                f"{dataloader_type}_d_cost", avg_d_cost, global_step=global_batch_cnt
                            )
                        for key in avg_d_agent_input_encoders_residual:
                            avg_d_agent_input_encoders_residual[key] /= data_samples_cnt
                            if self.logger:
                                self.logger.add_scalar(
                                    f"{dataloader_type}_d_residual_agent_{key}",
                                    np.log(avg_d_agent_input_encoders_residual[key] + 1e-10),
                                    global_step=global_batch_cnt,
                                )

                        for key in avg_d_input_encoders_residual:
                            avg_d_input_encoders_residual[key] /= data_samples_cnt
                            if self.logger:
                                self.logger.add_scalar(
                                    f"{dataloader_type}_d_residual_{key}",
                                    np.log(avg_d_input_encoders_residual[key] + 1e-10),
                                    global_step=global_batch_cnt,
                                )
                    if self.logger:
                        self.logger.add_scalar(
                            f"{dataloader_type}_ms_per_sample",
                            time_per_sample * 1000,
                            global_step=global_batch_cnt,
                        )
                    if self.logger:
                        for key in miss_rate_stats:
                            self.logger.add_scalar(
                                dataloader_type + "_" + key, miss_rate_stats[key], global_step=global_batch_cnt
                            )
                    interval_names = {}
                    interval_names["dt0"] = "data_processing"
                    interval_names["dt1"] = "trajectory_gen"
                    interval_names["dt2"] = "cost_computation"
                    interval_names["dt3"] = "backprop"
                    try:
                        for key in ["dt0", "dt1", "dt2", "dt3"]:
                            if self.logger:
                                self.logger.add_scalar(
                                    f"{dataloader_type}_ms_{interval_names[key]}_per_sample",
                                    time_intervals[key] / data_samples_cnt * 1000,
                                    global_step=global_batch_cnt,
                                )
                    except:
                        import IPython

                        IPython.embed()
                    if self.logger:
                        self.logger.add_scalar(
                            f"{dataloader_type}_g_residual",
                            np.log(avg_g_residual + 1e-10),
                            global_step=global_batch_cnt,
                        )
                        self.logger.add_histogram(
                            f"{dataloader_type}_hist_g_cost", np.hstack(g_costs), global_step=global_batch_cnt
                        )
                        if self.use_latent_factors:
                            self.logger.add_scalar(
                                f"{dataloader_type}_latent_factors_residual",
                                np.log(avg_latent_factors_residual + 1e-10),
                                global_step=global_batch_cnt,
                            )
                    if self.g_scheduler is not None and dataloader_type == "train":
                        if self.logger:
                            self.logger.add_scalar(
                                f"{dataloader_type}_learning_rate_g",
                                self.g_scheduler.get_lr()[0],
                                global_step=global_batch_cnt,
                            )
                    if self.use_discriminator:
                        if self.logger:
                            self.logger.add_scalar(
                                f"{dataloader_type}_d_residual",
                                np.log(avg_d_residual + 1e-10),
                                global_step=global_batch_cnt,
                            )
                            self.logger.add_histogram(
                                f"{dataloader_type}_hist_d_cost",
                                np.hstack(d_costs),
                                global_step=global_batch_cnt,
                            )

                        if self.d_scheduler is not None and dataloader_type == "train":
                            if self.logger:
                                self.logger.add_scalar(
                                    f"{dataloader_type}_learning_rate_d",
                                    self.d_scheduler.get_lr()[0],
                                    global_step=global_batch_cnt,
                                )

                    for stats_key, stats in batch_stats.items():
                        if isinstance(stats[0], np.ndarray):
                            if self.logger:
                                self.logger.add_histogram(
                                    stats_key,
                                    np.concatenate(stats, 0),
                                    global_step=global_batch_cnt,
                                )
                                self.logger.add_scalar(
                                    f"{stats_key}_mean",
                                    np.mean(stats[0]),
                                    global_step=global_batch_cnt,
                                )
                        else:
                            if self.logger:
                                self.logger.add_histogram(stats_key, np.array(stats), global_step=global_batch_cnt)

                    # Visualize the worst predictions
                    if not skip_visualization and self.logger:
                        for logger_key in worst_case_loggers:
                            num_logged_items = min(visual_size, worst_case_loggers[logger_key].worst_cost.shape[0])
                            tag_prefix = f"{dataloader_type}_worstcase_highest_{logger_key}"
                            visualize_sample_process(
                                batch_itm_idx=0,
                                batch_size=num_logged_items,
                                batch_itm=worst_case_loggers[logger_key].worst_itm,
                                g_stats=g_stats,
                                predicted=predicted,
                                predicted_trajectories_scene=worst_case_loggers[logger_key].worst_predicted_traj,
                                is_future_valid=worst_case_loggers[logger_key].worst_is_future_valid,
                                scale=scale,
                                offset_x=worst_case_loggers[logger_key].worst_mean_x,
                                offset_y=worst_case_loggers[logger_key].worst_mean_y,
                                cost=worst_case_loggers[logger_key].worst_cost,
                                param=self.trainer_param,
                                summary_writer=self.logger,
                                iter=global_batch_cnt,
                                tag_prefix=tag_prefix,
                                num_past_timepoints=num_past_timepoints,
                                label_weights=self.semantic_label_weights,
                                semantic_labels=worst_case_loggers[logger_key].worst_semantic_labels,
                                predicted_semantics=worst_case_loggers[logger_key].worst_predicted_semantics,
                                map_coordinates=worst_case_loggers[logger_key].worst_map_coordinates,
                                map_validity=worst_case_loggers[logger_key].worst_map_validity,
                                map_others=worst_case_loggers[logger_key].worst_map_others,
                            )

                    if (
                        self.use_semantics
                        and (epoch_semantic_labels is not None)
                        and (epoch_semantic_labels**2).sum() > 0
                    ):
                        _ = visualize_label_accuracy(
                            epoch_semantic_labels,
                            epoch_predicted_semantics,
                            self.semantic_label_weights,
                            self.trainer_param["latent_factors_file"],
                            self.logger,
                            dataloader_type,
                            global_batch_cnt,
                            skip_visualization,
                            self.trainer_param["visualization_image_format"],
                        )

                if dataloader_type == "train":
                    if self.additional_trainer_callbacks:
                        for cb in self.additional_trainer_callbacks:
                            cb.set_tqdm_iter_description(tqdm_iter, epoch, avg_g_cost, statistics, self.trainer_param)
                    elif self.use_discriminator:
                        tqdm_iter.set_description(
                            "Epoch {}, avg_g_cost {:.2f}, avg_d_cost {:.2f}, avg_l2_error {:.2f}, avg_ADE_error {:.2f}, avg_MoN_ADE_error {:.2f}".format(
                                epoch,
                                avg_g_cost,
                                avg_d_cost,
                                np.mean(np.concatenate(statistics["l2_error"])),
                                np.mean(np.concatenate(statistics["ade_error"])),
                                np.mean(np.concatenate(statistics["MoN_ade_error"])),
                            )
                        )
                    else:
                        tqdm_iter.set_description(
                            "Epoch {}, avg_g_cost {:.2f}, avg_l2_error {:.2f}, avg_ADE_error {:.2f}, avg_MoN_ADE_error {:.2f}".format(
                                epoch,
                                avg_g_cost,
                                np.mean(np.concatenate(statistics["l2_error"])),
                                np.mean(np.concatenate(statistics["ade_error"])),
                                np.mean(np.concatenate(statistics["MoN_ade_error"])),
                            )
                        )
                else:
                    if dataloader_type == "validation":
                        if self._early_stop_error_type in statistics:
                            regression_statistics.append(
                                np.mean(np.concatenate(statistics[self._early_stop_error_type]))
                            )
                    folder_postfix = "_best_fde"
                    # Rewrite key and postfix if necessary.
                    if self.additional_trainer_callbacks:
                        # The values are overwritten by the last callback.
                        for cb in self.additional_trainer_callbacks:
                            new_keys = cb.update_save_validation_criterion_key(self.trainer_param)
                            if new_keys:
                                folder_postfix = new_keys[1]

                    if len(regression_statistics) > 0:
                        new_save_validation_criterion = self.smoothed_error(regression_statistics)
                        if new_save_validation_criterion < save_validation_criterion:
                            save_validation_criterion = new_save_validation_criterion

                            # TODO(guy.rosman): save the lower-validation score model so far,
                            # TODO  in addition / instead of "latest model".
                            self.message_logger.log_message(
                                "Current best save criterion: {}".format(save_validation_criterion)
                            )
                            self.save_models(
                                epoch, global_batch_cnt, input_tensor, is_valid, folder_postfix=folder_postfix
                            )

                # No need to visualize for the vis dataloader because it will be visualized anyway. Similarly, we
                # also do not need to create files during debugging.
                skip_worst_case_visualization = (
                    skip_visualization or dataloader_type == "vis" or bool(self.trainer_param["data_debug_mode"])
                )
                for logging_handler in additional_logging_handlers:
                    logging_handler.epoch_end(
                        idx=epoch, global_batch_cnt=global_batch_cnt, skip_visualization=skip_worst_case_visualization
                    )

                if self.logger is not None:
                    self.logger.epoch_end(epoch=epoch, global_batch_cnt=global_batch_cnt)

                # TODO(igor.gilitschenski): This variable was originally neither a global batch count, nor a global
                # TODO  epoch count. Placing it here, turns it at least into a global epoch count. I did not rename it
                # TODO  for now due # to the big refactor freeze. But we should discuss how we want this to be.
                global_batch_cnt += 1
                self.profiler.step("after_dataset_loop")

            # Update schedulers
            if self.use_discriminator and self.d_scheduler:
                self.d_scheduler.step()
            if self.g_scheduler:
                self.g_scheduler.step()

            if self.debugger is not None:
                self.debugger.record_value("model_state", self.nondistributed_prediction_model.state_dict())
            self.profiler.step("epoch_end")

            self._save_periodic_models(epoch, global_batch_cnt, input_tensor, is_valid)
            if self.should_early_stop(regression_statistics):
                break

        if not self.trainer_param["disable_model_saving"]:
            self.message_logger.log_message("Saving model at end of training..")
            self.save_models(epoch, global_batch_cnt, input_tensor, is_valid)

        if self.debugger is not None:
            self.debugger.write_out(os.path.join(self.trainer_param["logs_dir"], self.tf_session_name))

    def get_session_name(self):
        """Get session name.

        Returns:
            str: The session name. Can be used to load the model afterwards.
        """
        return self.tf_session_name

    def get_optimizer_state(self):
        """Get optimizer dictionary -- to be used to keep optimization efficient between trainer calls.

        Returns:
            dict: The a mapping from names into optimizer variables.
        """
        optimizer_state_dict = {
            "g_optimizer_state": self.g_optimizer.state_dict(),
            "d_optimizer_state": self.d_optimizer.state_dict(),
        }

        return optimizer_state_dict

    def get_current_commit_comment(self):
        """Get commit comment so that we log which codebase is training.

        Returns
        -------
        str
            The abbreviated git commmit hash of HEAD.
        """
        try:
            res = check_output(["git", "log", "--pretty=fuller", "-n", "1"])
        except subprocess.CalledProcessError:
            self.message_logger.log_message("got subprocess.CalledProcessError in get_current_commit_comment()")
            res = ""
        return res
