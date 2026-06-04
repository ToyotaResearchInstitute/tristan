# Copyright 2020 Toyota Research Institute. All rights reserved.
import logging
from collections.abc import MutableMapping, MutableSequence
from itertools import islice
from typing import Callable, Iterator, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, SubsetRandomSampler

from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset, dataset_collate_fn
from triceps.protobuf.prediction_dataset_auxiliary import AuxiliaryStateIndices
from triceps.protobuf.prediction_dataset_map_handlers import MapDataIndices

# A parameter key for a map from source dataset, parameters to which prediction instances are valid.
from triceps.protobuf.protobuf_training_parameter_names import PARAM_FUTURE_TIMESTEPS, PARAM_PAST_TIMESTEPS


def walk_batch(item: Union[MutableSequence, MutableMapping], types, func: Callable):
    """A helper method to recursively walk through nested dicts and lists and modify them in place

    This method will walk through the provided item, recursively checking any lists or dicts it finds, and replace
    objects that match `isinstance(value, types)` with the result of `func(value)`.

    Other collection types besides dict and list, such as tuples, will not be checked recursively, even if
    they contain other dicts and lists.

    :param item: dict or list to walk through
    :param types: the types that `func` should be called on
    :param func: a callable that should expect an object of the specified types and returns a modified version of the
        object, which will replace the original object in the parent collection

    returns the modified item
    """
    if isinstance(item, MutableMapping):
        union_iterable = item.items()
    else:
        union_iterable = enumerate(item)

    for key, value in union_iterable:
        if isinstance(value, types):
            item[key] = func(value)
        elif isinstance(value, (MutableSequence, MutableMapping)):
            item[key] = walk_batch(value, types, func)

    return item


def coerce_numpy_dtype_bits(item: dict):
    # Normalize dtypes to 32 bit
    def coerce_array(array: Union[np.ndarray, np.number]):
        if array.dtype == np.float64:
            return array.astype(np.float32)
        else:
            return array

    return walk_batch(item, (np.ndarray, np.number), coerce_array)


def coerce_tensor_dtype_bits(item: Union[dict, list]):
    # Normalize dtypes to 32 bit
    def coerce_tensor(tensor: torch.Tensor):
        if tensor.dtype == torch.float64:
            return tensor.float()
        else:
            return tensor

    return walk_batch(item, torch.Tensor, coerce_tensor)


def compute_global_normalization(input_tensor, is_valid):
    """
    Compute global normalization, through averaging all valid points
    :param input_tensor: normalized trajectory tensor,
        with shape [size, num_agents, num_past_timepoints, trajectory_dimension].
    :param positions_tensor: raw trajectory tensor,
        with shape [size, num_agents, num_all_timepoints, trajectory_dimension].
    :param is_valid: validity of each time point,
        with shape [size, num_agents, num_past_timepoints].
    :return: updated input_tensor, with normalization offsets (offset_x, offset_y)
    """
    # num_agents = is_valid.shape[0]

    # Compute offsets as center of all valid positions.
    offset_x = input_tensor[..., 0][is_valid].mean()
    offset_y = input_tensor[..., 1][is_valid].mean()

    # Center trajectories.
    input_tensor[..., 0] -= offset_x
    input_tensor[..., 1] -= offset_y
    return input_tensor, offset_x, offset_y


def get_normalized_trajectories(
    params,
    positions_tensor: np.ndarray,
    is_valid: np.ndarray,
    num_past_timepoints: int,
    num_future_timepoints: int,
):
    # Compute agent-wise normalized positions.
    if params["disable_position_input"]:
        shape = positions_tensor.shape
        input_tensor = np.zeros((shape[0], shape[1], num_past_timepoints, shape[3]))
        offset_x = np.zeros(2)
        offset_y = np.zeros(2)
    else:
        input_tensor = np.copy(positions_tensor[..., :num_past_timepoints, :])
        input_tensor, offset_x, offset_y = compute_global_normalization(input_tensor, is_valid)

    # Obtain centered observed future trajectory from data.
    expected_trajectories_scene = np.copy(positions_tensor[..., -num_future_timepoints:, :2])
    expected_trajectories_scene[..., 0] -= offset_x
    expected_trajectories_scene[..., 1] -= offset_y

    return input_tensor, expected_trajectories_scene, offset_x, offset_y


def get_inputs_from_item(item, num_past_timepoints):
    additional_input_keys = [
        ProtobufPredictionDataset.DATASET_KEY_IMAGES,
        ProtobufPredictionDataset.DATASET_KEY_IMAGES_MAPPING,
        ProtobufPredictionDataset.DATASET_KEY_MAP_STUB,
        ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE,
    ]
    additional_inputs = {k: v for k, v in item.items() if k in additional_input_keys}

    if ProtobufPredictionDataset.DATASET_KEY_STUB in item:
        additional_inputs[ProtobufPredictionDataset.DATASET_KEY_STUB] = item[
            ProtobufPredictionDataset.DATASET_KEY_STUB
        ][:num_past_timepoints, ...]

    # Load additional inputs of each agent.
    agent_additional_input_keys = [
        ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES,
        ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_MAPPING,
    ]
    agent_additional_inputs = {k: v for k, v in item.items() if k in agent_additional_input_keys}
    if ProtobufPredictionDataset.DATASET_KEY_AGENT_STUB in item:
        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AGENT_STUB] = item[
            ProtobufPredictionDataset.DATASET_KEY_AGENT_STUB
        ][..., :num_past_timepoints, :]
    if ProtobufPredictionDataset.DATASET_KEY_AGENT_MAP_STUB in item:
        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AGENT_MAP_STUB] = item[
            ProtobufPredictionDataset.DATASET_KEY_AGENT_MAP_STUB
        ][..., :num_past_timepoints, :]

    if ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE in item:
        # Auxliary state includes (length, width, yaw, vel_x, vel_y) per time step per agent.
        auxiliary_state = item[ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE]
        auxiliary_state = auxiliary_state[..., :num_past_timepoints, :].astype(np.float32)
        # Add absolute velocity to auxiliary state.
        auxiliary_state_vel_abs = np.sqrt(
            auxiliary_state[..., AuxiliaryStateIndices.AUX_STATE_IDX_VEL_X] ** 2
            + auxiliary_state[..., AuxiliaryStateIndices.AUX_STATE_IDX_VEL_Y] ** 2
        )

        auxiliary_state = torch.cat(
            (torch.tensor(auxiliary_state), torch.tensor(auxiliary_state_vel_abs).unsqueeze(-1)), -1
        )

        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE] = auxiliary_state

    return additional_inputs, agent_additional_inputs


def augment_data(trainer_param, trajectories_global: np.ndarray) -> None:
    """Do data augmentation on batch_itm, rotation/translating/adding noise.

    The augmentation happens "in-place". Thus, no value is returend.

    Parameters
    ----------
    prediction_instance : dict
        The dataloader dictionary
    """
    num_agents, num_timestamps, _ = trajectories_global.shape

    # Augment trajectories by rotation.
    # TODO(guy.rosman): move to function.
    augmentation_angles = np.random.uniform()
    augmentation_angles -= 0.5
    augmentation_angles *= trainer_param["augmentation_angle_scale"] * 2.0

    sn = np.tile(np.sin(augmentation_angles), (num_agents, num_timestamps))
    cs = np.tile(np.cos(augmentation_angles), (num_agents, num_timestamps))

    augmentation_offset_x = np.tile(
        np.random.normal() * trainer_param["augmentation_translation_scale"], (num_agents, num_timestamps)
    )
    augmentation_offset_y = np.tile(
        np.random.normal() * trainer_param["augmentation_translation_scale"], (num_agents, num_timestamps)
    )
    # A noise source for every timestep
    random_walk = np.random.normal(size=trajectories_global[:, :, :2].shape)

    # Add a random walk with spring force, to prevent a big difference from the source.
    spring_force = trainer_param["augmentation_process_noise_spring"]
    for t in range(trajectories_global.shape[1]):
        # Set an additional force -- either spring force to bound the process noise, or 0 (first time step)
        if t > 0:
            dprocess_noise_t = -random_walk[:, t - 1, :] * spring_force

            # accumulate random steps
            random_walk[:, t, :] += random_walk[:, t - 1, :]
        else:
            dprocess_noise_t = 0

        random_walk[:, t, :] += dprocess_noise_t
    random_walk *= trainer_param["augmentation_process_noise_scale"]

    # TODO(guy.rosman): Fix augmentation for map inputs
    if not trainer_param["disable_map_input"]:
        # Currently map handlers do not transform their content. Hence, we prevent augmentation.
        cs = np.ones_like(cs)
        sn = np.zeros_like(sn)
        augmentation_offset_x *= 0
        augmentation_offset_y *= 0

    dx = np.random.normal(size=(num_agents, num_timestamps)) * trainer_param["augmentation_noise_scale"]
    dy = np.random.normal(size=(num_agents, num_timestamps)) * trainer_param["augmentation_noise_scale"]
    nx = (
        trajectories_global[:, :, 0] * cs
        - trajectories_global[:, :, 1] * sn
        + augmentation_offset_x
        + dx
        + random_walk[:, :, 0]
    )
    ny = (
        trajectories_global[:, :, 0] * sn
        + trajectories_global[:, :, 1] * cs
        + augmentation_offset_y
        + dy
        + random_walk[:, :, 1]
    )

    trajectories_global[:, :, 0] = nx
    trajectories_global[:, :, 1] = ny


def full_preprocess(prediction_instance: dict, trainer_param: dict):
    pass_through_keys = [
        ProtobufPredictionDataset.DATASET_KEY_AGENT_TYPE,
        ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO,
        ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT,  # one-hot is_relevant_pedestrian
        ProtobufPredictionDataset.DATASET_KEY_PREDICTION_TIMESTAMP,
        ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE,
        ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS,
        ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS_FULL,
        ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_LABELS,
        # These are expected by the worse case logger
        # TODO(nicholas.guyett.ctr) prevent the data duplication between this and the additional inputs
        ProtobufPredictionDataset.DATASET_KEY_POSITIONS,
        ProtobufPredictionDataset.DATASET_KEY_POSITIONS_FULL,
        ProtobufPredictionDataset.DATASET_KEY_DOT_KEYS,
        ProtobufPredictionDataset.DATASET_KEY_RAW_AGENT_IDS,
        ProtobufPredictionDataset.DATASET_KEY_IS_EGO_VEHICLE,
        ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS,
        ProtobufPredictionDataset.DATASET_KEY_IMAGES,
        ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES,
        ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_MAPPING,
        ProtobufPredictionDataset.DATASET_KEY_MAP,
        ProtobufPredictionDataset.DATASET_KEY_MAP_SEGMENTS_LENGTH,
        "map_vectors",
        "map_vectors_valid",
        ProtobufPredictionDataset.DATASET_KEY_LANES,
        ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE_FULL,
        # TODO allow data handlers to add additional results here
        # Thunderhill data
        "driver_id",
        "session_id",
        "acc",
        "throttle",
        "brake",
        "steer_angle",
        "racing_line_lateral_error",
        "racing_line_side",
        "subtitle_label",
        "subtitle_valid",
        # Filter
        "filter_result",
        "stats",
        # Waymo sequence
        "instance_label",
        "action",
        "action_label",
        "following_agent",
        "alpha_beta_gamma",
        "alpha_beta_gamma_noise",
    ]
    result = {k: v for k, v in prediction_instance.items() if k in pass_through_keys or "score_" in k}

    if trainer_param["augment_trajectories"]:
        # Augmentation must occur prior to any other preprocessing
        # TODO(guy.rosman): GENERALIZATION - make augmentation modifiable / a functor
        augment_data(trainer_param, prediction_instance[ProtobufPredictionDataset.DATASET_KEY_POSITIONS])

    # Read positions, validity, timestamps.
    num_past_timepoints = trainer_param[PARAM_PAST_TIMESTEPS]
    num_future_timepoints = trainer_param[PARAM_FUTURE_TIMESTEPS]

    full_timestamps = prediction_instance[ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS]
    result["timestamps_past"] = full_timestamps[:num_past_timepoints]
    result["timestamps_future"] = full_timestamps[-num_future_timepoints:]

    full_is_valid = prediction_instance[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][..., 2].astype(bool)
    is_valid_past = full_is_valid[:, :num_past_timepoints]
    is_valid_future = full_is_valid[:, -num_future_timepoints:]
    result["is_valid_past"] = is_valid_past
    result["is_valid_future"] = is_valid_future
    result["is_valid_full"] = full_is_valid

    if not trainer_param["ignore_ego"]:
        # Add ego car to relevant agent.
        is_rel = prediction_instance[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT]  # one-hot is_ego_vehicle
        is_ego = prediction_instance[ProtobufPredictionDataset.DATASET_KEY_IS_EGO_VEHICLE]  # one-hot is_ego_vehicle
        result[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT] = is_ego + is_rel

    scale = trainer_param["predictor_normalization_scale"]
    trajectories_global = prediction_instance[ProtobufPredictionDataset.DATASET_KEY_POSITIONS] * scale
    past_trajectories_scene, future_trajectories_scene, offset_x, offset_y = get_normalized_trajectories(
        trainer_param,
        trajectories_global,
        is_valid_past,
        num_past_timepoints,
        num_future_timepoints,
    )

    result["past_trajectories_scene"] = past_trajectories_scene
    result["future_trajectories_scene"] = future_trajectories_scene
    result["trajectories_global"] = trajectories_global
    result["offset_x"] = offset_x.repeat([trajectories_global.shape[0]])  # repeat is needed for backwards compatibility
    result["offset_y"] = offset_y.repeat([trajectories_global.shape[0]])  # repeat is needed for backwards compatibility

    # Load global additional inputs: images, maps, test stub.
    additional_inputs, agent_additional_inputs = get_inputs_from_item(prediction_instance, num_past_timepoints)

    if trainer_param["disable_position_input"]:
        additional_inputs["future_positions"] = future_trajectories_scene * 0
    else:
        additional_inputs["future_positions"] = future_trajectories_scene

    result["additional_inputs"] = additional_inputs
    result["agent_additional_inputs"] = agent_additional_inputs

    # Load map tensor, depending on different type.
    if not trainer_param["disable_map_input"] and ProtobufPredictionDataset.DATASET_KEY_MAP in result:
        if trainer_param["use_global_map"]:
            map_inputs = additional_inputs
        else:
            map_inputs = agent_additional_inputs

        # These point-map views are not consumed by the Waymo sequence model when
        # map_vectors are enabled. They are kept for map rendering, worst-case
        # logging, and offline visualization/debugging, so allow training runs
        # that do not visualize maps to avoid carrying them through the dataloader.
        map_coordinates, map_validity, map_others = scale_maps(
            trainer_param["map_input_type"],
            scale,
            prediction_instance,
            map_inputs,
            offset_x,
            offset_y,
        )

        if not trainer_param.get("disable_map_visualization_fields", False):
            result["map_coordinates"] = map_coordinates
            result["map_validity"] = map_validity
            result["map_others"] = map_others

    return coerce_numpy_dtype_bits(result)


def scale_maps(map_input_type, scale, itm, agent_additional_inputs, offset_x, offset_y):
    """Normalizes the scales and offsets of the maps to get the "global normalization".

    Parameters
    ----------
    scale : float
        The correction scale.
    itm : dict
        The dataloader dictionary
    agent_additional_inputs: dict
        The additional input tensors dictionary.
    offset_x : np.ndarray,
        The x offset for instances.
    offset_y : np.ndarray,
        The y offset for instances.

    Returns:
    -------
    map_positions_tensor : np.ndarray
        The positions of the map elements of shape (size, max_point_num, 2)
    map_validity_tensor : np.ndarray
        The positions of the map elements of shape (size, max_point_num)
    map_others_tensor : np.ndarray
        Point type, tangent information, normal information, point id.
        (size, max_point_num, 6). The elements in the last 6 dimensions are
        structured as follows (point type, sin(theta), cos(theta), cos(theta), -sin(theta), poind id)),
        where point type is corresponding to the integer in MapPointType.
    """
    num_past_timepoints = itm[ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS]
    map_positions_tensor = None
    map_validity_tensor = None
    map_others_tensor = None
    if ProtobufPredictionDataset.DATASET_KEY_MAP in itm:
        if map_input_type == "point":
            map_positions_tensor = itm[ProtobufPredictionDataset.DATASET_KEY_MAP][..., :2].astype(np.float32) * scale
            map_validity_tensor = itm[ProtobufPredictionDataset.DATASET_KEY_MAP][
                ..., MapDataIndices.MAP_IDX_VALIDITY
            ].astype(np.float32)
            map_others_tensor = itm[ProtobufPredictionDataset.DATASET_KEY_MAP][
                ..., MapDataIndices.MAP_IDX_TYPE :
            ].astype(np.float32)
        else:
            agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_MAP] = itm[
                ProtobufPredictionDataset.DATASET_KEY_MAP
            ][:num_past_timepoints, :, :, :]

        # Normalize map points in global frame (not per agent).
        if map_input_type == "point":
            agent_additional_inputs[
                ProtobufPredictionDataset.DATASET_KEY_MAP
            ] = update_global_normalized_map_point_tensors(
                map_positions_tensor,
                map_validity_tensor,
                map_others_tensor,
                offset_x,
                offset_y,
            )

    return map_positions_tensor, map_validity_tensor, map_others_tensor


def update_global_normalized_map_point_tensors(
    map_positions_tensor: np.ndarray,
    map_validity_tensor: np.ndarray,
    map_others_tensor: np.ndarray,
    offset_x,
    offset_y,
):
    """
    Compute global map tensors through shifting.
    :param map_positions_tensor: original map positions,
        with shape [size, num_max_points, point_dimension].
    :param map_validity_tensor: map validity,
        with shape [size, num_max_points].
    :param map_others_tensor: additional map information,
        with shape [size, num_max_points, additional_info_dimension].
    :param offset_x: delta x for global map centering, with shape [size, num_agents].
    :param offset_y: delta y for global map centering, with shape [size, num_agents].
    :return: Updated map inputs tensor.
    """
    # Read raw inputs.
    input_map_tensor = np.copy(map_positions_tensor)
    input_map_tensor_validity = np.copy(map_validity_tensor)
    input_map_tensor_others = np.copy(map_others_tensor)

    map_x = map_positions_tensor[..., 0] - offset_x
    map_y = map_positions_tensor[..., 1] - offset_y
    map_xy = np.stack((map_x, map_y), axis=-1)

    input_map_tensor[..., 0] = map_xy[..., 0] * input_map_tensor_validity
    input_map_tensor[..., 1] = map_xy[..., 1] * input_map_tensor_validity
    input_map_tensor = np.concatenate(
        (input_map_tensor, input_map_tensor_validity[..., None], input_map_tensor_others), axis=-1
    )
    # Get map output, with shape [num_batch, num_max_points, point_dim].
    return input_map_tensor


class DatasetsFilter:
    def __init__(self, params):
        pass

    def filter(self, datasets, other_inputs, dataset_type):
        return datasets, other_inputs


class NewPredictionDataset(ConcatDataset):
    def __init__(self, datasets, params, dataset_label=None, other_input=None):
        # TODO(nicholas.guyett.ctr) move dataset initialization logic from trainer_utils into here
        self.params = params
        self.dataset_label = dataset_label
        self.rel_filenames = [p.rel_filename for p in datasets]
        self.other_input = other_input
        super().__init__(datasets)

    def __getitem__(self, item):
        itm = super().__getitem__(item)
        itm["dataset_index"] = item
        if self.params["use_mpa_data"] or self.params["use_mpa_rad_data"]:
            return itm
        return full_preprocess(itm, self.params)

    def compute_dataset_stats(self):
        ret = {"total_instances": len(self.datasets)}
        return ret


class Buffer:
    def __init__(self, input_hz, output_hz, queue_time):
        self.input_hz = input_hz
        self.output_hz = output_hz
        self.queue_time = queue_time
        self.max_len = int(queue_time * 1.1) * input_hz
        self.queue = []
        assert input_hz % output_hz == 0

        step = self.input_hz // self.output_hz
        item_idxs = list(range(self.max_len))
        latest_first_item_idxs = item_idxs[::-1]
        self.picked_idxs = set(latest_first_item_idxs[::step])

    def append(self, data):
        self.queue.append(data)
        if len(self.queue) > self.max_len:
            self.queue.pop(0)

    def get(self):
        if len(self.queue) < self.max_len:
            return None
        arr = [d for i, d in enumerate(self.queue) if i in self.picked_idxs]
        return arr


class NewPredictionDataset_OnlineInferenceThunderhill(ConcatDataset):
    def __init__(self, datasets, params, dataset_label=None):
        # TODO(nicholas.guyett.ctr) move dataset initialization logic from trainer_utils into here
        self.params = params
        self.dataset_label = dataset_label
        self.rel_filenames = [p.rel_filename for p in datasets]
        self.carla_topic = ""
        super().__init__(datasets)

    def __getitem__(self, item):
        itm = super().__getitem__(item)
        itm["dataset_index"] = item
        if self.params["use_mpa_data"] or self.params["use_mpa_rad_data"]:
            return itm
        return full_preprocess(itm, self.params)


def collate_inputs(batch):
    batch = coerce_tensor_dtype_bits(batch)  # Coerces all float64 tensors to 32 bits

    batch = dataset_collate_fn(batch)

    return batch


class PseudoRandomSampler:
    """Given a list of indices, sample without replacement until exhausted the list."""

    def __init__(self, indices, epoch_size) -> None:
        self.epoch_size = epoch_size
        self.indices = indices
        self.random_indices_generator = None
        self.create_permutation()

    def create_permutation(self):
        # Create a permutation of the entire dataset
        self.random_indices_generator = (i for i in list(np.random.permutation(self.indices)))

    def __iter__(self):
        for i in range(self.epoch_size):
            try:
                a = next(self.random_indices_generator)
                yield a
            except StopIteration:
                self.create_permutation()
                a = next(self.random_indices_generator)
                yield a

    def __next__(self):
        try:
            a = next(self.random_indices_generator)
            return a
        except StopIteration:
            self.create_permutation()
            a = next(self.random_indices_generator)
            return a

    def __len__(self) -> int:
        return self.epoch_size


class WorstCaseSampler:
    """Sample worst scenarios more in the batch.
    The batch_cost is stored and sorted to select the top %10 worst case.

    Example:
        all batch_cost is stored, indexed by scenario_id.
        Then sort batch_cost to get the highest %10 scenarios and sample %20 of the batch from those.
        The rest of the %80 of the batch are randomly sampled as before.
    """

    def __init__(self, worst_case_info, indices, worst_case_ratio) -> None:
        self.normal_sampler = SubsetRandomSampler(indices)
        self.indices = indices
        self.worst_case_info = worst_case_info
        self.worst_case_ratio = worst_case_ratio

    def __iter__(self):
        if not self.worst_case_info.draw_with_worst:
            return iter(self.normal_sampler)

        normal_samples = list(iter(self.normal_sampler))
        size = len(normal_samples)
        num_worst_cases = int(size * self.worst_case_ratio)
        normal_cases = size - num_worst_cases
        worst_sample_idx = self.worst_case_info.sample_worst_case(num_worst_cases)
        worst_sample_idx = worst_sample_idx[:num_worst_cases]
        normal_samples = normal_samples[:normal_cases] + worst_sample_idx.tolist()
        return (i for i in np.random.permutation(normal_samples))

    def __len__(self) -> int:
        return len(self.indices)


def create_sampler(dataloader_info, params, epoch_size, balanced_indices):
    sampler = PseudoRandomSampler(balanced_indices, epoch_size)
    dataset = dataloader_info["dataset"]

    # Worst case resampler
    worst_case_ratio = params["resample_worst_case_ratio"]
    worst_case_info = dataloader_info.get("worst_case_info", None)
    use_worst_case_sampler = params["resample_worst_case_start_epoch"] >= 0
    if use_worst_case_sampler and worst_case_ratio > 0 and worst_case_info is not None:
        sampler = WorstCaseSampler(worst_case_info, balanced_indices, worst_case_ratio)
    if params["use_thunderhill_dataset"] and params["use_two_tone_sampler"]:
        should_shuffle = False  # shuffling occurs in sampler
        meta_data_info_dir = params["input_dir"][0]
        from intent.multiagents.data_specific.thunderhill.thunderhill_dataset import TwoToneSampler

        sampler = TwoToneSampler(dataset, meta_data_info_dir, params, dataloader_info["dataset_key"])
        dataloader_info["sampler"] = sampler
    return sampler


class PredictionDataLoader(DataLoader):
    def __init__(
        self,
        dataloader_info: dict,
        params: dict,
        balanced_indices=None,
        full_dataset_epochs=None,
        **kwargs,
    ):
        self.dataloader_info = dataloader_info

        dataset = dataloader_info["dataset"]
        batch_size = dataloader_info["batch_size"]
        num_workers = dataloader_info.get("num_workers", 0)
        worst_case_info = dataloader_info.get("worst_case_info", None)
        worst_case_ratio = params["resample_worst_case_ratio"]

        if full_dataset_epochs is None:
            full_dataset_epochs = self.dataloader_info["full_dataset_epochs"]
        if num_workers > 0:
            # Prefetch factor is only relevant when we have at least one worker process
            # DataLoader will throw an error if is num_workers = 0 and the prefetch factor is not left at the default
            kwargs.setdefault("prefetch_factor", params["dataloader_pre_fetch_factor"])

        # full data vs. sample from data
        if full_dataset_epochs:
            self.epoch_size = len(dataset)
            if self.epoch_size != dataloader_info["epoch_size"]:
                logging.warning("Ignoring configured epoch_size due to --full-dataset-epochs")
            should_shuffle = dataloader_info.get("should_shuffle", True)
        else:
            corrected_epoch_size = int(batch_size * max(dataloader_info["epoch_size"] // batch_size, 2))
            dataloader_info["epoch_size"] = corrected_epoch_size
            self.epoch_size = corrected_epoch_size
            # Don't shuffle (in Dataloader) when we use samplers
            should_shuffle = False

            if balanced_indices is None:
                balanced_indices = range(len(dataset))
            # Create sampler (not full data epoch), the sampler is persisted across epochs
            sampler = dataloader_info.get("sampler")
            if sampler is not None:
                kwargs["sampler"] = sampler
            else:
                sampler = create_sampler(dataloader_info, params, self.epoch_size, balanced_indices)
                kwargs["sampler"] = sampler
                dataloader_info["sampler"] = sampler
        logging.warning(f"Using epoch size of {self.epoch_size}")

        if len(dataset) < batch_size * 2:
            logging.warning("len(dataset)<batch_size*2, {},{}.".format(len(dataset), batch_size))
        dataset.trajectory_generator = None
        super().__init__(
            dataset,
            shuffle=should_shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=params["dataloader_pin_memory"],
            collate_fn=collate_inputs,
            **kwargs,
        )

    def __len__(self) -> int:
        total_batches = self.epoch_size / self.dataloader_info["batch_size"]
        if self.drop_last:
            total_batches = int(np.floor(total_batches))
        else:
            total_batches = int(np.ceil(total_batches))

        return min(super().__len__(), total_batches)

    def __iter__(self) -> Iterator[dict]:
        ret = islice(
            super().__iter__(),
            len(self),
        )
        if hasattr(self.sampler, "on_batch_end"):
            self.sampler.on_batch_end()
        return ret
