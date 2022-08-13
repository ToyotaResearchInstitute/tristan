import hashlib
import os
from typing import Callable, Iterator, Optional

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from loaders.ado_key_names import AGENT_TYPE_CAR, AGENT_TYPE_PEDESTRIAN

# TODO add clarity to this type declaration
synthetic_data_generator = Callable[[dict, int, int, int, list, str], dict]


def generate_synthetic_data(params, max_timesteps, max_agents, item, agent_types, filename):
    """
    Generates synthetic data for prediction training
    :param params: A dictionary of parameters. Should have at least img_height, img_width.
    :param max_timesteps: The number of timesteps to generate.
    :param max_agents: The number of agents to generate.
    :param item: The data id -- should be used as a deterministic key for data items.
    :param agent_types: a dictionary from type to agent type id.
    :return: A dictionary with dot_keys,positions,is_ego_vehicle,is_relevant_pedestrian,idx_car,idx_pedestrian,
    timestamps, prediction_timestamp, instance_info, agent_type_vector
    """
    # Use the same random seed for the same index.
    cur_rng_state = np.random.get_state()
    pb_filename = os.path.split(filename)[1]
    hs = hashlib.md5((str(item) + pb_filename).encode())
    seed = int(hs.hexdigest(), 16) % np.iinfo(np.int32).max
    np.random.seed(seed)

    timestamps = np.linspace(0, max_timesteps - 1, max_timesteps)
    positions = np.random.normal(size=[max_timesteps, max_agents, 3]) * 0.01
    piecewise_coefficient = params["synthetic_dataset_piecewise_coefficient"]
    quadratic_coefficient = params["synthetic_dataset_quadratic_coefficient"]
    process_noise_coefficient = params["synthetic_dataset_process_noise_coefficient"]
    positions += np.cumsum(np.random.normal(size=[max_timesteps, max_agents, 3]), 0) * process_noise_coefficient
    for agent_id in range(max_agents):
        hs = hashlib.md5((str(item + agent_id) + pb_filename).encode())
        positions[:, agent_id, 0] += ((int(hs.hexdigest(), 16) % 10) - 4.5) * timestamps
        positions[:, agent_id, 1] += (((int(hs.hexdigest(), 16) % 100) // 10) - 4.5) * timestamps
        positions[max_timesteps // 3 :, agent_id, 0] += (
            piecewise_coefficient
            * (((int(hs.hexdigest(), 16) % 1000) // 100) - 4.5)
            * (timestamps[max_timesteps // 3 :] - timestamps[max_timesteps // 3])
            / 5
        )
        positions[max_timesteps // 3 :, agent_id, 1] += (
            piecewise_coefficient
            * (((int(hs.hexdigest(), 16) % 10000) // 1000) - 4.5)
            * (timestamps[max_timesteps // 3 :] - timestamps[max_timesteps // 3])
            / 5
        )
        positions[(max_timesteps * 2) // 3 :, agent_id, 0] += (
            piecewise_coefficient
            * (((int(hs.hexdigest(), 16) % 100000) // 10000) - 4.5)
            * (timestamps[(max_timesteps * 2) // 3 :] - timestamps[(max_timesteps * 2) // 3])
            / 5
        )
        positions[(max_timesteps * 2) // 3 :, agent_id, 1] += (
            piecewise_coefficient
            * (((int(hs.hexdigest(), 16) % 1000000) // 100000) - 4.5)
            * (timestamps[(max_timesteps * 2) // 3 :] - timestamps[(max_timesteps * 2) // 3])
            / 5
        )
        positions[:, agent_id, 0] += (((int(hs.hexdigest(), 16) % 100000000) // 10000000) - 4.5) * max_timesteps / 2.0
        positions[:, agent_id, 1] += (((int(hs.hexdigest(), 16) % 1000000000) // 100000000) - 4.5) * max_timesteps / 2.0
        positions[:, agent_id, 2] = 1.0
        # Add quadratic components (non-straight parts) to the synthetic trajectories.
        positions[:, agent_id, 0] += (
            (((int(hs.hexdigest(), 16) % 10000000000) // 1000000000) - 4.5)
            * (timestamps / max_timesteps) ** 2
            * quadratic_coefficient
        )
        positions[:, agent_id, 1] += (
            (((int(hs.hexdigest(), 16) % 100000000000) // 10000000000) - 4.5)
            * (timestamps / max_timesteps) ** 2
            * quadratic_coefficient
        )
    instance = None
    time_addition = ((int(hs.hexdigest(), 16) % 1000000000000) // 10000) / 1000
    instance_timestamp = 1615307581.0 + time_addition
    instance_info = '{"timestamp":' + str(instance_timestamp) + ',"source_tlog":"","json_dir":""}'
    prediction_timestamp = timestamps[max_timesteps // 2]
    idx = range(max_agents)
    is_ego_vehicle = np.zeros([max_agents])
    is_ego_vehicle[0] = 1
    is_relevant_pedestrian = np.zeros([max_agents])
    # Avoid the assumption, and read the additional inputs to decide on the relevant / ego agent.
    if max_agents > 1:
        is_relevant_pedestrian[1] = 1
    dot_keys = np.array(np.arange(max_agents))
    dot_keys[0] = -2  # equivalent dot key of ego-vehicle
    agent_type_vector = np.zeros((max_agents, len(agent_types)))
    idx_car = agent_types.index(AGENT_TYPE_CAR)
    idx_pedestrian = agent_types.index(AGENT_TYPE_PEDESTRIAN)
    agent_type_vector[0, idx_car] = 1
    if max_agents > 1:
        agent_type_vector[1:, idx_pedestrian] = 1
    result = {
        "dot_keys": dot_keys,
        "positions": positions,
        "is_ego_vehicle": is_ego_vehicle,
        "is_relevant_pedestrian": is_relevant_pedestrian,
        "instance_info": instance_info,
        "idx_pedestrian": idx_pedestrian,
        "idx_car": idx_car,
        "timestamps": timestamps,
        "agent_type_vector": agent_type_vector,
        "prediction_timestamp": prediction_timestamp,
        "idx": idx,
        "instance": instance,
    }
    # Restore RNG state.
    np.random.set_state(cur_rng_state)
    return result


def generate_synthetic_linear_data(params, max_timesteps, max_agents, item, agent_types, filename):
    """
    Generates synthetic linear data for prediction training.
    The data includes hybrid sequences sampled from a static transition function and a linear dynamics under each mode.
    :param params: A dictionary of parameters. Should have at least img_height, img_width.
    :param max_timesteps: The number of timesteps to generate.
    :param max_agents: The number of agents to generate.
    :param item: The data id -- should be used as a deterministic key for data items.
    :param agent_types: a dictionary from type to agent type id.
    :param filename: unique file name.
    :return: A dictionary with dot_keys,positions,is_ego_vehicle,is_relevant_pedestrian,idx_car,idx_pedestrian,
    timestamps, prediction_timestamp, instance_info, agent_type_vector
    """
    # Use the same random seed for the same index.
    cur_rng_state = np.random.get_state()
    pb_filename = os.path.split(filename)[1]
    hs = hashlib.md5((str(item) + pb_filename).encode())
    seed = int(hs.hexdigest(), 16) % 1000000000
    np.random.seed(seed)

    timestamps = np.linspace(0, max_timesteps - 1, max_timesteps)
    prediction_timestamp = params["past_timesteps"]
    # Add standard random noise to positions.
    pos_std = 2.0
    positions = np.random.normal(size=[max_timesteps, max_agents, 2]) * pos_std
    validity = np.ones([max_timesteps, max_agents, 1])
    positions = np.concatenate((positions, validity), -1)
    modes = np.zeros([max_timesteps, max_agents])

    # Update positions given linear dynamics: x_{t+1} = x_t + B + noise.
    B = [1.0, 2.0]
    for agent_id in range(max_agents):
        for t in range(max_timesteps):
            if t == 0:
                continue
            positions_prev = positions[t - 1, agent_id]
            positions[t, agent_id, 0] += positions_prev[0] + B[0]
            positions[t, agent_id, 1] += positions_prev[1] + B[1]
    time_addition = ((int(hs.hexdigest(), 16) % 1000000000000) // 10000) / 1000
    instance_timestamp = 1615307581.0 + time_addition
    instance_info = '{"timestamp":' + str(instance_timestamp) + ',"source_tlog":"","json_dir":""}'
    instance = None
    idx = range(max_agents)
    is_ego_vehicle = np.zeros([max_agents])
    is_ego_vehicle[0] = 1
    is_relevant_pedestrian = np.zeros([max_agents])
    # Avoid the assumption, and read the additional inputs to decide on the relevant / ego agent.
    if max_agents > 1:
        is_relevant_pedestrian[1] = 1
    dot_keys = np.array(np.arange(max_agents))
    dot_keys[0] = -2  # equivalent dot key of ego-vehicle
    agent_type_vector = np.zeros((max_agents, len(agent_types)))
    idx_car = agent_types.index(AGENT_TYPE_CAR)
    idx_pedestrian = agent_types.index(AGENT_TYPE_PEDESTRIAN)
    agent_type_vector[0, idx_car] = 1
    if max_agents > 1:
        agent_type_vector[1:, idx_pedestrian] = 1
    result = {
        "dot_keys": dot_keys,
        "positions": positions,
        "modes": modes,
        "is_ego_vehicle": is_ego_vehicle,
        "is_relevant_pedestrian": is_relevant_pedestrian,
        "instance_info": instance_info,
        "idx_pedestrian": idx_pedestrian,
        "idx_car": idx_car,
        "timestamps": timestamps,
        "agent_type_vector": agent_type_vector,
        "prediction_timestamp": prediction_timestamp,
        "idx": idx,
        "instance": instance,
    }
    # Restore RNG state.
    np.random.set_state(cur_rng_state)
    return result


class SyntheticDataset(Dataset):
    def __init__(
        self,
        generator: synthetic_data_generator,
        params: dict,
        max_timesteps: int,
        max_agents: int,
        agent_types: list,
        filename: str = None,
        dataset_length: int = None,
    ):
        self.generator = generator
        self.params = params
        self.max_timesteps = max_timesteps
        self.max_agents = max_agents
        self.agent_types = agent_types
        self.filename = filename

        if dataset_length is not None:
            self.dataset_length = dataset_length
        else:
            self.dataset_length = params["max_files_count"]

    def __getitem__(self, index) -> T_co:
        return self.generator(self.params, self.max_timesteps, self.max_agents, index, self.agent_types, self.filename)

    def __iter__(self) -> Iterator[T_co]:
        for item in range(self.dataset_length):
            yield self[item]
