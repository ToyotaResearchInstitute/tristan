import hashlib
import os
from typing import Union

import numpy as np

from loaders.ado_key_names import AGENT_TYPE_CAR, AGENT_TYPE_PEDESTRIAN


def generate_hybrid_synthetic_data(
    params: dict, max_timesteps: int, max_agents: int, item: Union[int, np.int64], agent_types: list, filename: str
):
    """
    Generates synthetic data including maneuvers for hybrid prediction training.

    Parameters
    ----------
    params : dict
        A dictionary of parameters.
    max_timesteps : int
        The number of timesteps to generate.
    max_agents : int
        The number of agents to generate.
    item : int
        The data id -- should be used as a deterministic key for data items.
    agent_types : list
        A dictionary from type to agent type id.
    filename : str
        Unique file name.

    Returns
    -------
    result : dict
        A data dictionary.
    """
    # Use the same random seed for the same index.
    cur_rng_state = np.random.get_state()
    pb_filename = os.path.split(filename)[1]
    hs = hashlib.md5((str(item) + pb_filename).encode())
    seed = int(hs.hexdigest(), 16) % 1000000000
    np.random.seed(seed)

    timestamps = np.linspace(0, max_timesteps - 1, max_timesteps)
    prediction_timestamp = params["past_timesteps"]
    # TODO(cyrushx): Add noise to positions.
    positions = np.random.normal(size=[max_timesteps, max_agents, 2]) * 0.1
    validity = np.ones([max_timesteps, max_agents, 1])
    positions = np.concatenate((positions, validity), -1)
    modes = np.zeros([max_timesteps, max_agents])

    # Define three modes (forward, up, down) with different slopes.
    # TODO(cyrushx): Change slope to see if the network can learn.
    mode_slopes = [0, 2.0, -1.0]
    num_modes = len(mode_slopes)
    # The transition is unbiased.
    mode_transitions = np.array([[0.6, 0.2, 0.2], [0.2, 0.8, 0.0], [0.2, 0.0, 0.8]])
    # Get starting mode
    mode_priors = np.array([0.4, 0.3, 0.3])
    mode_start = np.random.choice(3, max_agents, p=mode_priors)

    # Update mode and positions given mode transitions and slopes.
    for agent_id in range(max_agents):
        for t in range(max_timesteps):
            if t == 0:
                mode = mode_start[agent_id]
                positions_prev = [0.0, 0.0]
            else:
                mode = np.random.choice(3, 1, p=mode_transitions[mode])[0]
                positions_prev = positions[t - 1, agent_id]
            modes[t, agent_id] = mode
            positions[t, agent_id, 0] += positions_prev[0] + 1.0
            positions[t, agent_id, 1] += positions_prev[1] + mode_slopes[mode]
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
