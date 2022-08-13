# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example loading code, synthetic example generation for prediction protobuf
import json

import numpy as np
import torch
import tqdm
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp
from torch.utils.data import DataLoader

# pylint: disable=import-error
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset

# pylint: disable=no-member
# pylint: disable=import-error
from triceps.protobuf.prediction_training_pb2 import (
    AgentTrajectory,
    Point3D,
    PredictionInstance,
    PredictionSet,
    TimestampedState,
)
from triceps.protobuf.protobuf_training_parameter_names import (
    PARAM_FUTURE_TIMESTEP_SIZE,
    PARAM_FUTURE_TIMESTEPS,
    PARAM_MAX_AGENTS,
    PARAM_PAST_TIMESTEP_SIZE,
    PARAM_PAST_TIMESTEPS,
)

# Parameters for generating a distribution of the agents' motion
STD_AGENT_INITIAL_POSITION = 0.1
STD_AGENT_VELOCITY = 0.1
STD_AGENT_MOTION = 0.005

# Time step for generated motion samples.
DELTA_T = 0.1

# Number of agents.
N_agents = 4

# Number of time steps.
N_timesteps = 10

# Number of generated prediction instances.
N_instances = 2000

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def generate_prediction_protobuf():
    instance_id = 0
    prediction_instances = []
    for _ in tqdm.tqdm(range(N_instances), desc="generating"):
        time_counter = 0.0
        agent_positions = {}
        agent_trajectories = {}
        agent_metadata = {}
        for agent_i in range(N_agents):
            agent_metadata[agent_i] = {"velocity": np.random.normal(0.0, 1.0, [2, 1]) * STD_AGENT_VELOCITY}
            agent_positions[agent_i] = np.random.normal(0.0, 1.0, [2, 1]) * STD_AGENT_INITIAL_POSITION
            agent_trajectories[agent_i] = []
        for time_t in range(N_timesteps):
            for agent_i in range(N_agents):
                agent_positions[agent_i] += (
                    np.random.normal(0.0, 1, [2, 1]) * STD_AGENT_MOTION + agent_metadata[agent_i]["velocity"]
                )
                position = Point3D(x=agent_positions[agent_i][0], y=agent_positions[agent_i][1])
                new_position = TimestampedState(
                    timestamp=Timestamp(
                        seconds=int(time_counter // 1), nanos=int((time_counter - time_counter // 1) * 1e9)
                    ),
                    position=position,
                )
                agent_trajectories[agent_i].append(new_position)
            time_counter = DELTA_T * time_t
        agent_trajectories_pb = {}
        for agent_i in range(N_agents):
            agent_trajectories_pb[agent_i] = AgentTrajectory(
                agent_id=str(agent_i), trajectory=agent_trajectories[agent_i]
            )
        prediction_instance = PredictionInstance(instance_id=str(instance_id))
        for k in agent_trajectories_pb:
            prediction_instance.agent_trajectories[str(k)].agent_id = agent_trajectories_pb[k].agent_id
            for x in agent_trajectories_pb[k].trajectory:
                prediction_instance.agent_trajectories[str(k)].trajectory.append(x)
            for x in agent_trajectories_pb[k].additional_inputs:
                prediction_instance.agent_trajectories[str(k)].additional_inputs.append(x)

        prediction_instances.append(prediction_instance)
        instance_id += 1
    prediction_set_result = PredictionSet(prediction_instances=prediction_instances)
    return prediction_set_result


if __name__ == "__main__":
    print("Generating data.")
    prediction_set = generate_prediction_protobuf()
    json_filename = "prediction_instance.json"
    pb_filename = "prediction_instance.pb"
    print("Saving data.")
    with open(json_filename, "w") as jsfile:
        json.dump(MessageToDict(prediction_set), jsfile, indent=2)
    with open(pb_filename, "wb") as pbfile:
        pbfile.write(prediction_set.SerializeToString())

    params = {}

    # These can be taken from loader.ado_key_names in RAD repo. Removed for self-sufficiency.
    # [
    #         AGENT_TYPE_CAR, AGENT_TYPE_BICYCLE, AGENT_TYPE_MOTORCYCLE, AGENT_TYPE_PEDESTRIAN, AGENT_TYPE_LARGEVEHICLE,
    #         AGENT_TYPE_TRUCK
    # ]

    params["agent_types"] = [2, 3, 4, 5, 6, 7]
    params[PARAM_MAX_AGENTS] = 4
    params[PARAM_PAST_TIMESTEPS] = 10
    params[PARAM_FUTURE_TIMESTEPS] = 10
    params[PARAM_PAST_TIMESTEP_SIZE] = 0.1
    params[PARAM_FUTURE_TIMESTEP_SIZE] = 0.1

    dataset = ProtobufPredictionDataset(pb_filename, params=params)
    print("Done.")
    for itm in tqdm.tqdm(dataset, desc="prediction_instances"):
        continue
    print("Done dataset read.")

    # TODO(guy.rosman): Migrate an example with images.
    tqdm_iter = tqdm.trange(1000, desc="iter")
    for iteration in tqdm_iter:
        dataloader = DataLoader(dataset, batch_size=64, num_workers=64)
        for batch_itm in tqdm.tqdm(dataloader, desc="prediction_instances"):

            batch_size = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape[0]
            timestamps = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape[1]
            num_agents = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape[2]
            num_past_timepoints = timestamps - 1
            assert num_agents == N_agents
            assert timestamps == N_timesteps

    print("Done dataloader read.")
