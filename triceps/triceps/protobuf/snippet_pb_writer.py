import json
import os
from collections import OrderedDict

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp

from loaders.ado_key_names import TIMESTAMPED_DATA
from triceps.protobuf.prediction_training_pb2 import *
from triceps.protobuf.protobuf_data_names import (
    ADDITIONAL_INPUT_EGOVEHICLE,
    ADDITIONAL_INPUT_KEY_TRI_DOT_ID,
    ADDITIONAL_INPUT_RELEVANT_PEDESTRIAN,
)

STD_AGENT_INITIAL_POSITION = 0.1
STD_AGENT_VELOCITY = 0.1
STD_AGENT_MOTION = 0.005
DELTA_T = 0.1
N_agents = 4
N_timesteps = 3
N_instances = 4000
HIDDEN_DIM = 7
OUTPUT_DIM = 4
STDEV_REGULARIZATION_COEFF = 1
L2_ONLY = False


def save_adovehicle_track_protobuf(
    file_prefix, save_dir, file_counter, track_data, unique_id=None, instance_info_handler=None
):
    """
    save track data for an adovehicle instance as a protobuf.

    :param file_prefix:
    :param file_counter:
    :param track_data: track_data = a list of dictionaroes with values: (update)
    * 'unique_id': unique_id of the main agent
    * 'timestamp': current_time,
    * 'position_L': current position of the main agent
    * 'nearby_agents': nearby_agents' unique_id and position_L,
    :param unique_id: unique id of protobuf.
    :param instance_info_handler: special handler for instance info.

    :return:
    """
    prediction_instances = []
    # Iterate through each full data example.
    for track_datum in track_data:
        agent_trajectories = {}
        for i, datum_dict in enumerate(track_datum[TIMESTAMPED_DATA]):
            timestamp = datum_dict["timestamp"][0]
            if i == 0:
                timestamp0 = timestamp
                lanes_pb = []
                for lane in datum_dict["lanes"]:
                    segments = []
                    for s_i in range(len(lane["center_line"]) - 1):
                        x0 = lane["center_line"][s_i][0]
                        y0 = lane["center_line"][s_i][1]
                        x1 = lane["center_line"][s_i + 1][0]
                        y1 = lane["center_line"][s_i + 1][1]
                        segment = Segment(start=Point3D(x=x0, y=y0), end=Point3D(x=x1, y=y1))
                        segments.append(segment)
                    lane_pb = Lane(id=str(lane["id"]), center_line=segments)
                    lanes_pb.append(lane_pb)
                local_map = LocalMap(lanes=lanes_pb)

            x = datum_dict["position_L"][0][0]
            y = datum_dict["position_L"][0][1]
            dt = timestamp - timestamp0
            ggl_timestamp = Timestamp(seconds=int(dt // 1), nanos=int((dt - dt // 1) * 1e9))

            timestamped_state = TimestampedState(position=Point3D(x=x, y=y, z=0), timestamp=ggl_timestamp)
            ego_uid = datum_dict["unique_id"]
            if ego_uid not in agent_trajectories:
                agent_trajectories[ego_uid] = []
            agent_trajectories[ego_uid].append(timestamped_state)
            agt_uids = []
            for agt in datum_dict["nearby_agents"]:
                agt_uid = agt["unique_id"]
                agt_uids.append(agt_uid)
                position = agt["position_L"]
                agt_timestamped_state = TimestampedState(
                    position=Point3D(x=position[0][0], y=position[0][1], z=0), timestamp=ggl_timestamp
                )
                if agt_uid not in agent_trajectories:
                    agent_trajectories[agt_uid] = []
                agent_trajectories[agt_uid].append(agt_timestamped_state)
        agent_trajectories_pb = OrderedDict()
        for agent_index, id in enumerate(agent_trajectories.keys()):
            additional_input_dot_id = TimestampedPredictionInput(
                vector_input_type_id=ADDITIONAL_INPUT_KEY_TRI_DOT_ID, vector_input=[float(id)]
            )
            additional_inputs = [additional_input_dot_id]
            if id == ego_uid:
                additional_input_ego = TimestampedPredictionInput(vector_input_type_id=ADDITIONAL_INPUT_EGOVEHICLE)
                additional_inputs += [additional_input_ego]
            elif id == agt_uids[0]:
                additional_input = TimestampedPredictionInput(vector_input_type_id=ADDITIONAL_INPUT_RELEVANT_PEDESTRIAN)
                additional_inputs.append(additional_input)
            agent_info = json.dumps({"agent_index": agent_index})
            traj_pb = AgentTrajectory(
                agent_id=str(id),
                trajectory=agent_trajectories[id],
                additional_inputs=additional_inputs,
                additional_agent_info=agent_info,
            )
            agent_trajectories_pb[str(id)] = traj_pb
        dt_prediction = datum_dict["prediction_timestamp"][0] - timestamp0
        ggl_pred_timestamp = Timestamp(
            seconds=int(dt_prediction // 1), nanos=int((dt_prediction - dt_prediction // 1) * 1e9)
        )

        if instance_info_handler is not None:
            prediction_instance_info = instance_info_handler(datum_dict, timestamp0)
            prediction_instance = PredictionInstance(
                agent_trajectories=agent_trajectories_pb,
                map_information=local_map,
                prediction_instance_info=prediction_instance_info,
                prediction_time=ggl_pred_timestamp,
                egovehicle_id=ego_uid,
            )
        else:
            prediction_instance = PredictionInstance(
                agent_trajectories=agent_trajectories_pb,
                map_information=local_map,
                prediction_time=ggl_pred_timestamp,
                egovehicle_id=ego_uid,
            )
        prediction_instances.append(prediction_instance)
    prediction_set_info = PredictionSetInformation()
    prediction_set = PredictionSet(prediction_instances=prediction_instances, information=prediction_set_info)

    pb_filename = (
        os.path.join(save_dir, file_prefix)
        + ("" if unique_id is None else "_" + str(unique_id))
        + ("" if file_counter is None else "_" + "{0:08d}".format(file_counter))
        + ".pb"
    )
    with open(pb_filename, "wb") as pbfile:
        pbfile.write(prediction_set.SerializeToString())


def convert_predictor_output_to_protobuf(predicted_trajectories_scene, agent_ids, decoding, stats, timestep, t0):
    """Convert predictor outputs into a protobuf for saving.

    Args:
        predicted_trajectories_scene (Tensor): tensor of size Batch x Num_agents x Num_timesteps x dim_coordinates x Num_samples
        decoding (dict): Dictionary of additional values. Not saved.
        stats (dict): Dictionary of additional values. Not saved.
        timestep (float): delta_t in seconds between samples.
        t0 (float): initial timepoint for time samples.

    Returns:
        list: A list of prediction instances in protobuf format.
    """
    cpu_predicted_trajectories = predicted_trajectories_scene.detach().cpu().numpy()
    batch_size = predicted_trajectories_scene.shape[0]
    num_agents = predicted_trajectories_scene.shape[1]
    num_timesteps = predicted_trajectories_scene.shape[2]
    dim_coordinates = predicted_trajectories_scene.shape[3]
    num_samples = predicted_trajectories_scene.shape[4]
    from triceps.protobuf.prediction_training_pb2 import (
        AgentTrajectory,
        Covariance3D,
        Point3D,
        PredictionInstance,
        PredictionSet,
        TimestampedState,
        TimestampedStateUncertainty,
    )

    results = []
    for n_batch in range(batch_size):
        prediction_instances = []
        instance_weights = []

        for n_sample in range(num_samples):
            agent_trajectories = {}
            for n_agent in range(num_agents):
                agent_id = agent_ids[n_batch, n_agent].item()
                # Exclude invalid agents.
                if agent_id == 0:
                    continue
                trajectory = []
                for n_timestep in range(num_timesteps):
                    x = cpu_predicted_trajectories[n_batch, n_agent, n_timestep, 0, n_sample]
                    y = cpu_predicted_trajectories[n_batch, n_agent, n_timestep, 1, n_sample]
                    position = Point3D(x=x, y=y, z=0.0)
                    covariance = Covariance3D(xx=1.0, xy=0.0, xz=0.0, yy=1.0, yz=0.0, zz=0.0)
                    uncertainty_estimate = TimestampedStateUncertainty(
                        position_covariance=covariance, velocity_covariance=covariance
                    )
                    timepoint = np.long((n_timestep * timestep + t0) * 1e9)
                    seconds = timepoint // np.long(1e9)
                    nanos = np.long((timepoint - seconds * 1e9))
                    timestamp = google_dot_protobuf_dot_timestamp__pb2.Timestamp(seconds=seconds, nanos=nanos)
                    new_pose = TimestampedState(
                        timestamp=timestamp, position=position, uncertainty_estimate=uncertainty_estimate
                    )
                    trajectory.append(new_pose)
                agent_trajectories[str(int(agent_id))] = AgentTrajectory(agent_id=str(n_agent), trajectory=trajectory)
            prediction_instances.append(
                PredictionInstance(
                    agent_trajectories=agent_trajectories,
                    prediction_time=google_dot_protobuf_dot_timestamp__pb2.Timestamp(seconds=0, nanos=0),
                )
            )
            instance_weights.append(1.0)
        results.append(PredictionSet(prediction_instances=prediction_instances, instance_weights=instance_weights))
    return results
