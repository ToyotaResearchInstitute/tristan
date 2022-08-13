import argparse
import hashlib
import json
import os

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import tqdm
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse_map_utils import MapFeaturesUtils
from map_utils import get_all_lanes_by_radius, get_candidate_lanes
from trajectory_utils import resample_lane, smooth_trajectory
from visualize_utils import visualize_stats_hist, visualize_traj_and_map

from data_sources.snippet_writers import save_adovehicle_track
from loaders.ado_key_names import (
    CITY,
    FILEDIR,
    FILEHASH,
    HEADING_L,
    LANES,
    NEARBY_AGENTS,
    POSITION_L,
    PREDICTION_TIMESTAMP,
    SAMPLE_COUNT,
    SOURCE_CSV,
    TIMESTAMP,
    TIMESTAMPED_DATA,
    UNIQUE_ID,
)
from triceps.protobuf.snippet_pb_writer import save_adovehicle_track_protobuf


def argoverse_instance_info_handler(data_dict, timestamp):
    """
    Add instance info to a dictionary.
    Parameters
    ----------
    data_dict: input data.

    Returns
    -------
    result: instance info.

    """
    result = json.dumps(
        {"json_dir": data_dict["file_dir"], "source_tlog": data_dict["source_csv"], "timestamp": timestamp},
        sort_keys=True,
    )
    return result


def extract_argoverse_from_csvs(
    input_dir,
    output_dir,
    convert_limit,
    get_selected_lanes=False,
    past_horizon=20,
    save_protobuf=True,
    visualize=False,
    max_example_size_protobuf=100,
    mode="train",
    resample=True,
    resample_distance=1.0,
    args=None,
):
    """
    Scan csvs, extract instances for prediction
    :param input_dir: name of csv directory containing input files
    :param output_dir: name of directory to save output files
    :param convert_limit: maximum number of samples to save
    :param get_selected_lanes: whether to obtain lanes given observed trajectory.
    :param past_horizon: past horizon of observed trajectory.
    :param save_protobuf: whether to save extracted data into protobufs.
    :param visualize: whether to visualize data statistics.
    :param max_example_size_protobuf: max number of examples to save in a protobuf.
    :param mode: train/val/test mode.
    :param resample: whether to resample equidistance for each lane.
    :param resample_distance: resample distance between points.
    :param args: additional argument.
    :return:
    """
    # Load Argoverse libraries.
    afl = ArgoverseForecastingLoader(input_dir)
    avm = ArgoverseMap()
    map_features_utils_instance = MapFeaturesUtils()

    # Create output directory.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Begin to extract data")
    print("Extracting {} files".format(len(afl)))

    dicts_to_save = []
    file_prefix = "argoverse_agent"

    # Stats to collect.
    lane_counts = []
    lane_point_counts = []

    max_map_points = 0

    # Extract data for each example.
    for data_idx, data in tqdm.tqdm(enumerate(afl)):
        # Obtain target agent trajectory and other relevant info.
        agent_traj = data.agent_traj

        # Smooth agent trajectory.
        if args.smooth:
            agent_traj_smoothed = smooth_trajectory(agent_traj, mode=mode)

            # Shift the smoothed trajectory, so that the most recent position is consistent.
            delta = agent_traj[past_horizon - 1] - agent_traj_smoothed[past_horizon - 1]
            agent_traj = agent_traj_smoothed + delta

        # Fake future trajectory with 0s if only observations are provided.
        if mode == "test":
            agent_traj_future = np.ones((30, 2)) * agent_traj[-1]
            agent_traj = np.concatenate([agent_traj, agent_traj_future])

        seq_len = agent_traj.shape[0]
        file_id = os.path.split(data.current_seq)[-1].split(".")[0]

        city = data.seq_df["CITY_NAME"][0]
        # track_id_list = data.track_id_list
        track_id_list = np.unique(data.seq_df["TRACK_ID"].values).tolist()
        # First track id should AV that collects the data.
        assert track_id_list[0] == "00000000-0000-0000-0000-000000000000"
        data_value = data.seq_df.values

        t, _ = data_value.shape
        # First timestamp.
        timestamp_current = data_value[0, 0]

        # Collect trajectory of target vehicle, as well as nearby agents.
        target_tracks = []
        nearby_agent_tracks = []
        target_track = []
        nearby_agent_track = {}
        timestamps = [timestamp_current]

        # Collect target and nearby agent track information at each time step.
        timestep = 0
        for i in range(t):
            if data_value[i, 0] == timestamp_current:
                if data_value[i, 2] == "AGENT":
                    target_agent_id = data_value[i, 1]
                    target_track = [agent_traj[timestep, 0], agent_traj[timestep, 1]]
                    timestep += 1
                else:
                    agent_id = data_value[i, 1]
                    nearby_agent_track[agent_id] = [data_value[i, 3], data_value[i, 4]]
            else:
                target_tracks.append(target_track)
                nearby_agent_tracks.append(nearby_agent_track)
                timestamps.append(data_value[i, 0])
                target_track = []
                nearby_agent_track = {}

                if data_value[i, 2] == "AGENT":
                    target_track = [agent_traj[timestep, 0], agent_traj[timestep, 1]]
                    timestep += 1
                else:
                    agent_id = data_value[i, 1]
                    nearby_agent_track[agent_id] = [data_value[i, 3], data_value[i, 4]]

                timestamp_current = data_value[i, 0]

        # Append data points at the last time step.
        target_tracks.append(target_track)
        nearby_agent_tracks.append(nearby_agent_track)

        # Fake nearby agent tracks and timestamps in test mode.
        if mode == "test":
            target_tracks = agent_traj
            nearby_agent_tracks = nearby_agent_tracks + [nearby_agent_tracks[-1]] * 30
            timestamps = timestamps + [timestamps[-1] + 0.1] * 30

        assert len(target_tracks) == seq_len and len(nearby_agent_tracks) == seq_len, "track length mismatch"
        assert len(timestamps) == seq_len, "timestamp length mismatch"
        assert np.all(np.array(target_tracks) == agent_traj), "agent trajectory mismatch"

        if get_selected_lanes:
            # Get map coordinates by searching over candidate lanes that can be followed by the agent_traj.
            lanes = get_candidate_lanes(
                avm, map_features_utils_instance, agent_traj, city, seq_len, obs_len=past_horizon, mode=mode
            )
        else:
            # Otherwise, find all lanes naively through geofencing, within a certain radius.
            # For each lane, return 10 points uniformly.
            lanes = get_all_lanes_by_radius(avm, agent_traj[min(past_horizon, len(agent_traj) - 1)], city=city)

        # Resample points if necessary (default to True).
        if resample:
            resampled_lanes = []
            for ln in lanes:
                resampled_lane = resample_lane(ln["center_line"], distance=resample_distance)
                resampled_lanes.append({"center_line": resampled_lane, "id": ln["id"]})
            lanes = resampled_lanes

        # Update maximum number of points.
        num_map_points = np.sum([len(lane["center_line"]) for lane in lanes])
        if num_map_points > max_map_points:
            max_map_points = num_map_points

        # Visualize trajectory and map coordinates.
        if visualize:
            if args.smooth:
                visualize_traj_and_map(data.agent_traj, lanes, file_id, output_dir, None, agent_traj)
            else:
                visualize_traj_and_map(data.agent_traj, lanes, file_id, output_dir, None)

        # Collect stats.
        lane_counts.append(len(lanes))
        lane_point_counts.append(np.max([len(ln["center_line"]) for ln in lanes]))

        # Obtain hash id for each sequence.
        hash_object = hashlib.sha1(str(target_tracks).encode("utf-8"))
        pbHash = hash_object.hexdigest()
        # Initialize orientation matrix.
        R_prev = np.array([[1.0, 0.0], [0.0, 1.0]])

        if save_protobuf:
            output_dicts = {TIMESTAMPED_DATA: []}

        # Save data at each timestamp.
        for i in range(seq_len):
            output_dict = {}
            output_dict[FILEHASH] = pbHash
            output_dict[TIMESTAMP] = [timestamps[i]]
            output_dict[POSITION_L] = [target_tracks[i]]
            output_dict[PREDICTION_TIMESTAMP] = [timestamps[19]]
            output_dict[UNIQUE_ID] = target_agent_id.split("-")[-1]
            output_dict[SAMPLE_COUNT] = data_idx
            output_dict[CITY] = city
            output_dict[FILEDIR] = input_dir
            output_dict[SOURCE_CSV] = file_id

            nearby_agent_data = []
            for track_id in track_id_list:
                # Skip target agent since it is already included.
                if track_id == target_agent_id:
                    continue

                if track_id in nearby_agent_tracks[i]:
                    position = nearby_agent_tracks[i][track_id]
                else:
                    # In case where initial position for AV is missing,
                    # use its next position to complete the trajectory.
                    if i == 0 and track_id.split("-")[-1] == "000000000000":
                        position = nearby_agent_tracks[i + 1][track_id]
                    # Otherwise, use None to indicate invalid positions.
                    else:
                        position = [None, None]
                nearby_agent_data.append({POSITION_L: [position], UNIQUE_ID: track_id.split("-")[-1]})
            output_dict[NEARBY_AGENTS] = nearby_agent_data
            output_dict[LANES] = lanes

            # Compute heading based on consecutive positions.
            if i < seq_len - 1:
                pos0 = agent_traj[i]
                pos1 = agent_traj[i + 1]
                velocity_v = np.array(pos1) - np.array(pos0)

                if np.linalg.norm(velocity_v) < 0.001:
                    R = R_prev
                else:
                    v1 = velocity_v / np.linalg.norm(velocity_v)
                    v2 = np.array([-v1[1], v1[0]])
                    R = np.array([v1, v2])
            else:
                R = R_prev
            output_dict[HEADING_L] = [R.tolist()]
            R_prev = R

            if save_protobuf:
                output_dicts[TIMESTAMPED_DATA].append(output_dict)
            else:
                save_adovehicle_track(
                    file_prefix, output_dir, None, output_dict, unique_id=file_id, sequence_counter=i, indent=1
                )

        if save_protobuf:
            dicts_to_save.append(output_dicts)

            # Save to protobuf if we have accumulated enough examples.
            if (len(dicts_to_save) >= max_example_size_protobuf) or (data_idx == len(afl) - 1):
                save_adovehicle_track_protobuf(
                    file_prefix,
                    output_dir,
                    None,
                    dicts_to_save,
                    unique_id=file_id,
                    instance_info_handler=argoverse_instance_info_handler,
                )
                dicts_to_save = []

        if convert_limit is not None and data_idx >= convert_limit - 1:
            if visualize:
                visualize_stats_hist(lane_counts, "lane counts", output_dir)
                visualize_stats_hist(lane_point_counts, "lane point counts", output_dir)
            return

    if visualize:
        visualize_stats_hist(lane_counts, "lane counts", output_dir)
        visualize_stats_hist(lane_point_counts, "lane point counts", output_dir)

    print("Maximum number of points", max_map_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", help="Input csvs files to read")
    parser.add_argument("--output-dir", help="Output jsons files to write")
    parser.add_argument(
        "-s",
        "--get-selected-lanes",
        action="store_true",
        help="Get candidate centerlines that can be followed by the target trajectory, rather than naive geofencing.",
    )
    parser.add_argument(
        "-l", "--limit", type=int, help="If specified, the number of snippets converted are limited to the number."
    )
    parser.add_argument("-p", "--past-horizon", type=int, default=20, help="Observed trajectory length.")
    parser.add_argument("--max-example-size-protobuf", type=int, default=1, help="Maximum size of protobuf file.")
    parser.add_argument("--protobuf", action="store_true", help="Save trajectories as protobufs")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize data statistics")
    parser.add_argument("-m", "--mode", default="train", help="Train/val/test mode selection")
    parser.add_argument("-r", "--resample", action="store_true", help="Resample equidistance points along lanes.")
    parser.add_argument("--resample-dist", type=float, default=1.0, help="Resample distance.")

    parser.add_argument("--smooth", action="store_true", help="Smooth agent trajectory.")
    args = parser.parse_args()

    assert args.max_example_size_protobuf == 1, "Currently only supporting --max-example-size-protobuf 1."

    extract_argoverse_from_csvs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        convert_limit=args.limit,
        get_selected_lanes=args.get_selected_lanes,
        past_horizon=args.past_horizon,
        save_protobuf=args.protobuf,
        visualize=args.visualize,
        max_example_size_protobuf=args.max_example_size_protobuf,
        mode=args.mode,
        resample=args.resample,
        resample_distance=args.resample_dist,
        args=args,
    )


if __name__ == "__main__":
    main()
