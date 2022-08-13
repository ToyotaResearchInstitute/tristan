import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from data_sources.snippet_writers import save_adovehicle_track

# usage:
# 1. Download argoverse dataset from https://www.argoverse.org/data.html#download-link
# 2. Install argoverse api from https://github.com/argoai/argoverse-api#installation
# 3. run python create_adovehicle_maneuver_labels_argoverse.py --input_dir [ARGOVERSE_DATA_DIR] --output_dir [OUTPUT_LABEL_JSON_DIR] -v -p


def smooth_index_label(lane_indices):
    """
    Smooth a list of indices, such that there is no subsequences that start and end with the same index, but has
    different indices in between.
    This is used to provide cleaner labels when an agent follows a lane but crosses other lanes.
    Parameters
    ----------
    lane_indices: input indices.

    Returns
    -------

    """
    lane_indices = copy.deepcopy(lane_indices)
    start_index = lane_indices[0]
    for i in range(1, len(lane_indices)):
        if lane_indices[i] == start_index:
            start_index = lane_indices[i]
        else:
            noise = False
            for j in range(i + 1, len(lane_indices)):
                if lane_indices[j] == start_index:
                    noise = True
                    for k in range(i, j):
                        lane_indices[k] = start_index
                    i = j
                    break
            if not noise:
                start_index = lane_indices[i]
    return lane_indices


def label_lane_sequence(agent_traj, lanes):
    """
    Label index of closest lane from a given trajectory.

    Args:
        agent_traj: agent trajectory.
        lanes: lane coordinates.

    Returns:
        lane_indices:
        lane_indices_smoothed: smoothed indices (see details in docstring in smooth_index_label).

    """
    traj_length = agent_traj.shape[0]
    lane_indices = []

    for i in range(traj_length):
        agent_pos = agent_traj[i]
        min_dists_to_agent_pos = []
        for j in range(len(lanes)):
            lane_pos = np.array(lanes[j]["center_line"])
            min_dist = np.min(np.sum((lane_pos - agent_pos) ** 2, -1))
            min_dists_to_agent_pos.append(min_dist)
        lane_indices.append(np.argmin(min_dists_to_agent_pos))

    lane_indices_smoothed = smooth_index_label(lane_indices)
    return lane_indices, lane_indices_smoothed


def label_lane_change_sequence(agent_traj, lanes):
    """
    Label maneuver sequence from a given trajectory.

    Args:
        agent_traj: agent trajectory.
        city: city where the data was collected.
        avm: Argoverse map library.

    Returns:
        Labelled maneuver sequence.

    """
    # Add import inside the function so that it will not be called during CI test.
    from argoverse.utils.centerline_utils import get_normal_and_tangential_distance_point

    traj_length = agent_traj.shape[0]

    # Check for lane change maneuvers based on distance to centerlines.
    # First obtain all candidate centerlines given the agent trajectory.
    candidate_centerlines = [lane["center_line"] for lane in lanes]

    # If there is only one lane, return lane keep (LK) at all time steps.
    if len(candidate_centerlines) == 1:
        return ["LK"] * traj_length

    # Compute closest lane to the first and last positions.
    i_start, d_start = -1, 100000
    i_end, d_end = -1, 100000
    for i, lane in enumerate(candidate_centerlines):
        _, d_s = get_normal_and_tangential_distance_point(agent_traj[0][0], agent_traj[0][1], lane)
        d_s = abs(d_s)
        if d_s < d_start:
            i_start = i
            d_start = d_s

        _, d_e = get_normal_and_tangential_distance_point(agent_traj[-1][0], agent_traj[-1][1], lane)
        d_e = abs(d_e)
        if d_e < d_end:
            i_end = i
            d_end = d_e

    # If both start and end have the same closest lane, maneuver is lane keep.
    if i_start == i_end:
        return ["LK"] * traj_length

    # Otherwise, iterate each time step to find maneuvers.
    maneuvers = []
    i_prev = -1
    d_prev = 100000
    for t in range(len(agent_traj)):
        pos_curr = agent_traj[t]

        # Compute the distance to the closest centerline from current position.
        i_curr, d_curr = -1, 100000
        for i, lane in enumerate(candidate_centerlines):
            _, dist = get_normal_and_tangential_distance_point(pos_curr[0], pos_curr[1], lane)
            dist = abs(dist)
            if dist < d_curr:
                i_curr = i
                d_curr = dist

        if t == 0:
            # Assume the first maneuver is always lane keep.
            maneuvers.append("LK")
        elif i_curr == i_prev:
            # Lane keep if closest lane does not change.
            maneuvers.append("LK")
        else:
            # Compute signed normal distance of current position to the lane
            # closest to the previous positions, and decide lane change
            # direction based on the difference in distances.
            _, dist_curr_to_prev = get_normal_and_tangential_distance_point(
                pos_curr[0], pos_curr[1], candidate_centerlines[i_prev]
            )
            if dist_curr_to_prev > d_prev:
                maneuvers.append("LL")
            else:
                maneuvers.append("LR")

        i_prev = i_curr
        d_prev = d_curr
    return maneuvers


def label_driving_mode_sequence(agent_traj, turn_threshold=2.0, vel_threshold=1.0, mode="train"):
    """
    Label mode sequence from a given trajectory.

    Parameters
    ----------
    agent_traj : np.ndarray
        Input trajectory.
    turn_threshold : float
        Turning angle threshold.
    vel_threshold : float
        Velocity threshold.
    mode : str
        Data mode, with options "train" and "test".
        In test mode, we ignore the last mode due to noise in Argoverse data.

    Returns
    -------
    driving_modes : list
        A list of driving modes.
    """
    traj_length = agent_traj.shape[0]
    driving_modes = []

    # Compute heading.
    theta = np.arctan2(agent_traj[1:, 1] - agent_traj[:-1, 1], agent_traj[1:, 0] - agent_traj[:-1, 0])
    theta = np.hstack((theta[:1], theta))
    theta = np.degrees(theta)

    # Compute angular velocity.
    theta_diff = theta[1:] - theta[:-1]
    theta_diff = np.hstack((theta_diff, theta_diff[-1:]))
    theta_diff[theta_diff < -180] += 360
    theta_diff[theta_diff > 180] -= 360

    velocity = np.sqrt(np.sum((agent_traj[1:] - agent_traj[:-1]) ** 2, -1))
    velocity = np.hstack((velocity[:1], velocity))
    acceleration = velocity[1:] - velocity[:-1]
    acceleration = np.hstack((acceleration, acceleration[-1:]))
    for i in range(traj_length):
        theta_diff_i = theta_diff[i]

        if theta_diff_i >= 1.0 * turn_threshold:
            driving_modes.append("LT")
        elif theta_diff_i <= -1.0 * turn_threshold:
            driving_modes.append("RT")
        else:
            if velocity[i] >= vel_threshold * 2:
                driving_modes.append("FF")
            elif velocity[i] >= vel_threshold:
                driving_modes.append("NF")
            elif velocity[i] >= 0.05:
                driving_modes.append("SF")
            else:
                driving_modes.append("ST")

    # Use the last but second mode for test, since the last mode is not always stable.
    # This is for Argoverse dataset only.
    if mode == "test":
        driving_modes[19] = driving_modes[18]
    return driving_modes


def extract_argoverse_labels_from_csvs(input_dir, output_dir, gather_statistics=False, save_plots=False):
    """
    Extract maneuver labels from csv data.
    Args:
        input_dir: Name of csv directory containing input files.
        output_dir: Name of output directory.
        gather_statistics:  Print maneuver sequence stats.
        save_plots: Save figures of trajectories and maneuvers.

    Returns:

    """
    # Add imports inside the function so that they will not be called during CI test.
    from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
    from argoverse.map_representation.map_api import ArgoverseMap
    from argoverse.utils.mpl_plotting_utils import visualize_centerline

    afl = ArgoverseForecastingLoader(input_dir)
    avm = ArgoverseMap()

    # create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Begin to extract data")
    print("Extracting {} files".format(len(afl)))

    file_prefix = "argoverse_agent_maneuver_labels"
    maneuver_data = []
    for data in tqdm.tqdm(afl):
        agent_traj = data.agent_traj
        city = data.city
        agent_id = os.path.split(data.current_seq)[-1].split(".")[0]

        # Compute maneuver sequences.
        maneuver_sequence = label_maneuver_sequence(agent_traj, city, avm)
        maneuver_data.append({"agent_id": agent_id, "maneuver_seq": maneuver_sequence})

    # save maneuver types
    save_adovehicle_track(file_prefix, output_dir, None, maneuver_data, indent=1)

    if gather_statistics:
        maneuvers = ["TL", "TR", "TU", "LK", "LCL", "LCR", "U"]
        maneuver_counts = {m: 0 for m in maneuvers}
        for m_d in maneuver_data:
            maneuver_sequence = m_d["maneuver_seq"]
            for m in maneuver_sequence:
                maneuver_counts[m] += 1
        print(maneuver_counts)

    if save_plots:
        print("Saving plots")
        # create output directory for figures
        figure_dir = os.path.join(output_dir, "figures")
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)

        for i, data in enumerate(tqdm.tqdm(afl, desc="examples")):
            agent_traj = data.agent_traj
            city = data.city
            candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_traj, city)
            agent_id = maneuver_data[i]["agent_id"]
            maneuver_sequence = maneuver_data[i]["maneuver_seq"]
            if "LCL" in maneuver_sequence:
                maneuver_sequence_txt = "LCL"
            elif "LCR" in maneuver_sequence:
                maneuver_sequence_txt = "LCR"
            else:
                maneuver_sequence_txt = "LK"

            plt.clf()
            fig = plt.figure()
            color_map = {"LCL": "b", "LCR": "g", "LK": "r", "U": "k", "TL": "k", "TR": "k", "TU": "k"}
            colors = [color_map[m] for m in maneuver_sequence]
            plt.scatter(agent_traj[:, 0], agent_traj[:, 1], c=colors, alpha=1, s=1)
            for centerline_coords in candidate_centerlines:
                visualize_centerline(centerline_coords)
            plt.title("File id - {}, maneuver sequence - {}".format(agent_id, maneuver_sequence_txt))
            fig.savefig(figure_dir + "/{}.png".format(agent_id))
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", help="Input csvs files to read")
    parser.add_argument("--output_dir", help="Output jsons files to write")
    parser.add_argument("-v", "--gather_statistics", action="store_true", help="Gather maneuver statistics")
    parser.add_argument(
        "-p", "--save_plots", action="store_true", help="Save trajectory plots with lanes and maneuver labels"
    )

    args = parser.parse_args()

    extract_argoverse_labels_from_csvs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gather_statistics=args.gather_statistics,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
