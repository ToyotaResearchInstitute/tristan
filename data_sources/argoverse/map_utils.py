import numpy as np

# Map Feature computations
_MANHATTAN_THRESHOLD = 10.0  # meters, used to find nearby lanes.
_DFS_THRESHOLD_FRONT_SCALE = 25.0  # meters
_DFS_THRESHOLD_BACK_SCALE = 2.0  # meters
_MAX_SEARCH_RADIUS_CENTERLINES = 50.0  # meters
_MAX_CENTERLINE_CANDIDATES_TEST = 15


def get_all_lanes_by_radius(avm, ego_pose, city):
    """
    Get the Lane Geometry in a box around the Ego pose.
    :param avm: ArgoverseMap class
    :param ego_pose: global position (x,y) of the target vehicle
    :param city: city where data was collected
    """
    lanes = []
    lane_ids = avm.get_lane_ids_in_xy_bbox(
        query_x=ego_pose[0],
        query_y=ego_pose[1],
        city_name=city,
        query_search_range_manhattan=_MAX_SEARCH_RADIUS_CENTERLINES,
    )
    for id in lane_ids:
        lane_centerline = avm.get_lane_segment_centerline(id, city)[:, :2].tolist()
        lane = {"center_line": lane_centerline, "id": id}
        lanes.append(lane)

    return lanes


def get_candidate_lanes(avm, map_features_utils_instance, agent_traj, city, seq_len, obs_len=20, mode="test"):
    """
    Get candidate centerlines that can be followed by the target trajectory using Argoverse API.
    :param avm: ArgoverseMap class
    :param map_features_utils_instance: MapFeaturesUtils class
    :param agent_traj: trajectory of the target vehicle
    :param city: city where data was collected
    :param seq_len: length of entire trajectory sequence,
    :param obs_len: length of observed trajectory sequence,
    :param mode: train/val/test modes.
    """
    if mode == "test":
        agent_traj = agent_traj[:obs_len]

    # Get candidate centerlines given a trajectory.
    candidate_centerlines = map_features_utils_instance.get_candidate_centerlines_for_trajectory(
        agent_traj,
        city,
        avm,
        viz=False,
        max_search_radius=_MAX_SEARCH_RADIUS_CENTERLINES,
        seq_len=seq_len,
        max_candidates=_MAX_CENTERLINE_CANDIDATES_TEST,
        mahattan_threshold=_MANHATTAN_THRESHOLD,
        dfs_front_scale=_DFS_THRESHOLD_FRONT_SCALE,
        dfs_back_scale=_DFS_THRESHOLD_BACK_SCALE,
        mode=mode,
    )

    # Save to a dictionary.
    lanes = []
    for id, lane_centerline in enumerate(candidate_centerlines):
        lane = {"center_line": lane_centerline.tolist(), "id": id}
        lanes.append(lane)

    return lanes


def get_lanes_by_traj(avm, df):
    """
    Get centerlines that lies within the range of trajectories.

    Parameters
    ----------
    avm: ArgoverseMap
        Argoverse map class to get map information.
    df: DataFrame
        Trajectory sequence.

    Returns
    -------
    list
        List of lanes with their IDs.
    """
    city_name = df["CITY_NAME"].values[0]
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]
    x_min = min(df["X"])
    x_max = max(df["X"])
    y_min = min(df["Y"])
    y_max = max(df["Y"])

    lanes = []
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline
        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lanes.append((lane_id, lane_props))
    return lanes
