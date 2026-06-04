import json
import math
from collections import OrderedDict, defaultdict
from typing import Any, Tuple

import numpy as np
from scipy import ndimage, stats
from scipy.interpolate import interp1d
from shapely.geometry import LinearRing, LineString, Point

import intent.IntentOSM as intent_osm

osm = intent_osm.IntentOSM()

try:
    osm.load_all()
except Exception as e:
    osm = None
    print("No OSM cache:\n{}".format(repr(e)))


class TokenEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.number):
            return o.item()
        else:
            return json.JSONEncoder.default(self, o)


def dist_sqr(pos1, pos2):
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]

    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def is_condition_log_id(res, past_horizon=23, future_horizon=23, tlog_hash=None, time_range=None):
    time_range = None
    result = True
    if not (tlog_hash is None):
        result = res["filehash"] == tlog_hash
        # if (result):

    if not (time_range is None):
        timestamp = res["timestamp"][len(res["past_positions"])]
        result = result and timestamp > time_range[0] and timestamp <= time_range[1]
    return result


def is_condition_tested_worst_cases(obj, statistics_json, slice_names, group_type="nll_highest"):
    """
    Compares the json object's fname and timestamp to statistics_json, getting the worst examples according to
    group type, for the slice names (e.g, nll_highest for all slices)
    TODO: replace with a more general comparison to match adovehicles or other data instances
    :param obj:
    :param statistics_json:
    :param slice_names:
    :param group_type:
    :return:
    """
    result = False
    # TODO: check condition logic for searching over json
    assert "slice_results" in statistics_json
    slices = statistics_json["slice_results"]
    if slice_names is None or not slice_names:
        slice_names = list(slices.keys())

    assert "fname_meta" in obj
    assert "timestamp" in obj
    fname = obj["fname_meta"]
    timestamp = obj["timestamp"]
    for slice_name in slice_names:
        slice_result = slices[slice_name]

        if group_type in slice_result:

            group = slice_result[group_type]
            for itm in group[1]:
                if itm["fname"] == fname and itm["timestamp"] == timestamp:
                    return True

    return result


### Functors to identify driving conditions
def is_condition_stop_f(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the future
    future_traj = np.array(res["future_positions"])
    future_traj = future_traj[:future_horizon]
    dist_future_sqr = dist_sqr(future_traj[0, :], future_traj[-1, :])

    return dist_future_sqr < 4.0


def is_condition_stop_p(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the past
    past_traj = np.array(res["past_positions"])
    past_traj = past_traj[-past_horizon:]
    dist_past_sqr = dist_sqr(past_traj[0, :], past_traj[-1, :])

    return dist_past_sqr < 4.0


def is_condition_stop_m(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the middle of a motion
    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]

    future_traj_sqr_diff = [dist_sqr(future_traj[i], future_traj[i + 1]) for i in np.arange(future_horizon - 1)]
    # print("future_traj_sqr_diff")
    # print(future_traj_sqr_diff)
    # TODO: Won't get _quite_ to zero
    return np.count_nonzero(future_traj_sqr_diff) < future_horizon * 4 / 5


def is_condition_stop(res, past_horizon=23, future_horizon=23):
    # Check for complete and partial stops
    return (
        is_condition_stop_f(res, past_horizon, future_horizon)
        or is_condition_stop_p(res, past_horizon, future_horizon)
        or is_condition_stop_m(res, past_horizon, future_horizon)
    )


def create_maneuver_condition(maneuver_name, adovehicle_data_dir):
    def __is_good_maneuver(res):
        from attic.data.adovehicle_database import AdovehicleDatabase

        adovehicle_db = AdovehicleDatabase(adovehicle_data_dir)
        try:
            from attic.adovehicle import imerit

            annotation = adovehicle_db.get_imerit_annotation(res["uuid"])
            return imerit.is_good_maneuver(annotation, maneuver_name)
        except KeyError:
            import traceback

            print("Annotation is not found.")
            traceback.print_exc()
            return False

    return __is_good_maneuver


def is_condition_nonstop(res, past_horizon=23, future_horizon=23):
    # Check for movements
    return not is_condition_stop(res, past_horizon, future_horizon)


def is_condition_forward(res, past_horizon=23, future_horizon=23):
    # Check for nonbackward movements

    # Make sure the vehicle moves
    # if is_condition_stop_f(res, past_horizon=23, future_horizon=23): return False
    if is_condition_stop_m(res, past_horizon=23, future_horizon=23):
        return False

    # check if vehicle moves forwards
    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]
    past_traj = np.transpose(res["past_positions"])
    past_traj = past_traj[-past_horizon:]

    x_last = future_traj[-1][0]
    y_last = future_traj[-1][1]
    # Why 5?
    # x_first = past_traj[-5][0]
    # y_first = past_traj[-5][1]
    # Change to last element, for now
    x_first = past_traj[-1][0]
    y_first = past_traj[-1][1]

    # make sure vehicle does not go too fast
    if x_last > 400:
        return False

    # make sure the vehicle goes forward in x for at least 0.1 meters
    # and moves for at least 0.5 meters
    return x_first <= 0 and x_last > 0.1 and x_last**2 + y_last**2 > 0.25


def is_condition_forward_and_slow(res, past_horizon=23, future_horizon=23):
    # check if velocity is reasonbale
    if not is_condition_forward(res, past_horizon=23, future_horizon=23):
        return False

    vel = res["velocity"]
    vel_norm = np.linalg.norm(vel)

    return vel_norm <= 10.0


def is_condition_straight(res, past_horizon=23, future_horizon=23):
    # Check for straight movements

    # Make sure the vehicle moves forward
    if not is_condition_forward(res, past_horizon=23, future_horizon=23):
        return False

    # Check if the future trajectory is straight
    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]

    # check linearity of the trajectory
    slope, intercept, r_value, p_value, std_err = stats.linregress(future_traj[:, 0], future_traj[:, 1])
    return np.abs(r_value**2) > 0.99


def is_condition_city(res, past_horizon=23, future_horizon=23):
    # Check for city samples

    # Make sure the vehicle moves forward
    if not is_condition_forward(res, past_horizon=23, future_horizon=23):
        return False

    non_city_filehashes = ["c04cc49555b774a93bfeb487228bb7f81418b5f6"]
    return not res["filehash"] in non_city_filehashes


def is_condition_at_intersection(res, past_horizon=23, future_horizon=23):
    # Check if the sample is close enough to the intersection

    # Make sure the data is collected in city
    if not is_condition_city(res, past_horizon=23, future_horizon=23):
        return False

    latlon = res["position_lla"]
    # middle = len(latlon) // 2
    # latlon = latlon[middle-1]
    dist_now = osm.dist_to_intersection(latlon)

    return np.abs(dist_now) <= 5

    # latlon_future = latlon[middle+5]
    # dist_future = osm.dist_to_intersection(latlon_future)

    # return dist_now > dist_future


def is_condition_turning(res, past_horizon=23, future_horizon=23):
    # Check if the vehicle is turning left or right

    # Make sure the data is collected in city
    if not is_condition_city(res, past_horizon=23, future_horizon=23):
        return False
    if not is_condition_at_intersection(res, past_horizon=23, future_horizon=23):
        return False

    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]

    # check linearity of the trajectory
    slope, intercept, r_value, p_value, std_err = stats.linregress(future_traj[:, 0], future_traj[:, 1])

    y_last = future_traj[-1][1]
    x_last = future_traj[-1][0]
    return np.abs(y_last) > 3.0 and np.abs(x_last) < 30


def is_condition_stop_and_turn(res, past_horizon=23, future_horizon=23):
    # Check if vehicle stops in the past and turns in the future

    # Make sure the data is collected in city
    # if not is_condition_city(res, past_horizon=23, future_horizon=23):
    #     return False
    # if not is_condition_at_intersection(res, past_horizon=23, future_horizon=23):
    #     return False
    non_city_filehashes = ["c04cc49555b774a93bfeb487228bb7f81418b5f6"]
    if "filehash" in res and res["filehash"] in non_city_filehashes:
        return False

    future_traj = np.transpose(res["future_positions"])
    future_traj = future_traj[:future_horizon]
    past_traj = np.transpose(res["past_positions"])
    past_traj = past_traj[-past_horizon:]

    x_last = future_traj[-1][0]
    y_last = future_traj[-1][1]
    x_first = past_traj[-5][0]
    y_first = past_traj[-5][1]

    return x_first <= 0 and x_first**2 + y_first**2 <= 0.01 and x_last > 0 and x_last**2 + y_last**2 > 0.25


def is_condition_all(res, past_horizon=23, future_horizon=23):
    # Dummy condition checker always returns true
    return True


def predicate_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filters the given agent trajectories

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t). The first value of the dictionary
        is the main car for the filter. Some filters may require more than one car as they are binary relations.
        For now, we assume fixed time-step samples, and the answer relates to the center of the timeline
        (e.g. is the car stopping at the central timepoint?)

    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
        map_elements: dict,
            A map from element id to a dictionary with fields:
              'type':string,
              'positions':np.array Mx2 of points for that map element. currently just center lines.

    params: dict
        The parameter dictionary for the filter,

    Returns
    -------
    out: list
        A list of dictionaries indicating when the predicate is true and the interacting agents.
    """
    return None


def create_duration_label(
    name: str, start_time: float, end_time: float, subject: str = None, value: float = None, **kwargs
):
    """Create the output dictionary given the label and duration for the label.

    Parameters
    ----------
    name: str
        The name of the label being true.
    start_time: float
        The start timestamp of the label.
    end_time: flaot
        The end time stamp of the label.
    subject: str
        The subject of the label, e.g. the agent being yield to or the agent being followed.

    Returns
    -------
    out: dict
        The output dictionary for a filter.
    """
    out = {"label": name, "start_time": start_time, "end_time": end_time, "value": value, **kwargs}
    if subject is not None:
        out["subject"] = subject
    return out


def turn_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the given agent is turning left or right.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t).
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        turn_threshold: float
            The threshold to determine if it is a turn.
        window_size: int
            The number of time steps to consider before and after the current time step to estimate the turns.
        window_std: float
            A point is too noisy and we don't estimate the turn if the standard deviation of angular changes in a window
            is larger than this number.

    Returns
    -------
    tokens: list
        A list of dictionaries that include label and timestamps indicating the duration for the turns.
    """
    turn_threshold = params["turn_threshold"]
    window_size = params["window_size"]
    window_std = params["window_std"]
    agent_traj = list(agent_trajectories.values())[0]
    # Compute heading.
    theta = np.arctan2(agent_traj[1:, 1] - agent_traj[:-1, 1], agent_traj[1:, 0] - agent_traj[:-1, 0])
    theta = np.hstack((theta[:1], theta))
    theta = np.degrees(theta)
    # Compute angular velocity.
    theta_diff = theta[1:] - theta[:-1]
    theta_diff = np.hstack((theta_diff, theta_diff[-1:]))
    theta_diff[theta_diff < -180] += 360
    theta_diff[theta_diff > 180] -= 360
    # Compute turns for all time steps.
    tokens = []
    current_token = None
    timestamps = []
    for t in range(window_size, theta_diff.shape[0] - window_size, 3):
        t_start = t - window_size if t - window_size > 0 else 0
        t_end = t + window_size if t + window_size < theta.shape[0] else theta.shape[0]
        theta_diff_before = theta_diff[t_start:t]
        theta_diff_after = theta_diff[t:t_end]
        if theta_diff_before.shape[0] < window_size or theta_diff_after.shape[0] < window_size:
            continue
        mean_before = theta_diff_before.mean()
        std_bfore = theta_diff_before.std()
        mean_after = theta_diff_after.mean()
        std_after = theta_diff_after.std()
        if std_bfore > window_std or std_after > window_std:
            # Uncertain if the angular velocity changes too much.
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                current_token = None
                timestamps = []
        elif mean_before >= 1.0 * turn_threshold and mean_after >= 1.0 * turn_threshold:
            # Turning left if keep moving toward left.
            if current_token != "TurnLeft" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "TurnLeft"
            timestamps.append(agent_traj[t][2].item())
        elif mean_before <= -1.0 * turn_threshold and mean_after <= -1.0 * turn_threshold:
            # Turning left if keep moving toward right.
            if current_token != "TurnRight" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "TurnRight"

            timestamps.append(agent_traj[t][2].item())
        else:
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                current_token = None
                timestamps = []
    if current_token is not None:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    return tokens


def turn_filter2(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the given agent is turning left or right.


    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t).
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        turn_direction: int
            1 for left turn, -1 for right turn
        turn_deg_low: float
            The lower bound of the turn degree accumulated over certain amount of time.
            At least one windows need to have more degree to activate the filter.
        window_size_low: int
            The window size for the degree lower bound, the amount of time is window_size_low*time_step_size
        turn_deg_high: float
            The higher bound of the turn degree accumulated over certain amount of time.
            All of the windows have smaller degree to activate the filter.
        window_size_high: int
            The window size for the degree higher bound, the amount of time is window_size_high*time_step_size

    Returns
    -------
    tokens: list
        A list of dictionaries that include label and timestamps indicating the duration for the turns.
    """
    time_step_size = params["time_step_size"]
    window_size_low = params["window_size_low"]
    window_size_high = params["window_size_high"]
    turn_deg_low = params["turn_deg_low"]
    turn_deg_high = params["turn_deg_high"]
    turn_direction = params["turn_direction"]  # 1 for left turn, -1 for right turn
    # TODO(xiongyi.cui) Pass in time step size rather than assuming 10Hz.
    # Smaller movement in a second will be discarded when computing heading velocity.
    # This is much lower than normal walking speed, which is about 1.4 m/s.
    tiny_movement_threshold_meter = 0.35
    # Max angular velocity for vehicles, per second.
    max_angular_velocity_deg = 180

    tiny_movement_per_timestamp = tiny_movement_threshold_meter * time_step_size
    max_angular_velocity_deg_per_timestamp = max_angular_velocity_deg * time_step_size

    agent_traj = list(agent_trajectories.values())[0]
    move_dist = (agent_traj[1:, 1] - agent_traj[:-1, 1]) ** 2 + (agent_traj[1:, 0] - agent_traj[:-1, 0]) ** 2
    small_move_idx = move_dist < tiny_movement_per_timestamp**2
    # Compute heading.
    theta = np.arctan2(agent_traj[1:, 1] - agent_traj[:-1, 1], agent_traj[1:, 0] - agent_traj[:-1, 0])
    theta = np.degrees(theta)

    # Compute angular velocity.
    theta_diff = theta[1:] - theta[:-1]
    theta_diff[theta_diff < -180] += 360
    theta_diff[theta_diff > 180] -= 360
    # Reset theta diff when the movement is tiny
    theta_diff[small_move_idx[:-1]] = 0
    # avoid jitter
    theta_diff = np.clip(theta_diff, -max_angular_velocity_deg_per_timestamp, max_angular_velocity_deg_per_timestamp)
    # Compute heading
    theta_accu = np.add.accumulate(theta_diff)

    theta_turn_low = theta_accu[window_size_low:] - theta_accu[:-window_size_low]
    theta_turn_high = theta_accu[window_size_high:] - theta_accu[:-window_size_high]

    # agent must turn enough.
    idx_low = turn_direction * theta_turn_low > turn_deg_low
    turn_enough_count = np.sum(idx_low)
    # agent can't overturn.
    idx_high = turn_direction * theta_turn_high > turn_deg_high
    turn_too_much_count = np.sum(idx_high)
    tokens = []

    if turn_too_much_count == 0 and turn_enough_count > 0:
        current_token = "Turn"
        first_timestamp = int(np.where(idx_low == True)[0][0])
        tokens.append(create_duration_label(current_token, first_timestamp, first_timestamp + window_size_low))
    return tokens


def velocity_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the given agent is moving fast/slow or stopping.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        vel_threshold: tuple
            The thresholds (fast_min, slow_min) to determine if the agent moves fast/slow.

    Returns
    -------
    tokens: list
        A list of dictionaries that include label and timestamps indicating the duration for velocity mode.
    """
    vel_threshold = params["vel_threshold"]
    agent_traj = list(agent_trajectories.values())[0]
    # Compute velocity.
    velocity = np.sqrt(np.sum((agent_traj[1:] - agent_traj[:-1]) ** 2, -1))
    velocity = np.hstack((velocity[:1], velocity))
    tokens = []
    current_token = None
    timestamps = []
    for t in range(velocity.shape[0]):
        if velocity[t] >= vel_threshold[0]:
            if current_token != "MoveFast" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "MoveFast"
        elif velocity[t] >= vel_threshold[1]:
            if current_token != "MoveSlow" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "MoveSlow"
        elif velocity[t] < vel_threshold[1]:
            if current_token != "Stop" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "Stop"
        timestamps.append(agent_traj[t][2].item())
    if current_token is not None:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    return tokens


def acceleration_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the given agent speeds up or slows down.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        acc_threshold: float
            The threshold to determine if the agent speeds up/down.
        window_size: int
            The number of time steps to consider before and after the current time step to estimate the acceleration.
        window_std: float
            A point is too noisy and we don't estimate the acceleration if the standard deviation of velocity changes in a
            window is larger than this number.

    Returns
    -------
    tokens: list
        A list of dictionaries that include label and timestamps indicating the duration for acceleration mode.
    """
    acc_threshold = params["acc_threshold"]
    window_size = params["window_size"]
    window_std = params["window_std"]
    agent_traj = list(agent_trajectories.values())[0]
    # Compute velocity.
    velocity = np.sqrt(np.sum((agent_traj[1:] - agent_traj[:-1]) ** 2, -1))
    velocity = np.hstack((velocity[:1], velocity))
    # Compute acceleration
    acceleration = velocity[1:] - velocity[:-1]
    acceleration = np.hstack((acceleration, acceleration[-1:]))
    # Compute acceleration for all time steps.
    tokens = []
    current_token = None
    timestamps = []
    for t in range(velocity.shape[0]):
        t_start = t - window_size if t - window_size > 0 else 0
        t_end = t + window_size if t + window_size < velocity.shape[0] else velocity.shape[0]
        acc_before = acceleration[t_start:t]
        if len(acc_before) == 0:
            continue
        acc_after = acceleration[t:t_end]
        mean_before = acc_before.mean()
        std_bfore = acc_before.std()
        mean_after = acc_after.mean()
        std_after = acc_after.std()
        # print(agent_id, t, mean_before, std_bfore, mean_after, std_after)
        if std_bfore > window_std or std_after > window_std:
            # Uncertain if the angular velocity changes too much.
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                current_token = None
                timestamps = []
        elif mean_before >= 1.0 * acc_threshold and mean_after >= 1.0 * acc_threshold:
            # Turning left if keep moving toward left.
            if current_token != "SpeedUp" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "SpeedUp"
            timestamps.append(agent_traj[t][2].item())
        elif mean_before <= -1.0 * acc_threshold and mean_after <= -1.0 * acc_threshold:
            # Turning left if keep moving toward right.
            if current_token != "SlowDown" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "SlowDown"
            timestamps.append(agent_traj[t][2].item())
        else:
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                current_token = None
                timestamps = []
    if current_token is not None:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    return tokens


def get_normal_and_tangential_distance_point(
    point: Point, centerline_ls: LineString, delta: float = 0.01, last: bool = False
) -> Tuple[float, float]:
    """Get normal (offset from centerline) and tangential (distance along centerline) for the given point,
    along the given centerline

    Parameters
    ----------
    point: Point
        A point (x,y)-coordinate in map frame
    centerline: LineString
        centerline along which n-t is to be computed
    delta: float
        Used in computing offset direction
    last: bool
        True if point is the last coordinate of the trajectory

    Returns
    -------
    (tang_dist, norm_dist): Tuple[float, float]
        tangential and normal distances
    """
    tang_dist = centerline_ls.project(point)
    norm_dist = point.distance(centerline_ls)
    point_on_cl = centerline_ls.interpolate(tang_dist)

    # Deal with last coordinate differently. Helped in dealing with floating point precision errors.
    if not last:
        pt1 = point_on_cl.coords[0]
        pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
        pt3 = point.coords[0]

    else:
        pt1 = centerline_ls.interpolate(tang_dist - delta).coords[0]
        pt2 = point_on_cl.coords[0]
        pt3 = point.coords[0]

    lr_coords = []
    lr_coords.extend([pt1, pt2, pt3])
    lr = LinearRing(lr_coords)

    # Left positive, right negative
    if lr.is_ccw:
        return (tang_dist, norm_dist)
    return (tang_dist, -norm_dist)


def lane_change_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict) -> list:
    """Filter if the agent switches lanes.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
        map_elements: dict
            The dictionary of map info. Contains
            'lane_centers': list of lane center lines for the lanes in the map.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        lane_threshold: float
            The threshold to consider as closest lane. If the distances of an agent to a lane compared to another lane
            is within this threshold, we consider both lanes as the closest lane.
        skip_threshold: float
            If the distance to the closet lane exceeds this threshold, we don't consider it as a valid lane change,
            It is likely that the lane center ends earlier.

    Returns
    -------
    tokens: list
        A list of dictionaries that include the label and timestamps indicating the duration for lane changes.
    """
    lane_threshold = params["lane_threshold"]
    skip_threshold = params["skip_threshold"]
    window_size = params["window_size"]
    agent_traj = list(agent_trajectories.values())[0]
    traj_length = len(agent_traj)
    # Check for lane change maneuvers based on distance to centerlines.
    # First obtain all candidate centerlines.
    map_elements = scene_information["map_elements"]
    candidate_centerlines = map_elements["lane_centers"]
    # If there is only one lane, we cannot say anything about lane changes.
    if len(candidate_centerlines) <= 1:
        return []
    ls_centerlines = [LineString(centerlines) for centerlines in candidate_centerlines]
    pts = [Point(agent_traj[t][0], agent_traj[t][1]) for t in range(traj_length)]

    def get_closest_lane(pos: Point, threshold: float = lane_threshold):
        """Get the closest lanes from a position.
        Parameters
        ----------
        pos: Point
            The point position.
        threshold: float
            The threshold to consider the closest lane.

        Returns
        -------
        set
            The set of closest lanes.
        dist
            The distance to the closest lanes.
        """
        lane_idx = []
        d = 100000
        for i, lane in enumerate(ls_centerlines):
            dist = pos.distance(lane)
            # Only consider the lane as closest if it is the first lane encountered
            # or it  is within the distance threshold.
            if len(lane_idx) == 0 or dist < threshold:
                lane_idx.append(i)
                if dist < d:
                    d = dist
            elif dist < d:
                lane_idx = [i]
                d = dist
        return set(lane_idx), d

    def num_intersect_lanes(lanes: LineString):
        """Compute number of intersections given a list of lanes."""
        n_intersects = 0
        for lane_i in lanes:
            for lane_j in lanes:
                if lane_i == lane_j:
                    continue
                n_intersects += int(ls_centerlines[lane_i].intersects(ls_centerlines[lane_j]))
        return n_intersects / 2

    tokens = []
    current_token = None
    timestamps = []
    closest_lanes_d = [get_closest_lane(pts[t]) for t in range(traj_length)]
    closest_lanes = [l[0] for l in closest_lanes_d]
    ds = [l[1] for l in closest_lanes_d]
    for t in range(1, traj_length):
        # Skip the time points if there are many intersecting points, likely an intersection or lane merging.
        pt = Point(agent_traj[t, :2]).buffer(params["intersection_radius"])
        lanes_around = [lane_idx for lane_idx, lane in enumerate(ls_centerlines) if pt.intersects(lane)]
        if num_intersect_lanes(lanes_around) > params["max_intersection"]:
            continue
        t_start = t - window_size if t - window_size > 0 else 0
        t_end = t + window_size if t + window_size < traj_length else traj_length
        lanes_before = set.union(*(closest_lanes[t_start:t]))
        lanes_after = set.union(*(closest_lanes[t:t_end]))
        # Skip the time points if the agent passing through many centerlines.
        if (
            len(lanes_before) > params["max_lanes_in_window"]
            or len(lanes_after) > params["max_lanes_in_window"]
            or len(lanes_before) == 0
            or len(lanes_after) == 0
        ):
            continue
        if ds[t] > skip_threshold:
            # Cannot compute lane change if the closest lane is too far.
            if current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
                current_token = None
        elif len(lanes_after.intersection(lanes_before)) > 0:
            # Lane keep if closest lane does not change.
            if current_token != "LaneKeep" and current_token is not None:
                tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                timestamps = []
            current_token = "LaneKeep"
            timestamps.append(agent_traj[t][2].item())
        else:
            # Compute signed normal distance of current position to the lane
            # closest to the previous positions, and decide lane change
            # direction based on the difference in distances.
            _, dist_curr_to_prev = get_normal_and_tangential_distance_point(pts[t], ls_centerlines[lanes_before.pop()])
            _, dist_curr = get_normal_and_tangential_distance_point(pts[t], ls_centerlines[lanes_after.pop()])
            if dist_curr_to_prev > dist_curr:
                if current_token != "LaneChangeLeft" and current_token is not None:
                    tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                    timestamps = []
                current_token = "LaneChangeLeft"
            else:
                if current_token != "LaneChangeRight" and current_token is not None:
                    tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
                    timestamps = []
                current_token = "LaneChangeRight"
            timestamps.append(agent_traj[t][2].item())
    if current_token is not None:
        tokens.append(create_duration_label(current_token, timestamps[0], timestamps[-1]))
    return tokens


def follow_filter(agent_trajectories: OrderedDict, scene_information: dict, params: dict):
    """Filter if the agent follows other agents.

    Parameters
    ----------
    agent_trajectories: OrderedDict
        A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        The parameter dictionary for the filter, including the following keys.
        min_overlaps: int
            The minimum overlapping time steps for computing 'follow'.

    Returns
    -------
    token: list
        A list of dictionaries indicating the duration when "follow" is true and which agent is followed.
    """
    min_overlaps = params["min_overlaps"]
    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    target_traj_length = target_agent_traj.shape[0]
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    target_ls = LineString(target_agent_traj[:, :2]).buffer(0.5)
    tokens = []
    for other_agent_id in other_agent_ids:
        # Compute the overlapping area of two trajectories.
        other_agent_traj = agent_trajectories[other_agent_id]
        other_ls = LineString(other_agent_traj[:, :2]).buffer(0.5)
        overlaps = target_ls.intersection(other_ls)
        if overlaps.area == 0:
            continue
        # Get the points in the overlapping area.
        target_overlap = [
            target_agent_traj[t]
            for t in range(target_traj_length)
            if overlaps.contains(Point(target_agent_traj[t, :2]))
        ]
        other_overlap = [
            other_agent_traj[t]
            for t in range(other_agent_traj.shape[0])
            if overlaps.contains(Point(other_agent_traj[t, :2]))
        ]
        # It is following if the trajectory of the two agents overlaps and the other agent arrives
        # the overlapping ara earlier.
        if (
            len(target_overlap) > min_overlaps
            and len(other_overlap) > min_overlaps
            and other_overlap[0][2] < target_overlap[0][2]
        ):
            tokens.append(
                create_duration_label("Follow", target_overlap[0][2], target_overlap[-1][2], subject=other_agent_id)
            )
    return tokens


def find_points_in_region(positions: np.ndarray, region: LineString):
    """Find the points in a region.

    Parameters
    ----------
    positions: np.ndarray
        The positions of a trajectory.
    region: LineString
        The line region.
    """
    if region.is_empty:
        return []
    out = []
    for i, pt in enumerate(positions):
        if region.contains(Point(pt)):
            out.append(i)
    return out


def conservative_yield_filter(agent_trajectories: OrderedDict, scene_information: OrderedDict, params: dict) -> list:
    """If agent1 and agent2 start on different paths, end up in the same path, and agent2 gets first to the joint part of the path; compared to yield, with looser conditions on the distance maintained and the motion profile, and a parameter to contribute to close merge

    Parameters
    ----------
    agent_trajectories: OrderedDict
        Map from agent id to the trajectory of the agents. The first agent is the agent that may/may not yield.
        The later agents are all the agent possibly yielded to.
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        Parameters:
        yielding_prefix - minimal threshold (meters) to verify that the trajectories are long enough
        merge_time_threshold - the time threshold to filter the arrival timing difference between agent 1 and agent 2 at intersection point (small being close merge, large being regular merge)
        projected_merge_time_threshold - maximal threshold of the projected merge timing difference between agent 2 and agent 1 non-yielding projected motion, for agent 1 to be considered valid / reasonable / necessary / not overly conservative
        stop_speed_criteria - maximal yielding speed to be considered as yielding at intersections (which require explicit stopping)


    Returns
    -------
    token_list: list
        A list of tokens detected ("conservative-yield").
    """
    if len(list(agent_trajectories.values())) < 2:
        return []

    def compute_trajectory_stats(traj, dilation):
        """Compute trajectory stats needed for yielding"""
        positions = traj[:, :2]
        v = positions[1:, :] - positions[:-1, :]
        v = np.concatenate((v, v[-2:-1, :]), axis=0)
        spd1 = (v**2).sum(1)
        t = traj[:, 2]
        sl = LineString(positions.tolist()).buffer(dilation)
        return positions, spd1, t, sl

    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    yielding_dilation_radius = params["yielding_dilation_radius"]

    positions1, spd1, t1, sl1 = compute_trajectory_stats(target_agent_traj, yielding_dilation_radius)

    def moving_window_spd_estimate(spd1, sw):
        # check speed estimate smoothness
        acc_thr = 7.5
        _s_cache = spd1[:sw]
        sspd1 = np.array(np.mean(_s_cache))
        for _s in spd1[sw:]:
            _s_cache = _s_cache[1:]
            _s_cache = np.append(_s_cache, _s)
            sspd1 = np.append(sspd1, np.mean(_s_cache))

        sspd1 = np.append(sspd1, np.array([sspd1[-1]] * sw))

        if max(abs(sspd1[1:] - sspd1[:-1])) > acc_thr:
            print("..more speed smoothness needs to be applied on moving window")

        return sspd1

    sw = 5  # window size
    # sspd1 = moving_window_spd_estimate(spd1, sw)

    def low_pass_filter_spd_estimate(spd1, rt):
        acc_thr = 7.5
        spd = spd1[0]
        sspd1 = spd
        for _s in spd1:
            spd = ratio * spd + (1 - ratio) * _s
            sspd1 = np.append(sspd1, spd)
        if max(abs(sspd1[1:] - sspd1[:-1])) > acc_thr:
            print("..more speed smoothness needs to be applied on lp filter")

        return sspd1

    ratio = 0.05
    sspd1 = low_pass_filter_spd_estimate(spd1, ratio)

    slow_down1 = False
    max_spd1 = max(sspd1)
    min_spd1 = min(sspd1)
    if (max_spd1 - min_spd1) > params["yielding_speed_threshold"]:
        slow_down1 = True
    else:
        return []

    projection_speed = max_spd1  # projected non-yielding motion for agent 1 (the yielder)
    has_stop1 = False
    if min_spd1 < params["stop_speed_criteria"]:
        has_stop1 = True

    tokens = []
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    for other_agent_id in other_agent_ids:
        other_agent_traj = agent_trajectories[other_agent_id]
        positions2, spd2, t2, sl2 = compute_trajectory_stats(other_agent_traj, yielding_dilation_radius)
        intersection = sl1.intersection(sl2)
        idxs1 = find_points_in_region(positions1, intersection)
        idxs2 = find_points_in_region(positions2, intersection)
        if (
            len(idxs1) == 0 or len(idxs2) == 0
        ):  # (sy.lo) -- once intersection is not null, is this check still necessary?
            continue

        # slowed_down_speed_threshold = 0.5
        # too_slow1 = spd1[: idxs1[0]] < slowed_down_speed_threshold
        # slow_before1 = np.cumprod(too_slow1[::-1])
        # # This assumes delta-t's are smooth/constant
        # dt1 = np.array([t1[1] - t1[0]] + list(t1[1:] - t1[:-1]))
        # slow_before_dt1 = dt1[: idxs1[0]]
        # waiting_period1 = np.sum(slow_before1 * slow_before_dt1)

        # The trajectories upto the intersection.
        positions1b = positions1[: idxs1[0], :]
        positions2b = positions2[: idxs2[0], :]
        if positions1b.shape[0] < 2 or positions2b.shape[0] < 2:
            continue

        # compute idx reference to begin forward-simulating for yielder non-yielding motion profile to check timing validity
        idx_minspd1 = np.where(sspd1 == min(sspd1))[0][0]
        idx_r1 = 0  # since beginning (assuming 2 agents involved only)
        if (
            idx_minspd1 >= idxs1[0]
        ):  # if low speed profile begins after the merge, likely a spontaneously lane-change as slowing down at intersections # r1
            continue

        if has_stop1:
            projection_speed = (
                min_spd1 + sspd1[idxs1[0]]
            ) / 2  # projected non-stopping yielding motion for agent1 (the yielder)
            idx_r1 = np.where(sspd1 < params["yielding_speed_threshold"])[0][
                0
            ]  # since agent1 speed has ever fall below threshold

        idx_r2 = idx_r1  # use same time idx; assuming t1 t2 clocks synced - to sanity CHECK

        projected_pass_time2 = t2[idxs2[0]] - t2[idx_r2]
        # dis = (positions1[idxs1[0], :] - positions1[idx_r1, :]) # distance estimate
        _projected_pass_distance1 = math.sqrt(sum((positions1[idxs1[0], :] - positions1[idx_r1, :]) ** 2))

        if idxs1[0] > (len(t1) - 1):  # merged too late
            continue

        _p_dt1 = t1[idx_r1 : idxs1[0]] - t1[idx_r1 + 1 : idxs1[0] + 1]
        _projected_pass_distance1c = abs(sum(spd1[: len(_p_dt1)] * _p_dt1))

        # sanity check required
        if projected_pass_time2 < 0:
            continue

        projected_pass_time1 = _projected_pass_distance1c / projection_speed  # projected pass time
        actual_pass_time1 = t1[idxs1[0]] - t1[idx_r1]  # real pass time

        if projected_pass_time1 > actual_pass_time1:
            # method sanity check for debugging: shouldn't be triggered
            print("passing time estimate too conservative")
            # print(
            #    "if has stop? projected pass time estimates v.s. actual: {}, {}, {}".format(
            #        has_stop1, projected_pass_time1, actual_pass_time1
            #    )
            # )
            # print(
            #    "min_spd v.s. intersection spd v.s. projection spd, idx_minspd1, idxs1[0] = {}, {}, {}, {}, {}".format(
            #        sspd1[: idx_minspd1 + 1],
            #        sspd1[idx_minspd1 + 1 : idxs1[0] + 1],
            #        projection_speed,
            #        idx_minspd1,
            #        idxs1[0],
            #    )
            # )
            continue

        # projected_pass_time1 = min(projected_pass_time1, actual_pass_time1) #conservative fastest pass time estimate v.s. real pass time #r2

        # if projected pass time difference is smaller than some threshold: non-conservative
        if abs(projected_pass_time1 - projected_pass_time2) < params["projected_merge_time_threshold"]:
            continue

        sl1b = LineString(positions1b[:, :2]).buffer(yielding_dilation_radius)
        sl2b = LineString(positions2b[:, :2]).buffer(yielding_dilation_radius)

        # The regions that cross each other
        sl1_only = (sl1b.difference(sl2)).buffer(-yielding_dilation_radius / 2.0)
        sl2_only = (sl2b.difference(sl1)).buffer(-yielding_dilation_radius / 2.0)
        # Skip if no overlaps in the trajectory.
        # idx1_m1 = max(0, idxs1[0] - params["yielding_dt"])
        if (
            sl1_only.length
            > params[
                "yielding_prefix"
            ]  # long enough prefix for trajectory 1 #(sy.lo) -- checking agent2 should be sufficient
            and sl2_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 2
            # and t1[idxs1[0]] - t2[idxs2[0]] > params["yielding_time_gap"]  # agent 1 is before agent 2
            and 0 < t1[idxs1[0]] - t2[idxs2[0]] < params["merge_time_threshold"]  # CLOSE-MERGE-TEST
            # and spd1[idx1_m1] - spd1[idxs1[0]] > -0.5  # non-increasing speed. #(sy.lo) -- should the sign of > be flipped?
            # and Point(positions2b[0, :]).distance(sl1_only)
            # > params["yielding_initial_distance"]  # initial point for 2 is far from trajectory 1
            # sy.lo: is this necessary if we have checked the prefix?
        ):
            new_token = create_duration_label("Conservative-Yield", t2[idxs2[0]], t2[idxs2[-1]], subject=other_agent_id)
            tokens.append(new_token)

    ## multi-agent check - only populate the closest merging agent
    # tokens_ = []
    # for _k in merge_time_idx1.keys():
    #    _idx = np.where(np.array(merge_time_diff[_k]) == min(merge_time_diff[_k]))[0][0]
    #    merge_idx = merge_time_idx1[_k][_idx]
    #    tokens_.append(tokens[merge_idx])

    return tokens


def merge_filter(agent_trajectories: OrderedDict, scene_information: OrderedDict, params: dict) -> list:
    """If agent1 and agent2 start on different paths, end up in the same path, and agent2 gets first to the joint part of the path; compared to yield, with looser conditions on the distance maintained and the motion profile, and a parameter to contribute to close merge

    Parameters
    ----------
    agent_trajectories: OrderedDict
        Map from agent id to the trajectory of the agents. The first agent is the agent that may/may not yield.
        The later agents are all the agent possibly yielded to.
    scene_information: dict
        Scene information, has map_elements->OrderedDict as a field.
    params: dict
        Parameters:
        yielding_prefix - minimal threshold (meters) to verify that the trajectories are long enough
        merge_time_threshold - the time threshold to filter the arrival timing difference between agent 1 and agent 2 at intersection point (small being close merge, large being regular merge)

    Returns
    -------
    token_list: list
        A list of tokens detected ("merge").
    """
    if len(list(agent_trajectories.values())) < 2:
        return []

    def compute_trajectory_stats(traj, dilation):
        """Compute trajectory stats needed for yielding"""
        positions = traj[:, :2]
        v = positions[1:, :] - positions[:-1, :]
        v = np.concatenate((v, v[-2:-1, :]), axis=0)
        spd1 = (v**2).sum(1)
        t = traj[:, 2]
        sl = LineString(positions.tolist()).buffer(dilation)
        return positions, spd1, t, sl

    target_agent_id = list(agent_trajectories.keys())[0]
    target_agent_traj = agent_trajectories[target_agent_id]
    yielding_dilation_radius = params["yielding_dilation_radius"]

    positions1, spd1, t1, sl1 = compute_trajectory_stats(target_agent_traj, yielding_dilation_radius)

    tokens = []
    merge_time_idx1 = defaultdict(list)
    merge_time_diff = defaultdict(list)
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    for other_agent_id in other_agent_ids:
        other_agent_traj = agent_trajectories[other_agent_id]
        positions2, spd2, t2, sl2 = compute_trajectory_stats(other_agent_traj, yielding_dilation_radius)
        intersection = sl1.intersection(sl2)
        idxs1 = find_points_in_region(positions1, intersection)
        idxs2 = find_points_in_region(positions2, intersection)
        if (
            len(idxs1) == 0 or len(idxs2) == 0
        ):  # (sy.lo) -- once intersection is not null, is this check still necessary?
            continue
        if idxs1[0] < idxs2[0]:
            # This means the agent1 is ahead of agent2
            continue
        # Agent1 is after the agent2

        # compute idx reference to begin forward-simulating for yielder non-yielding motion profile to check timing validity
        idx_minspd1 = np.where(spd1 == min(spd1))[0][0]
        if (
            idx_minspd1 >= idxs1[0]
        ):  # if low speed profile begins after the merge, likely a spontaneously lane-change as slowing down at intersections # r1
            continue

        # slowed_down_speed_threshold = 0.5
        # too_slow1 = spd1[: idxs1[0]] < slowed_down_speed_threshold
        # slow_before1 = np.cumprod(too_slow1[::-1])
        # # This assumes delta-t's are smooth/constant
        # dt1 = np.array([t1[1] - t1[0]] + list(t1[1:] - t1[:-1]))
        # slow_before_dt1 = dt1[: idxs1[0]]
        # waiting_period1 = np.sum(slow_before1 * slow_before_dt1)

        # The trajectories upto the intersection.
        positions1b = positions1[: idxs1[0], :]
        positions2b = positions2[: idxs2[0], :]
        if positions1b.shape[0] < 2 or positions2b.shape[0] < 2:
            continue

        sl1b = LineString(positions1b[:, :2]).buffer(yielding_dilation_radius)
        sl2b = LineString(positions2b[:, :2]).buffer(yielding_dilation_radius)

        # The regions that cross each other
        sl1_only = (sl1b.difference(sl2)).buffer(-yielding_dilation_radius / 2.0)
        sl2_only = (sl2b.difference(sl1)).buffer(-yielding_dilation_radius / 2.0)
        # Skip if no overlaps in the trajectory.
        # idx1_m1 = max(0, idxs1[0] - params["yielding_dt"])
        if (
            sl1_only.length
            > params[
                "yielding_prefix"
            ]  # long enough prefix for trajectory 1 #(sy.lo) -- checking agent2 should be sufficient
            and sl2_only.length > params["yielding_prefix"]  # long enough prefix for trajectory 2
            # and t1[idxs1[0]] - t2[idxs2[0]] > params["yielding_time_gap"]  # agent 1 is before agent 2
            and 0 < t1[idxs1[0]] - t2[idxs2[0]] < params["merge_time_threshold"]  # CLOSE-MERGE-TEST
            # and spd1[idx1_m1] - spd1[idxs1[0]] > -0.5  # non-increasing speed. #(sy.lo) -- should the sign of > be flipped?
            and Point(positions2b[0, :]).distance(sl1_only)
            > params["yielding_initial_distance"]  # initial point for 2 is far from trajectory 1
        ):  # is this necessary if we have checked the prefix?
            time_diff = t1[idxs1[0]] - t2[idxs2[-1]]
            aggressiveness_val = time_diff
            new_token = create_duration_label(
                "Merge",
                t2[idxs2[-1]],
                t1[idxs1[0]],
                subject=other_agent_id,
                value=aggressiveness_val,
                leading_agent=other_agent_id,
                following_agent=target_agent_id,
            )
            tokens.append(new_token)
            merge_time_idx1[idxs1[0]].append(len(tokens) - 1)
            merge_time_diff[idxs1[0]].append(t1[idxs1[0]] - t2[idxs2[0]])

    # multi-agent check - only populate the closest merging agent
    tokens_ = []
    for _k in merge_time_idx1.keys():
        _idx = np.where(np.array(merge_time_diff[_k]) == min(merge_time_diff[_k]))[0][0]
        merge_idx = merge_time_idx1[_k][_idx]
        tokens_.append(tokens[merge_idx])

    return tokens_

