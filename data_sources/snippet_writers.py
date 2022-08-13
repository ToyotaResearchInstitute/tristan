import copy
import json
import os

import numpy as np


def save_adovehicle_track(
    file_prefix, save_dir, file_counter, track_data, unique_id=None, sequence_counter=None, indent=4
):
    """
    save track data for an adovehicle instance. Creates the filename, with running counter of adovehicle instances,
    track_data is currently a dictionary of all desired values:

    :param file_prefix:
    :param file_counter:
    :param track_data: track_data = a dictionary with values:
    * 'unique_id': unique_id,
    * 'track_time': current_time,
    * 'normalizing_pose': 'normalizing_pose',
    * 'nearby_agents': nearby_agents,
    * 'track': normalized_track
    * 'timestamp': timestamps for the track points

    :return:
    """
    filename = (
        os.path.join(save_dir, file_prefix)
        + ("" if unique_id is None else "_" + str(unique_id))
        + ("" if sequence_counter is None else "_" + "{0:06d}".format(sequence_counter))
        + ("" if file_counter is None else "_" + "{0:08d}".format(file_counter))
        + ".json"
    )
    with open(filename, mode="w") as fp:
        json.dump(track_data, fp, indent=indent, sort_keys=True)


def normalize_track(track, normalizing_pose, is_normalizing=False):
    """
    Apply normalizing transform to a track
    :param track: a track, with fields 'position_L','heading_L'
    :param normalizing_pose: a pair (R,t)
    :param is_normalizing: is this a track used to create a normalizing pose (we can assert directions)
    :return:
    """
    res = copy.deepcopy(track)
    R, t = normalizing_pose

    for i in range(len(res["heading_L"])):
        if not res["heading_L"][i] is None:
            res["heading_L"][i] = (np.dot(R, res["heading_L"][i])).tolist()

    res["position_L"] = normalize_positions_list(res["position_L"], R, t)
    if is_normalizing:
        # assert that if the ado-vehicle is driving, the track after normalization is heading in a specific direction
        dx = np.array(res["position_L"][0]) - np.array(res["position_L"][-1])
        if np.linalg.norm(dx) > 2 and np.cos(np.arctan2(dx[1], dx[0])) > 0.5:
            return None

    return res


def normalize_positions_list(pos_list, R, t):
    res_list = copy.deepcopy(pos_list)
    for i in range(len(res_list)):
        if not res_list[i] is None:
            new_pose = (R @ np.array([res_list[i]]).transpose()) + np.array([t]).transpose()
            res_list[i] = new_pose.transpose().tolist()[0]
    return res_list


def create_normalizing_pose(track, future_length, velocity_threshold=4.0):
    """
    Extract a normalizing pose from a track
    :param track: contains 'position_L','heading_L'
    :return: R,t, a normalizing transform.
    """
    # TODO: add code in case heading is not None's
    # TODO: decide which reference point to use
    if len(track["position_L"]) <= future_length:
        return None
    pos1 = track["position_L"][-future_length]

    # otherwise, throw away the example - rotation/pose is unclear.
    # TODO: use heading
    for i in range(len(track["position_L"]) - 1):
        pos0 = track["position_L"][-i - 1]
        # TODO: better estimate
        velocity_v = np.array(pos1) - np.array(pos0)
        if np.linalg.norm(velocity_v) > velocity_threshold:
            break
    if np.linalg.norm(velocity_v) < velocity_threshold:
        return None
    # TODO: check if adovehicle headers are stable when they are available..
    # if not track['heading_L'][0] is None:
    #     R=[[np.cos(track['heading_L'][0]),np.sin(track['heading_L'][0])],[-np.sin(track['heading_L'][0]),np.cos(track['heading_L'][0])]]
    #     R=np.array(R)
    # else:

    v1 = velocity_v / np.linalg.norm(velocity_v)
    v2 = np.array([-v1[1], v1[0]])
    R = np.array([v1, v2])
    c = pos1
    t = -np.dot(R, c)
    return R, t
