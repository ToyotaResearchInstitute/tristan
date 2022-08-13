import numpy as np
from scipy import interpolate as interpolate
from shapely.geometry import LineString
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def resample_lane(lane_coordinates, distance=1.0):
    """
    Resample equidistance coordinates from a set of input lane coordinates.
    :param lane_coordinates: input lane coordinates.
    :param distance: distance between points.
    :return: resampled lane coordinates.
    """
    lane_ls = LineString(lane_coordinates)
    lane_distance = lane_ls.length
    sample_distance = distance
    sampled_lane_coordinates = [lane_coordinates[0]]
    while lane_distance > sample_distance:
        next_pos = lane_ls.interpolate(sample_distance)
        sampled_lane_coordinates.append([next_pos.x, next_pos.y])
        sample_distance += distance

    sampled_lane_coordinates.append(lane_coordinates[-1])
    return sampled_lane_coordinates


def smooth_trajectory(traj, smooth_type="gp", mode="train", sampling_rate=0.1):
    """
    Smooth agent trajectory. This function is designed for Argoverse dataset.

    Parameters
    ----------
    traj : np.ndarray,
        input trajectory.

    Returns
    -------

    """
    new_positions = np.zeros_like(traj)

    # This assumes 20 steps in the past and 30 steps in the future (for Argoverse).
    if mode == "test":
        timestamps = np.arange(0, 2, 0.1)
    elif mode == "train":
        timestamps = np.arange(0, 5, 0.1)
    else:
        timestamps = [sampling_rate * i for i in range(new_positions.shape[0])]

    # Smooth each dimension separately.
    for d in range(2):
        # spline not working so well.
        if smooth_type == "spline":
            f = interpolate.CubicSpline(timestamps, traj[:, d])
            new_positions[:, d] = f(timestamps)
        elif smooth_type == "gp":
            kernel = 1.0 * Matern(length_scale=1.0)  # default: 1.0 * RBF(1.0)
            noise_level = 1e-1  # default: 1e-10
            gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_level, n_restarts_optimizer=10)
            gp.fit(np.atleast_2d(timestamps).T, traj[:, d])
            new_positions[:, d] = gp.predict(np.atleast_2d(timestamps).T)
    return new_positions
