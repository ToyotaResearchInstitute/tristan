import hashlib
import json
import logging
import multiprocessing
import os
import subprocess
import time
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, cast

import numpy as np
import PIL
from filelock import FileLock, Timeout
from matplotlib import pyplot as plt
from scipy import interpolate as interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from torchvision import transforms

from radutils.misc import parse_protobuf_timestamp, remove_prefix
from triceps.protobuf import protobuf_training_parameter_names
from triceps.protobuf.prediction_dataset import ProtobufDatasetParsingError, ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_cache import CacheElement
from triceps.protobuf.protobuf_data_names import ADDITIONAL_INPUT_KEY_VISUAL_BBOX
from triceps.protobuf.protobuf_training_parameter_names import (
    PARAM_FUTURE_TIMESTEPS,
    PARAM_MAX_AGENTS,
    PARAM_PAST_TIMESTEPS,
)


class AuxiliaryStateIndices(IntEnum):
    """
    Save the indices for different element in auxiliary state tensor.
    """

    AUX_STATE_IDX_LENGTH = 0
    AUX_STATE_IDX_WIDTH = 1
    AUX_STATE_IDX_YAW = 2
    AUX_STATE_IDX_VEL_X = 3
    AUX_STATE_IDX_VEL_Y = 4


def compute_hash(string: str) -> str:
    return hashlib.md5(string.encode()).hexdigest()


# TODO(guy.rosman): create a parent class for image_processor_wrapper and use it in the typing of PA_Wrapper.
class ImageProcessor(object):
    """Pre-processes an image -- e.g. with PA or similar. Stub class to be implemented."""

    def __init__(self, image_processor_wrapper, output_dim: int, params: dict):
        """Inits ImageProcessor.

        Parameters
        ----------
        output_dim: int
          The dimensionality of the output.
        params: dict
          Dictionary with parameters.

        """
        assert image_processor_wrapper is not None
        self.image_processor_wrapper = image_processor_wrapper
        self.output_dim = output_dim

    def process(self, img, bboxes: Optional[Tuple]):
        """Process image, return a vector of features

        Parameters
        ----------
        img : array
          input image (dimensionality TBD)
        bboxes: tuple
          The (x,y,width,height) of a bbox

        Returns
        -------
        result : array
          1xD result array
        """
        # TODO: Move bboxes into the ImageProcessor.
        result, full_prediction = self.image_processor_wrapper._process_impl(img, bboxes)
        assert result.shape[1] == self.output_dim
        return result


def linecount_wc(filename):
    return int(
        subprocess.run(f"wc -l {filename}".split(" "), capture_output=True).stdout.decode().strip(" ").split(" ")[0]
    )
    # return int(os.popen(f"wc -l {filename}").read().split()[0])


def list_dir(find_cmd: list, cache_dir, cache_id, image_dir_prefix, num_key_letters, verbose=False):
    cache_pathname = os.path.join(cache_dir, cache_id + ".txt")
    lock = FileLock(cache_pathname + ".lock")
    try:
        lock.acquire(timeout=0.1)
        warning = f"Acquired file lock for dir: {cache_pathname}"
        print(warning)
        logging.warning(warning)

    except Timeout as e:
        now = time.monotonic()
        warning = f"There's another process listing the same dir: {cache_pathname}, wait for it to finish then return. Waiting."
        print(warning)
        logging.warning(warning)
        lock.acquire(poll_intervall=0.5)
        lock.release()
        now2 = time.monotonic()
        warning = f"Waited list_dir {cache_pathname} for: {now2-now} seconds."
        print(warning)
        logging.warning(warning)
        return

    try:
        # Create full permutation of num_key_letters digits of the hexadecimal
        key_letters = ["{:x}".format(n).zfill(num_key_letters) for n in range(16**num_key_letters)]
        target_cache_pathname_flag = os.path.join(cache_dir, cache_id + "_finish" + ".txt")

        tmp_cache_pathname = cache_pathname + ".tmp"
        with open(tmp_cache_pathname, "w") as f:
            cmd = find_cmd
            subprocess.run(cmd, stdout=f)
            f.flush()
        # Remove dir prefix
        sed_pattern = "s/^{}//".format(image_dir_prefix.replace("/", "\\/"))
        with open(cache_pathname, "w") as f:
            cmd = ["sed", sed_pattern, tmp_cache_pathname]
            subprocess.run(cmd, stdout=f)
            f.flush()

        print(f"line count is: {linecount_wc(cache_pathname)}, for {cache_pathname}")
        with ThreadPoolExecutor(max_workers=max(20, multiprocessing.cpu_count() * 2)) as executor:
            futures = []

            def grep_key(key):
                cache_pathname_subset = os.path.join(cache_dir, cache_id + "_" + key + ".txt")
                cmd = ["grep", "/" + key, cache_pathname]
                with open(cache_pathname_subset, "w") as f:
                    subprocess.run(cmd, stdout=f)
                if verbose:
                    print(
                        f"writing sub-file for {key}, at {cache_pathname_subset}, "
                        f"line count is: {linecount_wc(cache_pathname_subset)}"
                    )

            for key_letter in key_letters:
                future = executor.submit(grep_key, key_letter)
                futures.append(future)

            # Wait for them to finish.
            [f.result() for f in futures]

        with open(target_cache_pathname_flag, "w") as f:
            f.write("finished\n")

        os.remove(cache_pathname)
        os.remove(tmp_cache_pathname)
    finally:
        lock.release()


def get_image_file_map(
    params,
    img_dir,
    img_dir_prefix,
    img_data_folder,
    filename,
    num_key_letters=3,
    alternate_extension=None,
    verbose=True,
):
    """Gets the image path for an image filename/harvest folder, in a local img_dir.

    Parameters
    ----------
    params : dict
      The training params dictionary. Used for the cache_dir variable.
    img_dir : string
      A local folder with images.
    img_data_folder : string
      The harvest image folder (currently not used for hashing)
    filename : string
      The harvest image name - used to search/retrieve the image. The prefix is used to create smaller index files.
    verbose : bool
      Whether to print debugging info.

    Returns
    -------
    img_path : string
      Local image path. None if match is not found.
    """
    rel_img_dir = remove_prefix(img_dir, img_dir_prefix)
    cache_id = "imgfile_map_" + hashlib.md5(rel_img_dir.encode()).hexdigest()
    cache_dir = params["cache_dir_original"] + "/image_files_list/" + cache_id
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except:
        pass

    cache_pathname = os.path.join(cache_dir, cache_id + ".txt")
    key = filename[:num_key_letters]
    target_cache_pathname_flag = os.path.join(cache_dir, cache_id + "_finish" + ".txt")
    target_cache_pathname_subset = os.path.join(cache_dir, cache_id + "_" + key + ".txt")
    if verbose:
        print('key = "{}", target subset file {}'.format(key, target_cache_pathname_subset))

    if not os.path.exists(target_cache_pathname_subset) or not os.path.exists(target_cache_pathname_flag):
        print("looking for key file {}".format(target_cache_pathname_subset))
        print("list files for image dir {}, at {}".format(img_dir, cache_pathname))
        list_dir(
            ["find", os.path.join(img_dir, "")],
            cache_dir,
            cache_id,
            img_dir_prefix,
            num_key_letters=num_key_letters,
            verbose=verbose,
        )

    if verbose:
        print("target_cache_pathname_subset = {}".format(target_cache_pathname_subset))
        print("filename = {}".format(filename))

    result = None
    if filename is not None:
        if verbose:
            print("Searching filenames via grep: {}, {}".format(filename, target_cache_pathname_subset))
        grep_arg = Path(filename).stem
        process = subprocess.Popen(
            ["grep", "-m", "1", grep_arg, target_cache_pathname_subset], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = process.communicate()
        if verbose:
            print("Got: {}".format(output))
        if "No such file or directory" in output:
            print(f"Can't fine subset file {target_cache_pathname_subset}")
        elif len(output[0]) > 0:
            # print('found filename via grep: {}, {}'.format(filename, cache_pathname))
            result = output[0].decode().rstrip()
    if result is not None:
        result = img_dir_prefix + "/" + result
        if alternate_extension is not None:
            result = str(Path(result).with_suffix("." + alternate_extension))
    return result


def get_image_path(
    params,
    img_dirs,
    img_dir_prefix,
    img_data_folder,
    img_data_filename,
    num_key_letters,
    alternate_extension=None,
    verbose=False,
):
    """
    Returns the correct image path
    :param params: The training params dictionary. Used for the cache_dir variable.
    :param img_dirs: a list of local harvest folders with images.
    :param img_data_folder: the harvest image folder (currently not used for hashing)
    :param filename: the harvest image name - used to search/retrieve the image. The prefix is used to create smaller index files.
    :return: img_path - the local image path. None if no match is found in any of the folders in img_dirs.
    """
    if type(img_dirs) is str:
        img_dirs = [img_dirs]
    for img_dir in img_dirs:
        if verbose:
            print("Searching: {}".format(img_dir))
        if not os.path.exists(img_dir):
            raise ValueError("Missing image folder: {}".format(img_dir))
        img_path = get_image_file_map(
            params,
            img_dir,
            img_dir_prefix,
            img_data_folder,
            img_data_filename,
            num_key_letters,
            alternate_extension=alternate_extension,
            verbose=verbose,
        )
        if img_path is not None and len(img_path) > 0:
            break
    return img_path


def remove_duplicate_points(positions, timestamps):
    """
    :param positions: size [n_timestamps, 3]
    :param timestamps: all timestamps (relative to t_now)
    """
    coordinates = positions[:, :2]
    _, idx = np.unique(coordinates, return_index=True, axis=0)  # indices of unique points (first occurrences)
    idx = np.sort(idx)
    new_positions = positions[idx, :]  # remove duplicate points
    new_timestamps = timestamps[idx]

    return new_positions, new_timestamps


def truncate_timespan(positions, timestamps, past_timesteps, future_timesteps, past_dt, future_dt):
    """
    truncate timespan for interpolation at [-max_t//2 * dt, max_t//2 * dt]
    :param positions: size [n_timestamps, 3]
    :param timestamps: all timestamps (relative to t_now)
    :param max_t:
    :param past_dt: timestep size for the past
    :param future_dt: timestep size for the future
    """
    # add padding time for better interpolation (avoid extrapolation)
    padding_ratio = 0.4  # ratio relative to max_t for padding
    total_timesteps = past_timesteps + future_timesteps
    padding = int(padding_ratio * total_timesteps)
    earliest = (-past_timesteps - padding) * past_dt
    latest = (future_timesteps + padding) * future_dt
    idx = np.where((timestamps >= earliest) & (timestamps <= latest))[0]
    new_timestamps = timestamps[idx]
    new_positions = positions[idx, :]
    return new_positions, new_timestamps


def interpolate_trajectory(positions, timestamps, params, k, visualization_filename, handler_params):
    """interpolate a trajectory, may fail if the trajectory is invalid.

    Parameters
    ----------
    positions :
        positions with size [n_timestamps, 3]
    timestamps :
        all timestamps (relative to t_now)
    params
    k :
        agent_id
    visualization_filename
    handler_params:
        Is ego_vehicle / relevant_agent

    Returns
    -------

    positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate
    """
    is_too_short = False
    is_crucially_too_short = False
    is_inaccurate = False
    is_ego = handler_params["is_ego_vehicle"]
    is_relevant = handler_params["is_relevant_agent"]
    is_precheck = handler_params["is_precheck"]
    past_timesteps = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS]
    future_timesteps = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS]
    total_timesteps = past_timesteps + future_timesteps
    past_dt = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE]
    future_dt = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE]
    interp_types = params["interp_type"]
    visualize_interp = params["visualize_interp"] or params["visualize_interp_bad_only"]
    log_interp_excursions = params["log_interp_excursions"]
    past_only = params["past_only_dataloading"]
    is_valid = np.where(positions[:, 2] == 1.0)[0]
    positions = positions[is_valid, :]
    timestamps = timestamps[is_valid]
    minimum_timestamps_len = 2
    # TODO(guy.rosman) Add: if the whole trajectory is invalid, return critically too short.
    positions_no_dup, timestamps_no_dup = remove_duplicate_points(positions, timestamps)
    positions_trunc, timestamps_trunc = truncate_timespan(
        positions_no_dup, timestamps_no_dup, past_timesteps, future_timesteps, past_dt, future_dt
    )
    new_past_timestamps = np.arange((-past_timesteps + 1) * past_dt, 0, past_dt)
    new_future_timestamps = np.arange(0, (future_timesteps + 0.5) * future_dt, future_dt)
    new_timestamps = np.concatenate([new_past_timestamps, new_future_timestamps])
    is_ego_or_relevant = is_ego + is_relevant
    if params["trajectory_validity_criterion"] == "conservative-relevant":
        if is_ego_or_relevant:
            # if the agent is ego vehicle or relevant agent, valid trajectory should cover the range of new_timestamps
            if timestamps_trunc.min() > new_timestamps[0] or timestamps_trunc.max() < new_timestamps[-1]:
                is_too_short = True
                is_crucially_too_short = True
                return positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate
    elif params["trajectory_validity_criterion"] == "conservative-ego-only":
        EGO_SUPPORT_PADDING = params["trajectory_time_ego_padding"]  # in seconds
        ADO_SUPPORT_PADDING = params["trajectory_time_ado_padding"]  # in seconds
        assert params["augmentation_timestamp_scale"] < EGO_SUPPORT_PADDING
        assert params["augmentation_timestamp_scale"] < ADO_SUPPORT_PADDING
        if is_ego:
            if is_precheck:
                EGO_SUPPORT_PADDING -= params["augmentation_timestamp_scale"]
            if (
                timestamps_trunc.min() > new_timestamps[0] + EGO_SUPPORT_PADDING
                or ((not past_only) and timestamps_trunc.max() < new_timestamps[-1] - EGO_SUPPORT_PADDING)
                or len(timestamps_no_dup) < minimum_timestamps_len
            ):
                is_too_short = True
                is_crucially_too_short = True
                return positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate
        elif is_relevant:
            if is_precheck:
                ADO_SUPPORT_PADDING -= params["augmentation_timestamp_scale"]
            if (
                timestamps_trunc.min() > new_timestamps[0] + ADO_SUPPORT_PADDING
                or ((not past_only) and timestamps_trunc.max() < new_timestamps[-1] - ADO_SUPPORT_PADDING)
                or len(timestamps_no_dup) < minimum_timestamps_len
            ):
                is_too_short = True
                is_crucially_too_short = True
                return positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate
        elif len(timestamps_no_dup) < minimum_timestamps_len or len(new_timestamps) < 2 or len(timestamps_trunc) < 2:
            is_too_short = True
            is_crucially_too_short = True
            return positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate

    else:
        # if the agent belongs to others, valid trajectory should include a min number of points for both past/future
        if (
            timestamps_trunc.min() > (-params["min_valid_points"] + 1) * past_dt
            or ((not past_only) and timestamps_trunc.max() < (params["min_valid_points"] + 1) * future_dt)
            or len(timestamps_no_dup) < minimum_timestamps_len
            or len(new_timestamps) < 2
        ):
            is_too_short = True
            return positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate

    new_positions = np.zeros((total_timesteps, 3))
    idx_valid = np.where(
        np.logical_and(new_timestamps < timestamps_trunc.max(), new_timestamps > timestamps_trunc.min())
    )[0]
    if idx_valid.shape[0] == 0:
        is_too_short = True
        is_crucially_too_short = True
        return positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate

    interp_flag = False
    new_positions_vis = {}

    if "interpSpline" in interp_types:
        try:
            for d in range(2):
                f = interpolate.CubicSpline(timestamps_trunc, positions_trunc[:, d])
                new_positions[idx_valid, d] = f(new_timestamps[idx_valid])
            new_positions[idx_valid, 2] = 1.0
            new_positions_vis["interpSpline"] = new_positions[idx_valid, :2]
            interp_flag = True
        except:
            pass
    if "interp1d" in interp_types:
        try:
            for d in range(2):
                f = interpolate.interp1d(
                    timestamps_trunc, positions_trunc[:, d], kind="quadratic", fill_value="extrapolate"
                )
                new_positions[idx_valid, d] = f(new_timestamps[idx_valid])
            new_positions[idx_valid, 2] = 1.0
            new_positions_vis["interp1d"] = new_positions[idx_valid, :2]
            interp_flag = True
        except:
            pass
    if "interpGP" in interp_types:
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            kernel = 1.0 * Matern(length_scale=1.0)  # default: 1.0 * RBF(1.0)
            noise_level = 1e-2  # default: 1e-10
            gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_level, n_restarts_optimizer=10)
            for d in range(2):
                gp.fit(np.atleast_2d(timestamps_trunc).T, positions_trunc[:, d])
                if not warn:
                    new_positions[idx_valid, d] = gp.predict(np.atleast_2d(new_timestamps[idx_valid]).T)
                    new_positions[idx_valid, 2] = 1.0
                    interp_flag = True
            new_positions_vis["interpGP"] = new_positions[idx_valid, :2]
    if not interp_flag:
        new_positions = positions[:total_timesteps, :]
        new_timestamps = timestamps[:total_timesteps]

    if len(new_timestamps) < (past_timesteps + future_timesteps):
        raise RuntimeError(
            f"len(new_timestamps) = {len(new_timestamps)} < "
            f"(past_timesteps + future_timesteps)({ past_timesteps + future_timesteps})"
        )

    interp_delta_x = np.quantile(new_positions[idx_valid, 0], [0, 1]) - np.quantile(positions_trunc[:, 0], [0, 1])
    interp_delta_y = np.quantile(new_positions[idx_valid, 1], [0, 1]) - np.quantile(positions_trunc[:, 1], [0, 1])
    max_excursion = np.max([-interp_delta_x[0], -interp_delta_y[0], interp_delta_x[1], interp_delta_y[1]])

    if max_excursion > params["interpolation_excursion_threshold"]:
        visualize_subfolder = "inaccurate"
        is_inaccurate = True
        # TODO(guy.rosman): move to a logger / log somewhere the statistics of excursions.
        if log_interp_excursions:
            print("excursion: {}, {}".format(max_excursion, k))
    else:
        visualize_subfolder = "accurate"
        # Skip visualization of accurate interpolations
        if params["visualize_interp_bad_only"]:
            visualize_interp = False

    if visualize_interp:
        visualization_folder = os.path.join(
            os.path.expandvars(os.path.expanduser(params["interp_dir"])), visualize_subfolder
        )
        os.makedirs(visualization_folder, exist_ok=True)
        png_file = os.path.join(
            visualization_folder, os.path.basename(visualization_filename)[:-3] + "_" + str(k) + ".png"
        )
        base_dir, _ = os.path.split(png_file)
        os.makedirs(base_dir, exist_ok=True)
        visualize_interpolation(
            params,
            timestamps,
            positions,
            timestamps_no_dup,
            positions_no_dup,
            timestamps_trunc,
            positions_trunc,
            new_timestamps,
            idx_valid,
            new_positions_vis,
            png_file,
        )

    return new_positions, new_timestamps, is_too_short, is_crucially_too_short, is_inaccurate


def add_padding_to_bbox(bbox: List[int], padding_ratio: float):
    top, left, height, width = bbox
    vertical_padding = height * padding_ratio
    horizontal_padding = width * padding_ratio

    return np.array(
        [
            top - vertical_padding,
            left - horizontal_padding,
            height + vertical_padding * 2,
            width * horizontal_padding * 2,
        ],
        dtype=np.float,
    )


def visualize_interpolation(
    params,
    timestamps,
    positions,
    timestamps_no_dup,
    positions_no_dup,
    timestamps_trunc,
    positions_trunc,
    new_timestamps,
    idx_valid,
    new_positions_vis,
    png_file,
):
    """Visualization of removing duplicate points (left) and interpolating trajectory (right) in x (upper) and y (lower)

    Parameters
    ----------
    params : dict
        Global parameters
    timestamps : numpy.ndarray
        Original timestamps of shape (t, )
    positions : numpy.ndarray
        Original positions of shape (t, 3);
    timestamps_no_dup : numpy.ndarray
        Timestamps after removing duplicate points, with shape of (t_no_dup, )
    positions_no_dup : numpy.ndarray
        Points after removing duplicate ones, with shape of (t_no_dup, 3)
    timestamps_trunc : numpy.ndarray
        Timestamps after truncating, with shape of (t_trunc, )
    positions_trunc : numpy.ndarray
        Points after truncating, with shape of (t_trunc, 3)
    new_timestamps : numpy.ndarray
        New timestamps of interpolations, with shape of (t_new, )
    idx_valid : numpy.ndarray
        Indices of valid points in new timestamps
    new_positions_vis : dict of {key: numpy.ndarray}
        New positions under different interp types with keys of interp type (e.g., 'interpSpline')
    png_file : str
        Image file path
    """
    fig = plt.figure(figsize=[10.8, 4.8])

    # Visualization of removing duplicate points
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(timestamps, positions[:, 0], "r.-", label="original")
    ax1.plot(timestamps_no_dup, positions_no_dup[:, 0], "b.-", label="no duplicate")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("x (m)")
    ax1.legend(frameon=False)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(timestamps, positions[:, 1], "r.-", label="original")
    ax3.plot(timestamps_no_dup, positions_no_dup[:, 1], "b.-", label="no duplicate")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("y (m)")
    ax3.legend(frameon=False)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(timestamps_trunc, positions_trunc[:, 0], "r.-", label="truncated")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("x (m)")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(timestamps_trunc, positions_trunc[:, 1], "r.-", label="truncated")
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("y (m)")

    # Draw interpolation lines
    line_colors = ["g", "c", "b"]
    for i_type, interp_type in enumerate(params["interp_type"]):
        assert interp_type in ["interp1d", "interpSpline", "interpGP"]
        lc = line_colors[i_type]
        ax2.plot(
            new_timestamps[idx_valid],
            new_positions_vis[interp_type][:, 0],
            c=lc,
            marker=".",
            ls="--",
            label=interp_type,
        )
        ax4.plot(
            new_timestamps[idx_valid],
            new_positions_vis[interp_type][:, 1],
            c=lc,
            marker=".",
            ls="--",
            label=interp_type,
        )

    # Set titles and axis scale
    ax1.set_title("Remove duplicate points")
    ax2.set_title("Truncate and interpolation")
    ax2.legend(frameon=False)
    xmin, xmax, ymin, ymax = ax2.axis()
    yc = (ymin + ymax) / 2
    ymin = min(yc - 2, ymin)
    ymax = max(yc + 2, ymax)
    ax2.axis([xmin, xmax, ymin, ymax])

    ax4.legend(frameon=False)
    xmin, xmax, ymin, ymax = ax4.axis()
    yc = (ymin + ymax) / 2
    ymin = min(yc - 2, ymin)
    ymax = max(yc + 2, ymax)
    ax4.axis([xmin, xmax, ymin, ymax])

    plt.tight_layout()
    fig.savefig(png_file, dpi=200)
    plt.close(fig)


def interpolate_bbox(bboxes, timestamps_bbox, timestamps_img, bbox_dilate_scale):
    """
    :param bboxes: a list of bounding boxes with each [xmin, ymin, width, height]
    :param timestamps_bbox: the corresponding timestamps of bounding boxes
    :param timestamps_img: the timestamps of global images
    :return: new bounding boxes w.r.t timestamps_img
            numpy array, size (N_timestamps_img, 4), [xmin, ymin, width, height]
    """
    timestamps_bbox = np.asarray(timestamps_bbox, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    # remove outlier detections with value [0, 0, width, height]
    valid_idx = np.logical_or(bboxes[:, 0] != 0, bboxes[:, 1] != 0)
    bboxes = bboxes[valid_idx, :]
    timestamps_bbox = timestamps_bbox[valid_idx]
    # remove extra detections at the same timestamp
    uniq_timestamps_bbox, uniq_idx = np.unique(timestamps_bbox, return_index=True)
    bboxes = bboxes[uniq_idx, :]
    timestamps_img = np.asarray(timestamps_img, dtype=np.float32)
    # If there is only one bounding box, just duplicate it
    if bboxes.shape[0] == 1:
        new_bboxes = np.tile(bboxes, (len(timestamps_img), 1))
        return new_bboxes
    # Do interpolation/extrapolation based on the centers of bounding boxes
    center_x = bboxes[:, 0] + 0.5 * bboxes[:, 2]
    center_y = bboxes[:, 1] + 0.5 * bboxes[:, 3]
    f_x = interpolate.interp1d(uniq_timestamps_bbox, center_x, kind="linear", fill_value="extrapolate")
    new_center_x = f_x(timestamps_img)
    f_y = interpolate.interp1d(uniq_timestamps_bbox, center_y, kind="linear", fill_value="extrapolate")
    new_center_y = f_y(timestamps_img)
    # Based on the median of all bounding boxes, dilate the width and height by a scale
    width = np.median(bboxes[:, 2]) * bbox_dilate_scale
    height = np.median(bboxes[:, 3]) * bbox_dilate_scale
    # Compute the new bounding box [xmin, ymin, width, height], output size (N, 4)
    new_xmin = new_center_x - 0.5 * width
    new_ymin = new_center_y - 0.5 * height
    new_width = np.repeat(width, len(new_xmin))
    new_height = np.repeat(height, len(new_xmin))
    new_bboxes = np.transpose(np.vstack((new_xmin, new_ymin, new_width, new_height)))

    if np.isinf(new_bboxes).any() or np.isnan(new_bboxes).any():
        from IPython import embed

        embed(header="nan/inf in interpolate_bbox")
    return new_bboxes


def interpolate_nearest_timestamp(timestamps, params):
    """
    :param timestamps: all timestamps (relative to t_now)
    :param params:
    :return: indices of the nearest timestamps for new timestamps
    """
    past_timesteps = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS]
    future_timesteps = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS]
    past_dt = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE]
    future_dt = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE]
    new_past_timestamps = np.arange((-past_timesteps + 1) * past_dt, 0, past_dt)
    new_future_timestamps = np.arange(0, (future_timesteps + 0.5) * future_dt, future_dt)
    new_timestamps = np.concatenate([new_past_timestamps, new_future_timestamps])
    # import IPython;IPython.embed(header='{}'.format(new_timestamps))

    f = interpolate.interp1d(timestamps, range(len(timestamps)), kind="nearest", fill_value="extrapolate")
    idx_images = f(new_timestamps)
    return idx_images


def select_agent(
    params: dict,
    positions: np.ndarray,
    is_ego_vehicle: np.ndarray,
    is_relevant_agent: np.ndarray,
    dot_keys: np.ndarray,
    agent_type_vector: np.ndarray,
    additional_inputs: list,
    trajectories: list,
    ego_only: bool = False,
    skip_ego: bool = False,
    filename=None,
):
    """Select agents near to ego-vehicle and/or to relevant agents.

    Parameters
    ----------
    params : dict
        A dictionary of parameters.
    positions : np.ndarray
        Positions of agents, with shape [n_timestamps, n_agents, 3].
    is_ego_vehicle : np.ndarray
        Mask indicating which agent is ego vehicle, with shape [n_agents].
    is_relevant_agent : np.ndarray
        Mask indicating which agents are relevant vehicles, with shape [n_agents].
    dot_keys : np.ndarray
        Dot keys of each agent, with shape [n_agents].
    agent_type_vector : np.ndarray
        Agent types, with shape [n_agents, n_agent_types].
    additional_inputs: list
        Additional inputs used by downstream processes [n_agents]
    trajectories: list
         Agent trajectories used by downstream processes [n_agents]
    ego_only : bool
        Whether to filter by distances to ego agent only.
    skip_ego: bool
        Whether to filter by distances to relevant agents only.

    Returns
    -------
    new_positions : np.ndarray
        Positions of agents where non-ego and non-relevant entries are 0s, with shape [n_timestamps, max_agents, 3].
    new_is_ego_vehicle : np.ndarray
        Updated mask indicating which agent is ego vehicle, with shape [max_agents].
    new_is_relevant_agent : np.ndarray
        Updated mask indicating which agents are relevant vehicles, with shape [max_agents].
    new_dot_keys : np.ndarray
        Updated dot keys of each agent, with shape [max_agents].
    new_agent_type_vector : np.ndarray
        Updated agent types, with shape [max_agents, n_agent_types].
    new_additional_inputs : list
        Updated additional inputs [max_agents]
    new_idx : np.ndarray
        Selected agent indices, with shape [num_ego_agent + num_relevant_agent].
    """
    if not (positions.shape[1] == is_ego_vehicle.size == is_relevant_agent.size):
        import IPython

        IPython.embed(header="inconsistent size of agents in select_agent")

    total_timestamps = params[PARAM_PAST_TIMESTEPS] + params[PARAM_FUTURE_TIMESTEPS]
    new_positions = np.zeros(
        (total_timestamps, params[protobuf_training_parameter_names.PARAM_MAX_AGENTS], positions.shape[2])
    )
    new_is_ego = np.zeros(params[protobuf_training_parameter_names.PARAM_MAX_AGENTS])
    new_is_rel = np.zeros(params[protobuf_training_parameter_names.PARAM_MAX_AGENTS])
    new_dot_keys = np.zeros(params[protobuf_training_parameter_names.PARAM_MAX_AGENTS])
    new_agent_type_vector = np.zeros(
        (params[protobuf_training_parameter_names.PARAM_MAX_AGENTS], len(params["agent_types"]))
    )
    num_agent = positions.shape[1]

    # Get index of ego agent.
    # Assume there exists only one ego agent.
    if not skip_ego:
        try:
            idx_ego = np.nonzero(is_ego_vehicle)[0].item()
        except:
            import sys

            import IPython

            tb = IPython.core.ultratb.VerboseTB()
            print(tb.text(*sys.exc_info()))
            # IPython.embed(header="index of ego-vehicle is not unique")
            raise ProtobufDatasetParsingError(
                f"index of ego-vehicle is not unique, is_ego_vehicle: {str(is_ego_vehicle)}"
            )

    # Get index/indices of relevant agent(s).
    if not ego_only:
        idx_rel = np.nonzero(is_relevant_agent)[0]

    # Special case of max_k = 1.
    if params[PARAM_MAX_AGENTS] == 1 and num_agent >= 1:
        if not skip_ego:
            new_idx = idx_ego
            new_is_ego[0] = 1
            new_is_rel[0] = 0
        else:
            new_idx = idx_rel[0]
            new_is_ego[0] = 0
            new_is_rel[0] = 1

        new_positions[:, 0, :] = positions[:, new_idx, :]
        new_dot_keys[0] = dot_keys[new_idx]
        new_agent_type_vector[0, :] = agent_type_vector[new_idx, :]

    # New indices of agents: ego vehicle (if exists), followed by relevant agent(s), followed by other agents.
    elif num_agent <= params[PARAM_MAX_AGENTS]:
        if ego_only:
            idx_others = np.setdiff1d(np.arange(num_agent), idx_ego, assume_unique=True)
            new_idx = np.hstack((idx_ego, idx_others))
        elif skip_ego:
            idx_others = np.setdiff1d(np.arange(num_agent), idx_rel, assume_unique=True)
            new_idx = np.hstack((idx_rel, idx_others))
        else:
            idx_others = np.setdiff1d(np.arange(num_agent), np.hstack((idx_ego, idx_rel)), assume_unique=True)
            # Remove idx_ego from idx_rel to prevent redundancies,
            # as we assume idx_ego can be included in idx_rel.
            if idx_ego in idx_rel:
                idx_rel = np.setdiff1d(idx_rel, idx_ego, assume_unique=True)
                new_idx = np.hstack((idx_ego, idx_rel, idx_others))
            else:
                new_idx = np.hstack((idx_ego, idx_rel, idx_others))
        new_positions[:, :num_agent, :] = positions[:, new_idx, :]
        new_is_ego[:num_agent] = is_ego_vehicle[new_idx]
        new_is_rel[:num_agent] = is_relevant_agent[new_idx]
        new_dot_keys[:num_agent] = dot_keys[new_idx]
        new_agent_type_vector[:num_agent, :] = agent_type_vector[new_idx, :]

    # Select a subset of agents that are close to ego agent or relevant agent(s).
    else:
        # Get distance to ego agent and indices of closest agents.
        if ego_only:
            distances_to_ego = (((positions[..., :2] - positions[:, [idx_ego], :2]) * positions[..., [2]]) ** 2).sum(
                axis=2
            ).sum(axis=0) / (positions[..., 2].sum(axis=0) + 0.0001)
            idx_to_ego = np.argsort(distances_to_ego)

            idx_selected = idx_to_ego[: params[PARAM_MAX_AGENTS]]
            idx_selected = np.setdiff1d(idx_selected, idx_ego, assume_unique=True)
            new_idx = np.int64(np.hstack((idx_ego, idx_selected)))

            new_is_ego[0] = 1

        # Get distance to relevant agent and indices of closest agents.
        elif skip_ego:
            rel_pos_center = np.mean(positions[:, idx_rel, :2], 1, keepdims=True)
            distances_to_rel = (((positions[:, :, :2] - rel_pos_center) * positions[:, :, [2]]) ** 2).sum(axis=2).sum(
                axis=0
            ) / (positions[:, :, 2].sum(axis=0) + 0.0001)
            idx_to_rel = np.argsort(distances_to_rel)

            idx_to_rel = np.setdiff1d(idx_to_rel, idx_rel, assume_unique=True)
            new_idx = np.int64(np.hstack((idx_rel, idx_to_rel)))[: params[PARAM_MAX_AGENTS]]

            num_agent = new_idx.shape[0]
            new_is_rel[:num_agent] = is_relevant_agent[new_idx]

        # Get distance to ego and relevant agents and indices of closest agents.
        else:
            distances_to_ego = (((positions[..., :2] - positions[:, [idx_ego], :2]) * positions[..., [2]]) ** 2).sum(
                axis=2
            ).sum(axis=0) / (positions[..., 2].sum(axis=0) + 0.0001)
            idx_to_ego = np.argsort(distances_to_ego)

            rel_pos_center = np.mean(positions[:, idx_rel, :2], 1, keepdims=True)
            distances_to_rel = (((positions[:, :, :2] - rel_pos_center) * positions[:, :, [2]]) ** 2).sum(axis=2).sum(
                axis=0
            ) / (positions[:, :, 2].sum(axis=0) + 0.0001)
            idx_to_rel = np.argsort(distances_to_rel)

            # idx_selected_part1 will contain the indices of agents closest to the ego-vehicle.
            # idx_selected_part2 will contain the indices of agents closest to the relevant agent.
            idx_selected_part1 = idx_to_ego[: params[PARAM_MAX_AGENTS] // 2]
            if np.any(idx_selected_part1 == idx_rel):
                idx_selected_part1 = np.setdiff1d(idx_selected_part1, idx_rel, assume_unique=True)
                idx_selected_part1 = np.hstack((idx_selected_part1, idx_to_ego[params[PARAM_MAX_AGENTS] // 2]))

            idx_complement = np.setdiff1d(idx_to_rel, idx_selected_part1, assume_unique=True)
            num_part2 = params[PARAM_MAX_AGENTS] - params[PARAM_MAX_AGENTS] // 2
            idx_selected_part2 = idx_complement[:num_part2]
            idx_selected_part1 = np.setdiff1d(idx_selected_part1, idx_ego, assume_unique=True)
            idx_selected_part2 = np.setdiff1d(idx_selected_part2, idx_rel, assume_unique=True)
            idx_rel_ped_if_any = idx_rel if params[PARAM_MAX_AGENTS] > 1 else []
            new_idx = np.int64(np.hstack((idx_ego, idx_rel_ped_if_any, idx_selected_part1, idx_selected_part2)))

            # Cap new index at the maximum number of agents.
            new_idx = new_idx[: params[PARAM_MAX_AGENTS]]

        new_is_ego = is_ego_vehicle[new_idx]
        new_is_rel = is_relevant_agent[new_idx]

        new_positions = positions[:, new_idx, :]
        new_dot_keys = dot_keys[new_idx]
        new_agent_type_vector = agent_type_vector[new_idx, :]

    new_is_ego = np.int64(new_is_ego)
    new_is_rel = np.int64(new_is_rel)
    new_idx = np.int64(new_idx)
    if isinstance(new_idx, Iterable):
        new_trajectories = [trajectories[i] for i in cast(np.ndarray, new_idx)]
        new_additional_types = [additional_inputs[i] for i in cast(np.ndarray, new_idx)]
    else:
        new_trajectories = [trajectories[new_idx]]
        new_additional_types = [additional_inputs[new_idx]]

    # TODO replace with unit test
    assert len(new_is_ego) == params[protobuf_training_parameter_names.PARAM_MAX_AGENTS]
    assert len(new_is_rel) == params[protobuf_training_parameter_names.PARAM_MAX_AGENTS]
    assert len(new_dot_keys) == params[protobuf_training_parameter_names.PARAM_MAX_AGENTS]
    # assert len(new_additional_types) == params[
    #     protobuf_training_parameter_names.PARAM_MAX_AGENTS], f'filename: {filename}'

    return (
        new_positions,
        new_is_ego,
        new_is_rel,
        new_dot_keys,
        new_agent_type_vector,
        new_additional_types,
        new_trajectories,
        new_idx,
    )


class InputsHandler(ABC):
    def __init__(self, params):
        # self.main_param_hash = params["main_param_hash"]
        self.cache_dir = None
        self.cache_read_only = None

        # This is polymorphic, to get the hash of handler's parameter
        hash_params = self._get_params_for_hash(params)
        self.handler_param_hash = compute_hash(str(hash_params))
        # Create a cache folder for this handler class.
        self.cache_dir = os.path.join(params["cache_dir"], self.__class__.__name__, f"param_{self.handler_param_hash}")
        self.cache_read_only = params["cache_read_only"]
        self.use_cache_lock = params["use_cache_lock"]
        self.disable_cache = params["disable_cache"]

        # Dump handler params to file
        if not self.cache_read_only:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(os.path.join(self.cache_dir, "handler_params.json"), "w") as f:
                json.dump(hash_params, f, indent=2)

    @abstractmethod
    def _process_impl(
        self,
        result_dict: dict,
        params: dict,
        filename: str,
        index: int,
    ) -> dict:
        """Implement this function to do actually processing of the handler.

        Parameters
        ----------
        result_dict : dict
            The output of __getitem__ in ProtobufPredictionDataset.
            DO NOT use the this dict to store result. Always return a new dict
        instance : triceps.protobuf.prediction_training_pb2.PredictionInstance
            The corresponding profobuf instance.
        params : dict
            Command line arguments.
        filename : str
            The name of the current protobuf file.
        index : int
            The index of the element in the corresponding dataset class.

        Returns
        -------
            A new dict contains only the result from this handler.
            DO NOT use the result_dict from input, always make a new dict to store result.
        """
        return {}

    def process(
        self,
        result_dict: dict,
        params: dict,
        filename: str,
        index: int,
        cache_id: str = "",
    ) -> dict:
        """This function is used to wrap the _process_impl() and add caching around it.
        If there could be multiple cached files for one instance, override this function and provide cache_id,
            which is used to uniquely define the cache files under the same instance.

        Parameters
        ----------
        result_dict : dict
            The output of __getitem__ in ProtobufPredictionDataset.
        params : dict
            Command line arguments.
        filename : str
            The name of the current protobuf file.
        index : int
            The index of the element in the corresponding dataset class.
        cache_id : str
            Should be empty by default.
            Specify an unique cache id when one instance has multiple cache files.
        Returns
        -------
            A dict of the result data.
        """
        # Note: cache_is currently unused but is being preserved for future possible needs.
        data = self._load_cache(filename, index, cache_id)
        if data is None:
            # Load data
            data = self._process_impl(result_dict, params, filename, index)
            assert data is not result_dict, (
                "The return from _process_impl() must be a new dict containing only result from this handler,"
                " don't modify result_dict."
            )
            # Cache data
            self._save_cache(data, filename, index, cache_id)

        result_dict.update(data)
        return result_dict

    @abstractmethod
    def get_hash_param_keys(self) -> List[str]:
        """Return the parameter keys used for hashing

        Returns
        -------
        hash_param keys: list[str]
        """
        return []

    def _get_params_for_hash(self, params: dict) -> dict:
        """Return the params used for hashing, to get a unique ID for the parameters of the cached data.
        Override this function to do post process some parameters,
            for example, absolute file path, which is not transferable.
        """
        d = {k: params[k] for k in self.get_hash_param_keys()}
        return dict(sorted(d.items()))

    def _save_cache(self, output_dict: dict, filename: str, index: int, cache_id):
        """Save the data result from this handler."""
        self._get_cache(filename, index, cache_id).save(output_dict)

    def _load_cache(self, filename: str, index: int, cache_id):
        """Load the data result from this handler."""
        data = self._get_cache(filename, index, cache_id).load()
        return data

    def _get_cache(self, filename: str, index: int, cache_id):
        cache_name = compute_hash(f"file-{filename}-{index}-{cache_id}")
        cache_element = CacheElement(
            self.cache_dir,
            cache_name,
            "pkl",
            should_lock=self.use_cache_lock,
            read_only=self.cache_read_only,
            disable_cache=self.disable_cache,
        )
        return cache_element


def read_sensor_image(
    inp: dict,
    params: dict,
    filename: str,
    index: int,
    height: int,
    width: int,
    img_dir: list,
    img_dir_prefix: str,
    transform: transforms,
    crop: Optional[list] = None,
    cache_folder="img",
    alternate_extension=None,
) -> np.array:
    """

    Parameters
    ----------
    inp: additional input prediction protobuf item
      The additional input with the image specification.
    params: dict
      The parameters dictionary.
    filename: str
      The prediction protobuf filename
    index: int
      Prediction instance index.
    height: int
      Image height in pixels.
    width: int
      Image width in pixels.
    img_dir: list
      A list of image folders.
    img_dir_prefix: str
      Path prefix for img_dir, cf. params["image_dir_prefix"]
    transform: torchvision image transforms
      torchvision image transforms for the image.
    crop: 4-tuple or list
      [xmin, ymin, width, height] for a bounding box crop. If None, no cropping occurs.
    Returns
    -------
    img: numpy array
      loaded image, or None if failed to load.
    """
    read_stats = {}
    if "filename" in inp["sensorImageInput"]:
        try:
            img_data = json.loads(inp["sensorImageInput"]["filename"])
        except:
            msg = "failed parsing: " + str(inp["sensorImageInput"]["filename"])
            print(msg)
            import IPython

            IPython.embed()
            raise Exception(msg)

        frame_filename = os.path.splitext(img_data["filename"])[0]
        if alternate_extension is not None:
            img_data["filename"] = f"{frame_filename}.{alternate_extension}"

        crop_str = "" if crop is None else str(crop.astype(int))
        cache_id = hashlib.md5((img_data["folder"] + frame_filename + crop_str + str(transform)).encode()).hexdigest()
        cache = CacheElement(
            cache_folder,
            cache_id,
            "jpg",
            should_lock=params["use_cache_lock"],
            read_only=params["cache_read_only"],
            disable_cache=params["disable_cache"],
        )
        cached_data = cache.load()
        if cached_data is not None:
            try:
                img_ = cached_data
                img = np.asarray(img_, dtype=np.float32).transpose([2, 0, 1]) / 255.0
            except:
                if params["use_cache_lock"]:
                    cache.remove_from_cache()
                else:
                    raise ValueError("--use-cache-lock is set to false.. trying to remove/fix a pre-made image cache?")
                img = np.zeros([3, height, width], dtype=np.float32)

        else:
            # No cache available
            if img_dir is not None:
                img_path = get_image_path(
                    params,
                    img_dir,
                    img_dir_prefix,
                    img_data["folder"],
                    img_data["filename"],
                    params["image_list_key_num"],
                    alternate_extension=alternate_extension,
                )
                if img_path is None:
                    print("img_path", img_path)

                    # Image missing.
                    if params["silent_fail_on_missing_images"]:
                        img = np.zeros([3, height, width], dtype=np.float32)
                    else:

                        img_path = get_image_path(
                            params,
                            img_dir,
                            img_dir_prefix,
                            img_data["folder"],
                            img_data["filename"],
                            params["image_list_key_num"],
                            alternate_extension=alternate_extension,
                            verbose=True,
                        )
                        print("img_path2", img_path)

                        # TODO(guy.rosman) is this missing
                        err_str = (
                            "Trying to load image, but missing img folder {} for img {}. Filename: {}, "
                            "If images are not needed, set --scene-image-mode=none.".format(
                                str(img_dir), img_data["filename"], filename
                            )
                        )
                        raise ProtobufDatasetParsingError(err_str)
                elif not os.path.exists(img_path):
                    err_str = f"Error loading image: {img_data['filename']}, but it doesn't exist at {img_path}"
                    raise ProtobufDatasetParsingError(err_str)
                else:
                    try:
                        with PIL.Image.open(img_path) as img_:
                            img_ = img_.convert("RGB")
                            if crop is not None:
                                try:
                                    # crop:[xmin, ymin, width, height]
                                    top = int(crop[1])
                                    left = int(crop[0])
                                    height = int(crop[3])
                                    width = int(crop[2])
                                except OverflowError as err:
                                    # TODO(guy.rosman): Check if we still get overflow errors. Remove if not needed.
                                    import IPython

                                    IPython.embed(header=str([err, filename, index]))
                                img_ = transforms.functional.crop(img_, top, left, height, width)
                                read_stats["height"] = height
                                read_stats["width"] = width
                            if transform:
                                img = transform(img_)
                            else:
                                import IPython

                                IPython.embed()
                                # The default transform (at least Resize()) should be defined in "train_trajectory_prediction.py"
                                raise Exception("Image transform is None.")
                    except OSError as err:
                        err = "Failed with an OS error when opening image {}:\n{}".format(img_path, err)
                        logging.error(err)
                        raise ProtobufDatasetParsingError(err)
                    except AttributeError as err2:
                        import IPython

                        IPython.embed()
            else:
                # TODO(guy.rosman) is this missing
                raise Exception("Missing img folder, yet images are loaded. Set --scene-image-mode=none.")
            if not type(img).__name__ == "ndarray":
                img = img.numpy()
            cache.save(img.transpose([1, 2, 0]) * 255)
    elif "rawData" in inp["sensorImageInput"]:
        # TODO(guy.rosman): implement Usage for a specific type once there's data to test on.
        raw_data = inp["sensorImageInput"]["rawData"]["data"]
        data_img_width = inp["sensorImageInput"]["rawData"]["width"]
        data_img_height = inp["sensorImageInput"]["rawData"]["height"]
        img = None
    return img, read_stats


class ImageHandler(InputsHandler):
    def process(
        self,
        result_dict: dict,
        params: dict,
        filename: str,
        index: int,
        cache_id: str = "",
    ) -> dict:
        """For image handlers, skip the default caching behavior, because the images are cached one level below."""
        return self._process_impl(result_dict, params, filename, index)


class GlobalImageHandler(ImageHandler):
    """Initialize a global image handler -- reads the images and adds to an input tensor.

    Parameters
    ----------
    img_dir: list
      A list of possible image folders.
    height: int
      image height.
    width: int
      image width.
    total_timesteps : int
        Number of timesteps per datapoint / scene (including past and future).
    timepoints :  list, optional
        List of time indices for which agent images are loaded. The permitted
        range is between 0 and (past_timepoints + future_timepoints - 1). All
        timepoints are used if this is not set.
    transform: callable
      An image transform.
    image_processor: class
      A functor class to process an image into a feature vector. See ImageProcessor for example.
    """

    def __init__(
        self,
        params,
        img_dir: int,
        height: int,
        width: int,
        total_timesteps: int,
        timepoints: Optional[List[int]] = None,
        transform=None,
        image_processor=None,
    ) -> None:
        self.use_semantic_masks = params["use_scene_semantic_masks"] or False  # Must be before init
        super().__init__(params)
        self.img_dir = img_dir
        self.height = height
        self.width = width
        self.total_timesteps = total_timesteps
        self.timepoints = timepoints if timepoints is not None else list(range(self.total_timesteps))
        self.max_images = len(self.timepoints)
        self.transform = transform
        self.image_processor = image_processor
        if self.image_processor is not None:
            self.image_processor_dim = image_processor.output_dim

        if params["use_scene_semantic_masks"]:
            self.channels = 6
        else:
            self.channels = 3

    def get_hash_param_keys(self) -> List[str]:
        param_keys = [
            "scene_image_mode",
            "scene_image_timepoints",
            "img_height",
            "img_width",
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE,
        ]
        if self.use_semantic_masks:
            # Do not require cache rebuild for masks
            param_keys.append("use_scene_semantic_masks")
        return param_keys

    def _process_impl(
        self,
        result_dict: dict,
        params: dict,
        filename: str,
        index: int,
    ) -> dict:
        """Adds scene images to the datapoints dictionary

        Parameters
        ----------
        result_dict : dict
            The output of __getitem__ in ProtobufPredictionDataset.
        params : dict
            Command line arguments.
        filename : str
            The name of the current protobuf file.
        index : int
            The index of the element in the corresponding dataset class.

        Returns
        -------
        dict
            An updated version of the results dictionary containing the
            following new keys (and all of the existing ones):
            * "images": Numpy ndarray containing the images / processed
              embeddindgs. Its shape is (max_images, 3, height, width) if the
              class was constructed without an image_processor and (max_images,
              processor_output_dim) if there is one. Not existing images are
              filled with 0.
            * "images_mapping": numpy integer array of shape
              (max_images,) containing time index of the image. The location in the
              "agent_images" dict entry. For unused image slots, both values are
              set to -1.
        """
        images_mapping = -1 * np.ones(self.max_images, dtype=np.int64)
        if self.image_processor is None:
            images = np.zeros((self.max_images, self.channels, self.height, self.width), dtype=np.float32)
        else:
            images = np.zeros((self.max_images, self.image_processor_dim), dtype=np.float32)

        sub_dir_name = filename.split("/")[-2]

        if result_dict[ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS] is None:
            result_dict[ProtobufPredictionDataset.DATASET_KEY_IMAGES] = images
            result_dict[ProtobufPredictionDataset.DATASET_KEY_IMAGES_MAPPING] = images_mapping
            logging.warning(f"image file name is not supplied for {filename}")

            return result_dict

        prediction_timestamp = result_dict[ProtobufPredictionDataset.DATASET_KEY_PREDICTION_TIMESTAMP]

        additional_inputs = []
        for i, is_ego_vehicle in enumerate(result_dict[ProtobufPredictionDataset.DATASET_KEY_IS_EGO_VEHICLE]):
            if is_ego_vehicle:
                additional_inputs = result_dict[ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS][i]
                break

        if additional_inputs:
            timestamps_img = []
            timestamps_idx = []
            for inp_i, inp in enumerate(additional_inputs):
                if "sensorImageInput" not in inp or len(inp["sensorImageInput"]["filename"]) == 0:
                    continue

                timestamps_img.append(
                    parse_protobuf_timestamp(inp["timestamp"]).ToNanoseconds() / 1e9 - prediction_timestamp
                )
                timestamps_idx.append(inp_i)

            if len(timestamps_img) > 0:
                idx_images = interpolate_nearest_timestamp(timestamps_img, params)
                assert self.total_timesteps == len(
                    idx_images
                ), "Number of interpolated images is not equal to total_timesteps"

                cnt = 0
                img_idx = 0
                bbox = (0, 0, self.width, self.height)  # Dummy bounding box that encompasses the full image
                for inp_i in idx_images:
                    if cnt not in self.timepoints:
                        cnt += 1
                        continue

                    inp = additional_inputs[timestamps_idx[int(inp_i)]]
                    img, _ = read_sensor_image(
                        inp,
                        params,
                        filename,
                        index,
                        self.height,
                        self.width,
                        self.img_dir,
                        params["image_dir_prefix"],
                        self.transform,
                        cache_folder=os.path.join(self.cache_dir, sub_dir_name),
                    )

                    if params["use_scene_semantic_masks"]:
                        # TODO(nicholas.guyett.ctr) Can we de-duplicate this between scene and agent images?
                        semantic_mask, _ = read_sensor_image(
                            inp,
                            params,
                            filename,
                            index,
                            self.height,
                            self.width,
                            [
                                img_dir.replace(params["image_dir_prefix"], params["mask_dir_prefix"])
                                for img_dir in self.img_dir
                            ],
                            params["mask_dir_prefix"],
                            self.transform,
                            cache_folder=os.path.join(self.cache_dir, f"{sub_dir_name}-mask"),
                            alternate_extension="png",
                        )
                        img = np.concatenate((img, semantic_mask), axis=0)

                    images_mapping[img_idx] = cnt
                    if self.image_processor is None:
                        images[img_idx, :, :, :] = img
                    else:
                        images[img_idx, :] = self.image_processor._process_impl(img, bbox)

                    img_idx += 1
                    cnt += 1
        else:
            logging.warning(f"image file name is not supplied for {filename}")

        result_dict[ProtobufPredictionDataset.DATASET_KEY_IMAGES_MAPPING] = images_mapping
        result_dict[ProtobufPredictionDataset.DATASET_KEY_IMAGES] = images
        return result_dict


class AgentImageHandler(ImageHandler):
    """Handler for integrating per-agent images.

    Parameters
    ----------
    img_dir : str or None
    height, width : int
        Desired height and width of the outputted agent images.
    max_agents: int
        Maximum number of agents.
    total_timesteps : int
        Number of timesteps per datapoint / scene (including past and future).
    agents : list, optional
        List of agent indexes for which agent images are loaded. The permitted
        range is between 0 and (num_agents-1). All agents are used if this is
        not set.
    timepoints :  list, optional
        List of time indices for which agent images are loaded. The permitted
        range is between 0 and (past_timepoints + future_timepoints - 1). All
        timepoints are used if this is not set.
    transform : transforms.Compose, optional
        Optional torchvision image transforms
    image_processor : ImageProcessor, optional
        Additional custom image processing routines.
    """

    # TODO(igor.gilitschenski): Currently, we inject some of the command line parameters as arguments to the constructor
    # TODO  whereas others are used as part of the params dictionary in process(). This is inconsistent and we
    # TODO  should unify it for all handlers.
    def __init__(
        self,
        params,
        img_dir: Optional[List[str]],
        height: int,
        width: int,
        max_agents: int,
        total_timesteps: int,
        agents: Optional[List[int]] = None,
        timepoints: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        image_processor: Optional[ImageProcessor] = None,
    ) -> None:
        self.use_semantic_masks = params["use_agent_semantic_masks"] or False  # Must be before init
        self.use_pose_estimates = params["use_agent_pose_estimates"] or False  # Must be before init
        super().__init__(params)
        self.img_dir = img_dir
        self.height = height
        self.width = width
        self.max_agents = max_agents
        self.total_timesteps = total_timesteps
        self.agents = agents if agents is not None else list(range(self.max_agents))
        self.timepoints = timepoints if timepoints is not None else list(range(self.total_timesteps))
        self.transform = transform
        self.image_processor = image_processor
        if self.image_processor is not None:
            self.image_processor_dim = image_processor.output_dim

        self.channels = 3
        if self.use_semantic_masks:
            self.channels += 3
        if self.use_pose_estimates:
            self.channels += 3

    def get_hash_param_keys(self) -> List[str]:
        param_keys = [
            "agent_image_mode",
            "agent_image_agents",
            "agent_image_timepoints",
            "agent_image_processor",
            "agent_img_width",
            "agent_img_height",
            "require_agent_images",
            "bbox_dilate_scale",
            "agent_image_processor",
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE,
        ]
        # Do not require cache rebuild for masks, poses
        if self.use_semantic_masks:
            param_keys.extend(["use_agent_semantic_masks", "agent_semantic_mask_padding_ratio"])
        if self.use_pose_estimates:
            param_keys.append("use_agent_pose_estimates")
        return param_keys

    def _process_impl(
        self,
        result_dict: dict,
        params: dict,
        filename: str,
        index: int,
    ) -> dict:
        """Adds agent images to the datapoints dictionary

        The size of the image tensor is determined automatically based on the
        desired / maximum possible number of images / timepoints.

        Parameters
        ----------
        result_dict : dict
            The output of __getitem__ in ProtobufPredictionDataset.
        params : dict
            Command line arguments.
        filename : str
            The name of the current protobuf file.
        index : int
            The index of the element in the corresponding dataset class.

        Returns
        -------
        dict
            An updated version of the results dictionary containing the
            following new keys (and all of the existing ones):
            * "agent_images": Numpy ndarray containing the images. Its shape is
              (max_images, 3, height, width). Not existing images are filled
              with 0.
            * "agent_images_mapping": numpy integer array of shape
              (max_images, 2) containing agent index and time index of the
              image. The first dimension corresponds to the location in the
              "agent_images" dict entry. For unused image slots, both values are
              set to -1.
            * "agent_image_stats": String with additional information about the
              image in json format (currently width & height).
        """
        stats = {}
        stats["width"] = []
        stats["height"] = []
        sub_dir_name = filename.split("/")[-2]

        max_total_images = len(self.agents) * len(self.timepoints)

        cur_img_idx = 0
        if self.image_processor is None:
            images = np.zeros((max_total_images, self.channels, self.height, self.width), dtype=np.float32)
        else:
            images = np.zeros((max_total_images, self.image_processor_dim), dtype=np.float32)

        images_mapping = -1 * np.ones((max_total_images, 2), dtype=np.int64)

        if result_dict[ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS]:
            prediction_timestamp = result_dict[ProtobufPredictionDataset.DATASET_KEY_PREDICTION_TIMESTAMP]

            # get bounding box information from additional inputs of the relevant agent
            # TODO(rui.yu) support multiple sequences of bounding boxes (multiple relevant agents)
            bboxes = {}
            for k, additional_inputs in zip(
                result_dict[ProtobufPredictionDataset.DATASET_KEY_DOT_KEYS],
                result_dict[ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS],
            ):
                timestamps_bbox = []
                if additional_inputs:
                    for inp in additional_inputs:
                        if "vectorInputTypeId" in inp and (
                            inp["vectorInputTypeId"]
                            in (
                                ADDITIONAL_INPUT_KEY_VISUAL_BBOX,
                                # for backward compatibility, tlog harvested before Aug 2021 is using 'pa_bbox'
                                "additional_input_pa_bbox",
                            )
                        ):
                            if k not in bboxes:
                                bboxes[k] = []

                            timestamps_bbox.append(
                                parse_protobuf_timestamp(inp["timestamp"]).ToNanoseconds() / 1e9 - prediction_timestamp
                            )
                            bboxes[k].append(inp["vectorInput"])
                else:
                    continue
                if len(bboxes) > 0:
                    break

            additional_inputs_ev = []
            for i, is_ego_vehicle in enumerate(result_dict[ProtobufPredictionDataset.DATASET_KEY_IS_EGO_VEHICLE]):
                if is_ego_vehicle:
                    additional_inputs_ev = result_dict[ProtobufPredictionDataset.DATASET_KEY_ADDITIONAL_INPUTS][i]

            timestamps_img = []
            timestamps_idx = []
            # get image information from additional inputs of ego-vehicle (ev)
            if additional_inputs_ev:
                for inp_i, inp in enumerate(additional_inputs_ev):
                    if "sensorImageInput" not in inp or len(inp["sensorImageInput"]["filename"]) == 0:
                        continue
                    timestamps_img.append(
                        parse_protobuf_timestamp(inp["timestamp"]).ToNanoseconds() / 1e9 - prediction_timestamp
                    )
                    timestamps_idx.append(inp_i)
                bbox_condition = len(timestamps_img) > 0 and len(bboxes) > 0
                # params['require_agent_images'] allow to skip images when they are not available.
                if not bbox_condition and params["require_agent_images"]:
                    import IPython

                    IPython.embed(header="Require agent images, but missing")
                    raise Exception("Missing agent images")
                elif bbox_condition:
                    idx_images = interpolate_nearest_timestamp(timestamps_img, params)
                    assert self.total_timesteps == len(
                        idx_images
                    ), "Number of interpolated images is not equal to max_t"
                    new_timestamps_img = [timestamps_img[int(i)] for i in idx_images]
                    assert len(bboxes) > 0, "bboxes are empty. Set --agent-image-mode=none for old data."
                    new_bboxes = {}
                    for k in bboxes:
                        new_bboxes[k] = interpolate_bbox(
                            bboxes[k], timestamps_bbox, new_timestamps_img, params["bbox_dilate_scale"]
                        )

                    # TODO(igor.gilitschenski): Check if this is the right location for cnt or if it should be inside
                    # TODO the for loop.This seems to be unproblematic for now as we only have one agent image per scene
                    cnt = 0
                    for agent_key in new_bboxes:
                        for i, inp_i in enumerate(idx_images):
                            try:
                                agent_idx = np.where(
                                    result_dict[ProtobufPredictionDataset.DATASET_KEY_DOT_KEYS] == float(agent_key)
                                )[0]
                            except:
                                import IPython

                                IPython.embed(header="Could not match DOT key in AgentImageHandler")

                            try:
                                if not (agent_idx in self.agents and cnt in self.timepoints):
                                    cnt += 1
                                    continue
                            except ValueError as e:
                                raise e
                            inp = additional_inputs_ev[timestamps_idx[int(inp_i)]]
                            crop = new_bboxes[agent_key][i, :]
                            # crop:[xmin, ymin, width, height]
                            img, read_stats = read_sensor_image(
                                inp,
                                params,
                                filename,
                                index,
                                self.height,
                                self.width,
                                self.img_dir,
                                params["image_dir_prefix"],
                                self.transform,
                                crop=crop,
                                cache_folder=os.path.join(self.cache_dir, sub_dir_name),
                            )

                            if params["use_agent_semantic_masks"]:
                                # TODO(nicholas.guyett.ctr) Can we de-duplicate this between scene and agent images?
                                mask_crop = new_bboxes[agent_key][i, :]

                                if params["agent_semantic_mask_padding_ratio"]:
                                    padding_ratio = params["agent_semantic_mask_padding_ratio"]
                                    mask_crop = add_padding_to_bbox(mask_crop, padding_ratio)

                                semantic_mask, _ = read_sensor_image(
                                    inp,
                                    params,
                                    filename,
                                    index,
                                    self.height,
                                    self.width,
                                    [
                                        img_dir.replace(params["image_dir_prefix"], params["mask_dir_prefix"])
                                        for img_dir in self.img_dir
                                    ],
                                    params["mask_dir_prefix"],
                                    self.transform,
                                    crop=mask_crop,
                                    cache_folder=os.path.join(self.cache_dir, f"{sub_dir_name}-mask"),
                                    alternate_extension="png",
                                )
                                img = np.concatenate((img, semantic_mask), axis=0)

                            if params["use_agent_pose_estimates"]:
                                agent_pose, _ = read_sensor_image(
                                    inp,
                                    params,
                                    filename,
                                    index,
                                    self.height,
                                    self.width,
                                    [
                                        img_dir.replace(params["image_dir_prefix"], params["pose_dir_prefix"])
                                        for img_dir in self.img_dir
                                    ],
                                    params["pose_dir_prefix"],
                                    self.transform,
                                    crop=crop,
                                    cache_folder=os.path.join(self.cache_dir, f"{sub_dir_name}-pose"),
                                    alternate_extension="png",
                                )
                                img = np.concatenate((img, agent_pose), axis=0)

                            for k in "height", "width":
                                if k in read_stats:
                                    stats[k].append(read_stats[k])

                            if self.image_processor is None:
                                images[cur_img_idx] = img
                            else:
                                images[cur_img_idx] = self.image_processor.process(img)
                            images_mapping[cur_img_idx, 0] = agent_idx
                            images_mapping[cur_img_idx, 1] = cnt
                            cur_img_idx += 1
                            cnt += 1

        result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES] = images
        result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_MAPPING] = images_mapping
        result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_STATS] = json.dumps(stats)

        return result_dict


class VelocityHandler(InputsHandler):
    """
    Adds velocities (x,y, valid) to the dataset, valid=1 iff velocity present in protobuf
    Note: Does not use trajectory interpolation
    """

    def __init__(self, params):
        super().__init__(params)

    def get_hash_param_keys(self) -> List[str]:
        return [
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_MAX_AGENTS,
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE,
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params: dict,
        filename: str,
        index: int,
    ):

        velocity_list = []
        for trajectory in result_dict[ProtobufPredictionDataset.DATASET_KEY_TRAJECTORIES]:
            velocity_k = []
            past_timesteps = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS]
            future_timesteps = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS]
            total_timesteps = past_timesteps + future_timesteps

            for cnt in range(min(total_timesteps, len(trajectory))):
                is_velocity_valid = (
                    len(trajectory[cnt]["velocity"]) >= 2 and np.sum(np.isnan(trajectory[cnt]["velocity"])) == 0
                )
                if is_velocity_valid:
                    velocity_k.append([trajectory[cnt]["velocity"][0], trajectory[cnt]["velocity"][1], 1.0])
                else:
                    velocity_k.append([0.0, 0.0, 0.0])  # interesting to consider vehicles without velocity

            if cnt < total_timesteps:
                for cnt2 in range(total_timesteps - len(velocity_k)):
                    # Pad with invalid positions.
                    velocity_k.append([0.0, 0.0, 0.0])

            velocity_list.append(velocity_k)
        try:
            velocities = np.array(velocity_list).transpose(1, 0, 2)
        except ValueError as e:
            print("Got: at " + self.filename + ", index " + str(index) + ": \n" + str(e))
            raise (e)

        agent_idx_full = result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IDX]
        agent_idx_valid = agent_idx_full[agent_idx_full >= 0]
        velocities_ = velocities[:, agent_idx_valid, :]
        velocities = np.zeros(
            [total_timesteps, params[protobuf_training_parameter_names.PARAM_MAX_AGENTS], velocities.shape[2]]
        )
        velocities[: velocities_.shape[0], : velocities_.shape[1], :] = velocities_

        # if self.transpose_agent_times: # Is this really any options?
        velocities = np.transpose(velocities, [1, 0, 2])

        ret = {}
        ret[ProtobufPredictionDataset.DATASET_KEY_VELOCITIES] = velocities

        return ret


class BoundingBoxHandler(InputsHandler):
    """
    Adds bounding box (width, length, valid) to the dataset, valid=1 iff heading present in protobuf
    Note: Does not use trajectory interpolation
    """

    def __init__(self, params):
        super().__init__(params)

    def get_hash_param_keys(self) -> List[str]:
        return [
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_MAX_AGENTS,
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE,
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):

        past_timesteps = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS]
        future_timesteps = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS]
        total_timesteps = past_timesteps + future_timesteps
        wlh_list = []
        for trajectory in result_dict[ProtobufPredictionDataset.DATASET_KEY_TRAJECTORIES]:
            wlh_k = []
            for cnt in range(min(total_timesteps, len(trajectory))):
                is_bbox_valid = trajectory[cnt]["width"] > 0 and trajectory[cnt]["length"] > 0
                if is_bbox_valid:
                    wlh_k.append([trajectory[cnt]["width"], trajectory[cnt]["length"], is_bbox_valid])
                else:
                    wlh_k.append([0.0, 0.0, 0.0])
            if cnt < total_timesteps:
                for cnt2 in range(total_timesteps - len(wlh_k)):
                    # Pad with invalid positions.
                    wlh_k.append([0.0, 0.0, 0.0])
            wlh_list.append(wlh_k)

        wlhs = np.array(wlh_list).transpose(1, 0, 2)
        agent_idx_full = result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IDX]
        agent_idx_valid = agent_idx_full[agent_idx_full >= 0]
        wlhs_ = wlhs[:, agent_idx_valid, :]
        wlhs = np.zeros([total_timesteps, params[protobuf_training_parameter_names.PARAM_MAX_AGENTS], wlhs.shape[2]])
        wlhs[: wlhs_.shape[0], : wlhs_.shape[1], :] = wlhs_
        wlhs = np.transpose(wlhs, [1, 0, 2])

        ret = {}
        ret[ProtobufPredictionDataset.DATASET_KEY_WLHS] = wlhs

        return ret


class HeadingHandler(InputsHandler):
    """
    Adds Quarternion heading  (a,b,c,d,valid) where q=( a + bi + cj + dk), with a>0 (similar to pygeometry)
    to the dataset, valid=1 iff heading present in protobuf
    Note: Does not use trajectory interpolation
    """

    def __init__(self, params):
        super().__init__(params)

    def get_hash_param_keys(self) -> List[str]:
        return [
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_MAX_AGENTS,
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE,
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):

        past_timesteps = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS]
        future_timesteps = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS]
        total_timesteps = past_timesteps + future_timesteps
        heading_list = []
        for trajectory in result_dict[ProtobufPredictionDataset.DATASET_KEY_TRAJECTORIES]:
            heading_k = []
            for cnt in range(min(total_timesteps, len(trajectory))):

                is_heading_valid = len(trajectory[cnt]["heading"]) == 4
                if is_heading_valid:
                    heading_k.append(
                        [
                            trajectory[cnt]["heading"][0],
                            trajectory[cnt]["heading"][1],
                            trajectory[cnt]["heading"][2],
                            trajectory[cnt]["heading"][3],
                            np.float32(is_heading_valid),
                        ]
                    )
                else:
                    heading_k.append([0.0, 0.0, 0.0, 0.0, 0.0])

            if cnt < total_timesteps:
                for cnt2 in range(total_timesteps - len(heading_k)):
                    # Pad with invalid positions.
                    heading_k.append([0.0, 0.0, 0.0, 0.0, 0.0])
            heading_list.append(heading_k)

        # TODO(nbuckman): allow for self.transpose_agent_time=False
        headings = np.array(heading_list).transpose(1, 0, 2)

        agent_idx_full = result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IDX]
        agent_idx_valid = agent_idx_full[agent_idx_full >= 0]
        headings_ = headings[:, agent_idx_valid, :]
        headings = np.zeros(
            [total_timesteps, params[protobuf_training_parameter_names.PARAM_MAX_AGENTS], headings.shape[2]]
        )
        headings[: headings_.shape[0], : headings_.shape[1], :] = headings_
        headings = np.transpose(headings, [1, 0, 2])

        ret = {}
        ret[ProtobufPredictionDataset.DATASET_KEY_HEADINGS] = headings
        return ret


class CollisionHandler(InputsHandler):
    """
    Adds collision label to the dataset taken from prediction_instance_info, valid=1 iff heading present in protobuf
    Note: Does not use trajectory interpolation
    """

    def __init__(self, params):
        super().__init__(params)

    def get_hash_param_keys(self) -> List[str]:
        return []

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):
        instance_info_dict = json.loads(result_dict[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO])
        collision = np.array(instance_info_dict["collision"], dtype=float)

        ret = {}
        ret[ProtobufPredictionDataset.DATASET_KEY_COLLISION] = collision
        return ret


class WaymoStateHandler(InputsHandler):
    """
    Adds additional Waymo state (length, width, heading, velocity_x, velocity_y) to the dataset.
    """

    def __init__(self, params):
        super().__init__(params)

    def get_hash_param_keys(self) -> List[str]:
        return [
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS,
            protobuf_training_parameter_names.PARAM_MAX_AGENTS,
            protobuf_training_parameter_names.PARAM_PAST_TIMESTEP_SIZE,
            protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE,
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):
        # This assumes no subsampling.
        past_timesteps = params[protobuf_training_parameter_names.PARAM_PAST_TIMESTEPS]
        future_timesteps = params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS]
        total_timesteps = past_timesteps + future_timesteps
        state_list = []
        for trajectory in result_dict[ProtobufPredictionDataset.DATASET_KEY_TRAJECTORIES]:
            state_k = []
            for cnt in range(min(total_timesteps, len(trajectory))):
                is_state_valid = (
                    "x" in trajectory[cnt]["position"]
                    and "y" in trajectory[cnt]["position"]
                    and trajectory[cnt]["position"]["x"] != 0
                    and trajectory[cnt]["position"]["y"] != 0
                )
                if is_state_valid:
                    state_k.append(
                        [
                            trajectory[cnt]["length"],
                            trajectory[cnt]["width"],
                            trajectory[cnt]["heading"][0],
                            trajectory[cnt]["velocity"][0],
                            trajectory[cnt]["velocity"][1],
                        ]
                    )
                else:
                    state_k.append([0.0] * 5)
            if cnt < total_timesteps:
                for cnt2 in range(total_timesteps - len(state_k)):
                    # Pad with invalid positions.
                    state_k.append([0.0] * 5)
            state_list.append(state_k)
        states = np.array(state_list).transpose(1, 0, 2)
        agent_idx_full = result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IDX]
        agent_idx_valid = [x for x in agent_idx_full if x >= 0 and x < params["max_agents"]]
        states_ = states[:, agent_idx_valid, :]
        states = np.zeros(
            [total_timesteps, params[protobuf_training_parameter_names.PARAM_MAX_AGENTS], states.shape[2]]
        )
        states[: states_.shape[0], : states_.shape[1], :] = states_
        states = np.transpose(states, [1, 0, 2])

        ret = {}
        ret[ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE] = states

        return ret
