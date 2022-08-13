import hashlib
import os
import subprocess
import tarfile
from collections import OrderedDict
from typing import Dict, List

import torch

from radutils.misc import remove_prefix

# This is the version number for the data and handler.
# Any code changes made to the data or the handlers should bump this number.
# And regenerate the cache.
DATA_AND_HANDLER_VERSION = 6

try:
    # Load TRI specific data path.
    from intent.multiagents.cache_data_paths_tri import *
except ModuleNotFoundError:
    # These presets are not available in public repo.
    INPUT_DIRS = {}
    CACHE_S3_PATH = {}
    CACHE_TYPES = {}
    NAMED_PARAM_SETS = {}


def get_input_dirs_for_dataset(dataset_name):
    assert dataset_name in INPUT_DIRS, f"The requested dataset name is not available, {dataset_name}"
    return INPUT_DIRS[dataset_name]


def cache_s3_path(cache_type: str, torch_ver: str, param_hash: str) -> List[str]:
    if cache_type not in CACHE_S3_PATH:
        raise ValueError(f"Cache type '{cache_type}' not found, supported values are {CACHE_TYPES}")
    if DATA_AND_HANDLER_VERSION not in CACHE_S3_PATH[cache_type]:
        raise ValueError(
            "Handler version '{}' not found, supported values are {}".format(
                DATA_AND_HANDLER_VERSION, CACHE_S3_PATH[cache_type].keys()
            )
        )
    if torch_ver not in CACHE_S3_PATH[cache_type][DATA_AND_HANDLER_VERSION]:
        raise ValueError(
            "Torch version '{}' not found, supported values are {}".format(
                torch_ver, CACHE_S3_PATH[cache_type][DATA_AND_HANDLER_VERSION].keys()
            )
        )

    param_map = CACHE_S3_PATH[cache_type][DATA_AND_HANDLER_VERSION][torch_ver]
    if param_hash not in param_map:
        raise ValueError(
            "Param hash '{}' not found for cache type {}, version {}, torch {}, supported values are {}".format(
                param_hash,
                cache_type,
                DATA_AND_HANDLER_VERSION,
                torch_ver,
                param_map.keys(),
            )
        )

    s3_keys = param_map[param_hash]
    if not isinstance(s3_keys, list):
        s3_keys = [s3_keys]
    return s3_keys


# The dir params to determine the hash of the split
SPLIT_HASH_DIRS = ["input_dir", "input_training_dir", "input_validation_dir"]

# This is a list of parameters that are used in the dataloader and data handlers.
# Different hash would be generated at run time when those parameter changes,
# and therefore read different cache.
PARAM_HASH_LIST = (
    "data_debug_mode",
    "cache_dataset_item",
    # interpolate_trajectory
    "augmentation_timestamp_scale",
    "past_timesteps",
    "future_timesteps",
    "past_timestep_size",
    "future_timestep_size",
    "interp_type",
    "past_only_dataloading",
    "trajectory_validity_criterion",
    "trajectory_time_ego_padding",
    "trajectory_time_ado_padding",
    "interpolation_excursion_threshold",
    # agent_selection_handler: select_agent
    # This is in the main cache
    "ego_agent_only",
    "ignore_ego",
    "max_agents",
    "agent_types",
    "min_valid_points",
    # Map 'cache
    # "disable_map_input",
    # "map_input_type",
    ### map 'handler: PointMapHandler
    # "map_points_max",
    # "map_sampling_length",
    # "map_sampling_minimum_length",
    # RasterMapHandler
    # "map_halfwidth",
    # "map_halfheight",
    # "map_scale",
    # add_polynomial_features
    # This is only used with PointMap
    # "map_polyline_feature_degree",
    # global_image_handler
    ### "scene_image_mode",
    # "scene_image_timepoints",
    # "img_height",
    # "img_width",
    # Agen_image_handler',
    ###"agent_image_mode",
    # "agent_image_agents",
    # "agent_image_timepoints",
    # "agent_image_processor",
    # "agent_img_height",
    # "agent_img_height",
    # "require_agent_images",
    # "bbox_dilate_scale",
    # semantic_handler
    # "use_semantics",
    # "latent_factors_file",
    # "max_semantic_targets",
)


def compute_hash(string: str) -> str:
    return hashlib.md5(string.encode()).hexdigest()


def get_dir_names(params_in):
    # Replace absolute directories with base folder for transferability between machines.
    # This assumes the folder name is renamed after a dataset has been modified.
    ret_param = {}
    for dir_list in SPLIT_HASH_DIRS:
        assert dir_list in params_in, "Parameter --{} needs to be provided.".format(dir_list)
        if params_in[dir_list] is not None:
            # normpath strips off any trailing slashes, and basename gives the last part of the path.
            dir_all = [remove_prefix(os.path.normpath(p), params_in["input_dir_prefix"]) for p in params_in[dir_list]]
            # Sort bast directories.
            dir_all.sort()
        else:
            dir_all = params_in[dir_list]
        ret_param[dir_list] = str(dir_all)
    return ret_param


def get_hashing_params(params_in: dict, param_key_list=PARAM_HASH_LIST) -> dict:
    param = {key: params_in[key] for key in param_key_list}

    if "latent_factors_file" in param:
        param["latent_factors_file"] = os.path.basename(param["latent_factors_file"])
    if not isinstance(param["data_debug_mode"], bool):
        # param["data_debug_mode"] is a function, extract its function name
        param["data_debug_mode"] = str(param["data_debug_mode"]).split(" at ")[0]
        params_in["data_debug_mode_str"] = param["data_debug_mode"]

    # Add additional params to the list, in case a non-standard predictor (i.e. hybrid, language) is used.
    # This additional list should be specified by the training script, and not used by default.
    # So that changing the additional params wouldn't affect the param_hash who don't use it.
    if "additional_cache_param_list" in params_in:
        additional_cache_params = params_in["additional_cache_param_list"]
        for p in additional_cache_params:
            assert p in params_in, "Parameter --{} needs to be provided.".format(p)
            param[p] = params_in[p]
    param = dict(sorted(param.items()))
    return param


def compute_param_hash(params_in: dict, param_key_list=PARAM_HASH_LIST) -> str:
    param = get_hashing_params(params_in, param_key_list)
    return compute_hash(f"{param}")


def split_reading_hash(params: dict, postfix: str = "") -> str:
    """Computes the hash for reading data splits

    Parameters
    ----------
    params: dict
        The parameters dictionary for training.
    postfix: str
        The postfix to add to the parameters.

    Returns
    -------
    hash: str
        The hash string.
    """
    params_copy = OrderedDict()
    for key in params:
        params_copy[key] = params[key]
    del_keys = []
    for key, val in params_copy.items():
        if callable(val) or len(str(type(val))) > 35:
            del_keys.append(key)

        if isinstance(val, list) and len(val) > 0 and (callable(val[0]) or str(type(val[0])).startswith("<class")):
            del_keys.append(key)
    for key in del_keys:
        del params_copy[key]
    # remove values that do not affect the dataset or that are not fixed
    for key in [
        "datasets_name_lists",
        "batch_size",
        "val_batch_size",
        "vis_batch_size",
        "num_workers",
        "val_num_workers",
        "vis_num_workers",
        "epoch_size",
        "val_epoch_size",
        "vis_epoch_size",
    ]:
        params_copy.pop(key, None)

    return compute_hash(str(params_copy) + postfix)


def download_and_extract(cache_dir, s3_key):
    s3_prefix = os.path.dirname(s3_key)
    s3_key_filename = os.path.basename(s3_key)
    sync_args = [
        "aws",
        "s3",
        "sync",
        "--no-progress",
        s3_prefix,
        cache_dir,
        "--exclude",
        "*",
        "--include",
        f"{s3_key_filename}",
    ]
    os.makedirs(cache_dir, exist_ok=True)
    print(subprocess.list2cmdline(sync_args))
    subprocess.check_call(sync_args)
    tar_target = os.path.join(cache_dir, s3_key_filename)
    print(f"extracting cache tar: {tar_target} to ")
    with tarfile.open(tar_target) as tar:
        tar.extractall(cache_dir)


torch_ver_to_conda_env = {
    "1.9": "pt190",
}


def torch_version() -> str:
    torch_ver = ".".join(torch.__version__.split(".")[0:2])
    torch_ver = torch_ver_to_conda_env[torch_ver]
    return torch_ver


def download_cache_from_s3(cache_type: str, param_hash: str, cache_dir: str) -> None:
    """Download and extract cache archive file"""
    print(f"downloading cache file {cache_type} from S3 for hash {param_hash}")
    torch_ver = torch_version()
    s3_keys = cache_s3_path(cache_type, torch_ver, param_hash)
    for key in s3_keys:
        download_and_extract(cache_dir, key)


def download_caches_from_s3(param_hash: str, params: Dict) -> None:
    """Download and extract cache archive files"""
    for cache_type in params["dataset_names"]:
        download_cache_from_s3(cache_type, param_hash, params["cache_dir_original"])
