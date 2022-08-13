import argparse
import json
import multiprocessing
import os
import platform
import resource
import sys
import tempfile
from collections import namedtuple
from functools import partial
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from typeguard import typechecked

from triceps.protobuf.prediction_dataset_synthetic import generate_synthetic_data

# Temporary fix for issue https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/5
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


DEFAULT_NUM_WORKERS = min(64, multiprocessing.cpu_count())
DEFAULT_ARTIFACT_DIR = "intent/artifacts"
DEFAULT_MODEL_DIR = "intent/models"


# This dataset names list to be consistent with intnet/multiagents/cache_util.py    INPUT_DIRS.keys()
AVAILABLE_DATASET_NAMES = [
    "original",
    "new",
    "test_vehicle",
    "test_vehicle_aug_train",
    "test_vehicle",
    "test_vehicle_aug_train",
    "train_old-test_sep",
    "train_old_august-test_sep",
    "train_old_new_august-test_sep",
    "train_old_new-test_sep",
    "train_old_new-test_old",
    "train_half_old-test_old",
    "train_old-test_old__0_5_train_ratio",
]

# Parameters whose defaults can be overriden by the environment
RAD_ENVIRONMENT_OVERRIDES = [
    "global_dir_prefix",  # RAD_GLOBAL_DIR_PREFIX
    "input_dir_prefix",  # RAD_INPUT_DIR_PREFIX
    "image_dir_prefix",  # RAD_IMAGE_DIR_PREFIX
    "mask_dir_prefix",  # RAD_MASK_DIR_PREFIX
    "pose_dir_prefix",  # RAD_POSE_DIR_PREFIX
    "cache_dir",  # RAD_CACHE_DIR
    "cache_read_only",  # RAD_CACHE_READ_ONLY
    "image_encoder_nan_retries",  # RAD_IMAGE_ENCODER_NAN_RETRIES
]


def add_environment_fallback_alerts(param_names: List[str], parser: argparse.ArgumentParser):
    """Monkey patches provided parser to generate a print message when default value is sourced from environment"""
    # TODO(nicholas.guyett.ctr) replace this with a less hacky solution
    fallback_sentinel = "!!!fallback_sentinel!!!"  # used as temporary default for detection purposes

    def create_fallback(param: str, original_type: Callable, original_default: Any):
        def do_fallback(value):
            env_name = f"RAD_{param.upper()}"
            if value == fallback_sentinel:
                if env_name in os.environ:
                    value = os.environ[env_name]
                    print(f"Parameter --{param.replace('_', '-')} using ENV value: {value}")
                else:
                    value = original_default

            if original_type:
                return original_type(value)
            else:
                return value

        return do_fallback

    parser_actions = [action for action in parser._actions if action.dest in param_names]
    for action in parser_actions:
        original_type = action.type
        orginal_default = action.default

        action.type = create_fallback(action.dest, original_type, orginal_default)
        action.default = fallback_sentinel

    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ArgKeyValuePair:
    def __init__(
        self,
        string: str,
        separator: Any = "=",
        key_type: Callable[[str], Any] = str,
        value_type: Callable[[str], Any] = str,
    ):
        # TODO(nicholas.guyett.ctr) preserve separator for advanced usages with regexes
        key, value = string.split(separator, maxsplit=1)
        self.key = key_type(key)
        self.value = value_type(value)


def nullable_str(string):
    return None if string == "none" else string


def image_path_key_list():
    return [
        "image_dir",
    ]


def input_path_key_list():
    return [
        "input_dir",
        "datasets_name_lists",
        "merge_cached_splits",
        "input_training_dir",
        "input_validation_dir",
    ]


def outputs_path_key_list():
    return [
        "cache_dir",
        "logs_dir",
        "interp_dir",
        "model_save_folder",
        "model_load_folder",
        "artifacts_folder",
    ]


def path_prefix_keys_list():
    return [
        # "global_dir_prefix" is handled explicitly in `expand_user_path_params`
        "input_dir_prefix",
        "image_dir_prefix",
        "output_dir_prefix",
        "mask_dir_prefix",
        "pose_dir_prefix",
    ]


def path_keys_list():
    return path_prefix_keys_list() + input_path_key_list() + outputs_path_key_list()


@typechecked
def normalize_path(path: Optional[str], default_prefix: str):
    if path is None or isinstance(path, Path):
        return path

    if path.startswith("."):
        path = os.path.join(os.curdir, path)

    expanded = os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
    if os.path.isabs(expanded):
        return expanded
    else:
        return os.path.join(default_prefix, expanded)


class ArgPath(type(Path())):
    def __init__(self, *string_paths: Union[str, Path], prefix_name="global_dir_prefix"):
        # Paths don't have an init and are instead constructed within __new__, so super().__init__ is object.__init__
        # and we don't pass string_paths in init
        # pylint: disable=unused-argument
        self.prefix_name = prefix_name
        self._parts = list(normalize_path(str(part), default_prefix="") for part in self._parts)
        if str(string_paths[0]).startswith("./"):
            self._parts = [os.getcwd()] + self._parts
        super().__init__()

    def with_prefix(self, params):
        if self.is_absolute():
            return self.resolve()
        else:
            return ArgPath(params[self.prefix_name], self, prefix_name=self.prefix_name).resolve()

    def without_prefix(self, params):
        return ArgPath(self.relative_to(self.prefix_name), prefix_name=self.prefix_name)


@typechecked
def expand_user_path_params(params: Dict, additional_keys: List[str] = None) -> Dict:
    """Expands user relative paths in parameter dictionaries"""
    # Additional checks
    if not params["data_debug_mode"]:

        input_dir = params.get("input_dir")
        if input_dir is None:
            input_dir = params.get("input_training_dir")
        if params["use_waymo_dataset"] is None and any("waymo" in f for f in input_dir):
            # If "waymo" is present in the path, ask the user explicitly specify "--use-waymo-dataset"
            raise RuntimeError("Is this using Waymo dataset? Please specify --use-waymo-dataset True/False.")
    # Expand variables
    global_dir_prefix = normalize_path(params["global_dir_prefix"], "")
    if not os.path.isabs(global_dir_prefix):
        raise Exception("Global directory prefix must expand to an absolute path.")

    # Expand prefix parameters first
    for key in path_prefix_keys_list():
        params[key] = normalize_path(params[key], default_prefix=global_dir_prefix)

    # Expand any ArgPaths in the parameters
    for key, value in params.items():
        # TODO clean up collection handling
        if isinstance(value, ArgPath):
            params[key] = value.with_prefix(params)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, ArgPath):
                    value[i] = item.with_prefix(params)
                elif isinstance(item, ArgKeyValuePair):
                    arg_key, arg_val = item.key, item.value
                    if isinstance(arg_key, ArgPath):
                        arg_key = arg_key.with_prefix(params)
                    if isinstance(arg_val, ArgPath):
                        arg_val = arg_val.with_prefix(params)

                    value[i] = arg_key, arg_val

    # Expand any remaining strings found in
    for default_prefix, keys_list in (
        (params["input_dir_prefix"], input_path_key_list()),
        (params["image_dir_prefix"], image_path_key_list()),
        (params["output_dir_prefix"], outputs_path_key_list()),
        (global_dir_prefix, additional_keys),
    ):
        for key in keys_list:
            if params[key] is not None:
                if type(params[key]) == str:
                    params[key] = normalize_path(params[key], default_prefix)
                elif type(params[key]) == list:
                    params[key] = [normalize_path(path, default_prefix) for path in params[key]]

    return params


def args_to_dict(args, additional_file_path_params=None):
    params = expand_user_path_params(vars(args), additional_keys=additional_file_path_params)

    if params["data_debug_mode"]:
        params["data_debug_mode"] = generate_synthetic_data

    return params


@typechecked
def ensure_command_param(
    arg_list: List[str],
    param: str,
    value: str = None,
    param_alias: str = None,
) -> None:
    """
    :param arg_list: Parsed command-line string
    :param param: Name of the parameter to modify or append
    :param value: Value to which to set the parameter. Leave None for flags.
    :param param_alias: A different version of the param, for example short '-' version vs. long '--' version.
        Will be replaced by the param.
    """
    param_alias = param_alias or param
    found_it = False
    for i, arg in enumerate(arg_list):
        if arg.startswith(param_alias) or arg.startswith(param):
            if "=" in arg:
                if value:
                    arg_list[i] = f"{param}={value}"
                else:
                    raise ValueError(f"Param {param} has '=' form but no value provided")
            else:
                arg_list[i] = param
                if value:
                    if i < len(arg_list) - 1 and not arg_list[i + 1].startswith("-"):
                        arg_list[i + 1] = value
                    else:
                        if i < len(arg_list) - 1:
                            raise ValueError(f"Param {param} followed by param {arg_list[i + 1]} instead of value")
                        else:
                            raise ValueError(f"Param {param} provided a value, but no value to replace")
            found_it = True
    if not found_it:
        arg_list.extend([param, value])


def semantic_mask_argument_setter(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("semantic mask")

    group.add_argument(
        "--mask-dir-prefix",
        default="semantic_masks",
        help="Path to directory to read semantic mask images from.  Relative paths prefixed with `--global-dir-prefix`",
    )
    group.add_argument(
        "--pose-dir-prefix",
        default="pose_estimates",
        help="Path to directory to read pose estimate data from.  Relative paths prefixed with `--global-dir-prefix`",
    )

    group.add_argument(
        "--use-agent-semantic-masks",
        type=str2bool,
        help="Whether to use semantic masks during agent image encoding",
    )
    group.add_argument(
        "--agent-semantic-mask-padding-ratio",
        type=float,
        help="Additional padding to apply to bounding box of semantic masks, relative to agent image bounding box",
    )

    group.add_argument(
        "--use-scene-semantic-masks",
        type=str2bool,
        help="Whether to use semantic masks during scene image encoding",
    )
    group.add_argument(
        "--use-agent-pose-estimates",
        type=str2bool,
        help="Whether to use agent pose estimates during agent image encoding",
    )

    return parser


def parse_arguments(
    args: Optional[List[str]] = None,
    additional_arguments_setter: Optional[List[Callable[[argparse.ArgumentParser], argparse.ArgumentParser]]] = None,
    default_override: Optional[dict] = None,
) -> argparse.Namespace:
    """
    Parse arguments to run training.
    :param args: List of arguments to parse. Command line arguments are used if this is set to none.
    :param additional_arguments_setter: a function that takes an argparse.ArgumentParser object and adds arguments
    to it for a specific usecase. Can be a list of those.
    :param default_override: A dictionary to override the default values
    TODO: move any GNN-prediction-training specific flags to the argument setter.
    :return:
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )
    param_sets = {}
    parser.param_sets = param_sets  # May be overridden by argument setters

    parser.add_argument(
        "--input-dir",
        nargs="+",
        help="Input directories with protobuf files to read.  Relative paths prefixed with `--input-dir-prefix`",
    )
    parser.add_argument(
        "--input-dir-names", nargs="+", help="The name for the dataset to use, NOTE this will override the --input-dir."
    )
    # Separate training and validation dir.
    parser.add_argument(
        "--input-training-dir",
        nargs="+",
        help="Input directories with training protobuf files to read.  Relative paths prefixed with `--input-dir-prefix`",
    )
    parser.add_argument(
        "--input-validation-dir",
        nargs="+",
        help="Input directories with validation protobuf files to read.  Relative paths prefixed with `--input-dir-prefix`",
    )
    parser.add_argument(
        "--global-dir-prefix",
        default="~/",
        help="Directory base path.  Sets the default parent directory for all relative paths.  Defaults to '~/'",
    )
    parser.add_argument(
        "--input-dir-prefix",
        default=None,
        help="Input directories base, the relative data dir should be invariant across machines. Default to global dir prefix.  Relative paths prefixed with `--global-dir-prefix`",
    )
    parser.add_argument(
        "--image-dir-prefix",
        default=None,
        help="Image directories base, the relative data dir should be invariant across machines. Default to global dir prefix.  Relative paths prefixed with `--global-dir-prefix`",
    )
    parser.add_argument(
        "--output-dir-prefix",
        default=None,
        help="Output directory base path, output and model dirs will be relative this prefix. Default to global dir prefix.  Relative paths prefixed with `--global-dir-prefix`",
    )

    parser.add_argument(
        "--image-dir",
        nargs="+",
        help="Image input directories with raw images to read.  Relative paths prefixed with `--image-dir-prefix`",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="intent/cache",
        help="Cache directory.  Relative paths are prefixed with `--output-dir-prefix`",
    )
    parser.add_argument(
        "--cache-read-only",
        type=str2bool,
        default=False,
        help="Use the cache in read only mode.",
    )
    parser.add_argument(
        "--use-cache-lock",
        type=str2bool,
        default="false",
        help="Lock the cache files for images. Set to false if you have a pre-created image cache dir.",
    )
    parser.add_argument(
        "--dataset-names",
        # choices=AVAILABLE_DATASET_NAMES,
        nargs="+",
        default=None,
        help="The cache data set to download",
    )
    parser.add_argument(
        "--image-list-key-num",
        type=int,
        default=1,
        help="Lock the cache files for images. Set to false if you have a pre-created image cache dir.",
    )

    parser.add_argument(
        "--fixed-ego-orientation",
        type=str2bool,
        default="false",
        help="Fix ego orientation along x axis for crossing calculation, used for test vehicle data.",
    )

    parser.add_argument("--cache-dataset-item", type=str2bool, default="false", help="Cache dataset items.")
    parser.add_argument("--cache-map-handler", type=str2bool, default="true", help="Cache map handler.")
    parser.add_argument("--cache-map-encoder", type=str2bool, default="false", help="Cache map encoder.")
    parser.add_argument("--cache-latent-factors", type=str2bool, default="false", help="Cache latent factors.")
    parser.add_argument("--cache-splits", type=str2bool, default="true", help="Cache dataset splits.")
    parser.add_argument("--merge-cached-splits", nargs="+", help="Cache dataset splits.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable usage of cache globally.")

    parser.add_argument(
        "--interp-dir",
        type=str,
        default=os.path.expanduser("intent/interp_vis"),
        help="directory for saving interpolate plots.  Relative paths prefixed with `--output-dir-prefix`",
    )
    parser.add_argument("--async-save", type=str2bool, default="false", help="Save the model in another thread")
    parser.add_argument("--save-to-s3", type=str2bool, default="false", help="Save the model to S3")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="intent/logs",
        help="Logs directory.  Relative paths are prefixed with `--output-dir-prefix`",
    )
    parser.add_argument(
        "--data-debug-mode", type=str2bool, default="false", help="Replace data with linear random trajectories."
    )
    parser.add_argument(
        "--artifacts-folder",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help="Saved train artifacts directory. Relative paths prefixed with `--output-dir-prefix`",
    )
    parser.add_argument(
        "--model-save-folder",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Saved models directory. Relative paths prefixed with `--output-dir-prefix`",
    )
    parser.add_argument(
        "--model-load-folder",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Loaded models directory. Relative paths prefixed with `--output-dir-prefix`",
    )
    parser.add_argument(
        "--resume-session-name", type=str, default=None, help="Session name of saved model for resuming training"
    )
    parser.add_argument(
        "--current-session-name",
        type=str,
        default=os.environ.get("RAD_SESSION_ID", None),
        help="The name for this session. Used for automation.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set specific random seed")
    parser.add_argument(
        "--max-files-count",
        type=int,
        default=np.inf,
        help="How many pb files to ingest, limit this for testing quickly.",
    )
    parser.add_argument("--add-input-stub", action="store_true", help="Add a stub input, model for testing.")
    parser.add_argument(
        "--add-input-map-stub", action="store_true", help="Add a global stub map input, model for testing."
    )
    parser.add_argument(
        "--add-agent-map-input-stub", action="store_true", help="Add a stub map input for agents, model for testing."
    )
    parser.add_argument(
        "--add-agent-input-stub", action="store_true", help="Add a stub agent input, model for testing."
    )
    parser.add_argument(
        "--stub-requires-grads", type=str2bool, default="false", help="Sets the requires_grad of input stubs"
    )
    parser.add_argument("--visualize-items", action="store_true", help="Vizualize the data items in training.")
    parser.add_argument(
        "--visualization-image-format",
        default="jpeg",
        choices=[
            "jpeg",
            "png",
        ],
        help="Sets the image format that visualization images will be saved as",
    )
    parser.add_argument(
        "--visualize-interp", action="store_true", help="Visualize resampling and interpolation with plots"
    )
    parser.add_argument(
        "--log-interp-excursions",
        action="store_true",
        help="Write errors in interpolation excursions or other interpolation-time rejections",
    )
    parser.add_argument(
        "--visualize-interp-bad-only",
        action="store_true",
        help="Visualize resampling and interpolation with plots, but only the inaccurate ones",
    )
    parser.add_argument("--disable-tqdm", type=str2bool, default="true", help="Disable the tqdm iteration display.")

    parser.add_argument(
        "--print-model-params", action="store_true", help="Print the number of params in neural networks"
    )
    parser.add_argument(
        "--agent-input-stub-embed-size", type=int, default=8, help="The embedding dimension for agent input stub."
    )
    parser.add_argument("--input-stub-embed-size", type=int, default=8, help="The embedding dimension for input stub.")
    parser.add_argument("--input-stub-dim", type=int, default=64, help="Stub input size.")
    parser.add_argument(
        "--input-map-stub-embed-size", type=int, default=8, help="The embedding dimension for map input stub."
    )
    parser.add_argument("--input-map-stub-dim", type=int, default=64, help="Stub map input size.")
    parser.add_argument("--agent-input-stub-dim", type=int, default=64, help="Stub agent input size.")
    parser.add_argument("--agent-map-input-stub-dim", type=int, default=64, help="Stub agent map input size.")
    parser.add_argument("--max-agents", type=int, default=4, help="Maximum number of agents to read.")
    parser.add_argument("--past-timesteps", type=int, default=15, help="Number of past timesteps.")
    parser.add_argument("--future-timesteps", type=int, default=15, help="Number of future timesteps.")
    parser.add_argument(
        "--past-timestep-size",
        type=float,
        default=0.2,
        help="Timestep size (i.e., sampling period) for the past. Need to update past-timesteps if changing this timestep size.",
    )
    parser.add_argument(
        "--future-timestep-size",
        type=float,
        default=0.2,
        help="Timestep size (i.e., sampling period) for the future. Need to update future-timesteps if changing this timestep size.",
    )
    parser.add_argument("--max-semantic-targets", type=int, default=30, help="Maximum number of semantic targets.")
    parser.add_argument("--max-language-tokens", type=int, default=50, help="Maximum number of language tokens.")
    parser.add_argument("--save-iteration-interval", type=int, default=5, help="Interval for saving the model.")
    parser.add_argument(
        "--save-iteration-interval-offset", type=int, default=1, help="Initial offset for the model saving interval."
    )
    parser.add_argument(
        "--ignore-ego", action="store_true", help="Ignores ego vehicle in data loading and cost computation."
    )
    parser.add_argument("--val-epoch-size", type=int, default=int(256), help="Size of validation epochs.")
    parser.add_argument("--val-batch-size", type=int, default=24, help="Size of validation batch.")
    parser.add_argument(
        "--val-num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of validation dataloader workers."
    )
    parser.add_argument("--vis-epoch-size", type=int, default=int(24), help="Size of visualization batch.")
    parser.add_argument("--vis-batch-size", type=int, default=24, help="Size of visualization batch.")
    parser.add_argument(
        "--vis-num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of visualization dataloader workers."
    )
    parser.add_argument("--epoch-size", type=int, default=int(3600), help="Size training epochs.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=16, help="Size of training batch.")
    parser.add_argument(
        "--dataloader-pin-memory", type=str2bool, default="false", help="Pin memory in training dataloaders."
    )
    parser.add_argument(
        "--dataloader-pre-fetch-factor", type=int, default=3, help="Pre-fetch factor in training dataloaders."
    )
    parser.add_argument(
        "--dataloader-shuffle",
        type=str2bool,
        default=True,
        help="Whether the dataloader should shuffle results.  Defaults to true",
    )

    parser.add_argument(
        "--num-rebalance-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader workers for rebalancing if needed.",
    )
    parser.add_argument("--num-epochs", type=int, default=int(1e5), help="Number of training epochs.")
    parser.add_argument("--img-height", type=int, default=256, help="Height of input global image after resize.")
    parser.add_argument("--img-width", type=int, default=384, help="Width of input global image after resize.")
    parser.add_argument("--image-embed-size", type=int, default=8, help="The embedding dimension for images.")
    parser.add_argument(
        "--scene-image-mode",
        choices=["all", "custom", "none"],
        default="all",
        help="""
            Selection mode of scene images. If this is set to 'custom',
            --scene-image-timepoints can be used for fine-grained control of
            which images to choose. If no timepoints are given, the training
            injects images at all timepoints.
            """,
    )
    parser.add_argument(
        "--scene-image-model",
        type=str,
        choices=["vgg11", "mobilenetv2", "custom_network"],
        default="custom_network",
        help="CNN Architecture used for the scene image backbone.",
    )
    parser.add_argument(
        "--scene-image-timepoints",
        nargs="+",
        type=int,
        default=None,
        help="""
            List of timepoints for which scene images are used. The values have
            to be non-negative and smaller than the total number of timesteps
            (which is the sum of --past-timesteps and --future-timesteps). This
            argument is only relevant if --scene-image-mode is 'custom'.
            """,
    )
    parser.add_argument(
        "--scene-image-fc-widths",
        nargs="+",
        type=int,
        default=None,
        help="""
            List of network widths for the fully connect part of the scene image encoders.
            """,
    )
    parser.add_argument(
        "--bbox-dilate-scale", type=float, default=3.0, help="The detection bbox is dilated by a scale for cropping."
    )
    parser.add_argument(
        "--agent-img-height", type=int, default=128, help="Height of cropped image of the relevant agent after resize."
    )
    parser.add_argument(
        "--agent-img-width", type=int, default=64, help="Width of cropped image of the relevant agent after resize."
    )

    parser.add_argument(
        "--agent-image-processor",
        choices=["pa", "none"],
        default="none",
        help="""
            Processing mode of agent images. If this is set to 'pa',
            the handler should run PA on the images rather than feed images to white-box CNN encoder.
            """,
    )
    parser.add_argument(
        "--agent-image-mode",
        choices=["all", "custom", "none"],
        default="all",
        help="""
            Selection mode of agent images. If this is set to 'custom',
            --agent-image-agents and --agent-image-timepoints can be used for
            fine-grained control of which images to choose. If either of these
            arguments is not given, the training behaves as if they have been
            set to use all agents or timepoints respectively.
            """,
    )
    parser.add_argument(
        "--agent-image-model",
        type=str,
        choices=["vgg11", "mobilenetv2", "custom_network"],
        default="custom_network",
        help="CNN Architecture used for the agent image backbone.",
    )
    parser.add_argument(
        "--agent-image-agents",
        nargs="+",
        type=int,
        default=None,
        help="""
            List of agent indices to be included in agent images. The values
            have to range between 0 and the value of --max-agents. This
            argument is only relevant if --agent-image-mode is 'custom'.
            """,
    )
    parser.add_argument(
        "--agent-image-fc-widths",
        nargs="+",
        type=int,
        default=None,
        help="""
            List of network widths for the fully connect part of the agent image encoders.
            """,
    )
    parser.add_argument(
        "--agent-image-timepoints",
        nargs="+",
        type=int,
        default=None,
        help="""
            List of timepoints for which agent images are used. The values have
            to be non-negative and smaller than the total number of timesteps
            (which is the sum of --past-timesteps and --future-timesteps). This
            argument is only relevant if --agent-image-mode is 'custom'.
            """,
    )
    parser.add_argument(
        "--image-encoder-nan-retries",
        type=int,
        default=0,
        help="Number of times the image encoder should try to regenerate an image encoding if a nan output is detected",
    )
    parser.add_argument("--min-valid-points", type=float, default=4, help="The min valid point of a trajectory")
    parser.add_argument("--map-halfheight", type=float, default=20, help="Height of the map, m.")
    parser.add_argument("--map-halfwidth", type=float, default=20, help="Width of the map, m.")
    parser.add_argument("--map-scale", type=float, default=0.4, help="Resolution of the map, m.")
    parser.add_argument(
        "--map-attention-type", default="point", choices=["none", "element", "point"], help="Map attention type."
    )

    parser.add_argument("--training-set-ratio", type=float, default=0.7, help="Ratio of training set.")
    parser.add_argument("--val-interval", type=int, default=int(3), help="How many iterations between validations.")
    parser.add_argument(
        "--val-interval-offset", type=int, default=0, help="Initial offset for validation iteration interval."
    )
    parser.add_argument("--disable-validation", action="store_true", help="Disables automatic model validation.")
    parser.add_argument("--vis-interval", type=int, default=int(20), help="How many iterations between visualizations.")
    parser.add_argument(
        "--vis-interval-offset", type=int, default=0, help="How many iterations between visualizations."
    )
    parser.add_argument("--disable-visualizations", action="store_true", help="Disables automatic visualizations.")
    parser.add_argument(
        "--interp-type",
        nargs="+",
        default=["interpSpline"],
        help="kind of interpolation to use -- e.g. 'interp1d', 'interpSpline', 'interpGP'",
    )
    parser.add_argument("--multigpu", action="store_true", help="Make the training use multiple GPUsd.")
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU usage, useful for debugging.")
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["train", "infer"],
        default="train",
        help="training / inference mode for the script.",
    )
    parser.add_argument("--save-example-outputs", action="store_true", help="Save example output data.")
    parser.add_argument(
        "--save-cases-for-table",
        action="store_true",
        help="Save all inference output data (used for test vehicle table).",
    )
    parser.add_argument(
        "--table-output-folder-name",
        type=str,
        default="",
        help="""
            The one output path for all inference output data (used for test vehicle table).
            """,
    )
    parser.add_argument(
        "--pa-service-exe-path",
        type=str,
        default="none",
        help="""
            The path to the Perceptive Automata SOMAI service.bin file, found within the somai data componenet
            in the `bin` directory.
            """,
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug statements.")
    parser.add_argument(
        "--pa-service-do-detections",
        action="store_true",
        help="""
            If set, the PA SOMAI component will perform its own detections rather than use the provided
            bounding boxes.
            """,
    )
    parser = semantic_mask_argument_setter(parser)

    if additional_arguments_setter is not None:
        for setter in additional_arguments_setter:
            parser = setter(parser)

    if default_override is not None:
        parser.set_defaults(**default_override)
        default_param_set = default_override.get("param_set", None)
    else:
        default_param_set = None

    param_set_arg_flag = "--param-set"
    param_set_arg_definition = dict(
        nargs="+",
        default=default_param_set,
        choices=list(param_sets.keys()),
        help="Name of the param-set you want to use",
    )
    parser.add_argument(param_set_arg_flag, **param_set_arg_definition)

    # Pre-parse arguments to get the param_sets
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument(param_set_arg_flag, **param_set_arg_definition)
    pre_parsed_args, *_ = temp_parser.parse_known_args(args)
    combined_param_set = {}
    if pre_parsed_args.param_set:
        for name in pre_parsed_args.param_set:
            for key, value in param_sets.get(name).items():
                if key not in combined_param_set:
                    combined_param_set[key] = value
                elif isinstance(value, list):
                    # Merge list parameters, e.g. --input_dir
                    combined_param_set[key] += [item for item in value if item not in combined_param_set[key]]
                elif combined_param_set[key] != value:
                    formatted_param = f"--{key.replace('_', '-')}"
                    parser.error(
                        f"Invalid combination of parameter sets. {formatted_param} has multiple conflicting values"
                    )
                # else values already match

        parser.set_defaults(**combined_param_set)

    parser = add_environment_fallback_alerts(RAD_ENVIRONMENT_OVERRIDES, parser)
    parsed_args = parser.parse_args(args)

    if parsed_args.param_set:
        # Validate that no user provided parameters conflict with the request param sets
        invalid_parameters = []
        for key, value in combined_param_set.items():
            provided_value = parsed_args.__dict__[key]
            if isinstance(provided_value, list):
                # Don't mark this invalid if the user provides the same values in a different order
                provided_value = sorted(provided_value)
                value = sorted(value)

            if provided_value != value:
                invalid_parameters.append(key)

        if invalid_parameters:
            parser.error(
                f"Some arguments conflict with the requested param sets: {invalid_parameters}."
                + "  Please omit these flags or ensure they are compatible with param sets requested."
            )

    parsed_args.input_dir_prefix = parsed_args.input_dir_prefix or parsed_args.global_dir_prefix
    parsed_args.image_dir_prefix = parsed_args.image_dir_prefix or parsed_args.global_dir_prefix
    parsed_args.output_dir_prefix = parsed_args.output_dir_prefix or parsed_args.global_dir_prefix

    parsed_args.provided_args = sys.argv[1:].copy() if args is None else args

    # Multithreading breaks on OSX
    # TODO Fix multithreading on OSX
    if platform.system() == "Darwin" and parsed_args.num_workers > 0:
        print("Multithreading is not currently supported on OSX.  All worker parameters have been set to 0")
        parsed_args.num_workers = 0
        parsed_args.vis_num_workers = 0
        parsed_args.val_num_workers = 0
        parsed_args.num_rebalance_workers = 0

    return parsed_args


def verify_and_update_arg(parser, arg_name, **kwargs):
    """Verify the flag exists, then update parameters.

    Parameters
    ----------
    parser : parser
        The argument parser to use.
    arg_name : string
        The name of the flag, e.g. --input-dir.
    kwargs : arguments
        The arguments to use for the flag.
    """
    # TODO(guy.rosman): Find a stable way to get default arguments from the parser
    # TODO  (e.g. help message for an existing flag) rather than having to copy them.
    arg_dict = vars(parser.parse_args([]))
    arg_name_stripped = arg_name[2:].replace("-", "_")
    assert arg_name_stripped in arg_dict
    return parser.add_argument(arg_name, **kwargs)
