import datetime
import multiprocessing
import os
from pathlib import Path
from typing import Dict

from typeguard import typechecked

from intent.multiagents.cache_utils import NAMED_PARAM_SETS
from intent.trajectory_prediction_misc import str2bool
from loaders.ado_key_names import (
    AGENT_TYPE_BICYCLE,
    AGENT_TYPE_CAR,
    AGENT_TYPE_LARGEVEHICLE,
    AGENT_TYPE_MOTORCYCLE,
    AGENT_TYPE_PEDESTRIAN,
    AGENT_TYPE_TRUCK,
)
from model_zoo.intent.prediction_latent_factors import (
    constant_latent_factors_generator,
    explicit_duration_factors_generator,
)
from model_zoo.intent.prediction_latent_factors_attention import attention_factors_generator

try:
    from model_zoo.intent.prediction_latent_factors_causality import causality_factors_generator
except ModuleNotFoundError:
    causality_factors_generator = None
from triceps.protobuf.prediction_dataset_synthetic import generate_synthetic_data
from triceps.protobuf.proto_arguments import nullable_str, verify_and_update_arg

DEFAULT_NUM_WORKERS = min(64, multiprocessing.cpu_count())

# The agent types considered in the predictor by default.
# Those correspond to numeric values [2, 3, 4, 5, 6, 7]
# One-hot encoded in batch_items
DEFAULT_AGENT_TYPES = [
    AGENT_TYPE_CAR,
    AGENT_TYPE_BICYCLE,
    AGENT_TYPE_MOTORCYCLE,
    AGENT_TYPE_PEDESTRIAN,
    AGENT_TYPE_LARGEVEHICLE,
    AGENT_TYPE_TRUCK,
]

# Those correspond to numeric values [2, 5, 3]
# One-hot encoded in batch_items
WAYMO_AGENT_TYPES = [
    AGENT_TYPE_CAR,
    AGENT_TYPE_PEDESTRIAN,
    AGENT_TYPE_BICYCLE,
]

AGENT_TYPES_MAP = {
    "car": AGENT_TYPE_CAR,
    "bicycle": AGENT_TYPE_BICYCLE,
    "motorcycle": AGENT_TYPE_MOTORCYCLE,
    "pedestrian": AGENT_TYPE_PEDESTRIAN,
    "largevehicle": AGENT_TYPE_LARGEVEHICLE,
    "truck": AGENT_TYPE_TRUCK,
}


def pedestrian_prediction_setter(parser):
    parser.param_sets.update({name: dataset["params"] for name, dataset in NAMED_PARAM_SETS.items()})

    parser.add_argument("--val-batch-size", type=int, default=24, help="Size of validation batch.")
    parser.add_argument(
        "--val-num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of validation dataloader workers."
    )
    parser.add_argument("--vis-epoch-size", type=int, default=64, help="Size of visualization batch.")
    parser.add_argument("--vis-batch-size", type=int, default=24, help="Size of visualization batch.")
    parser.add_argument(
        "--vis-num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of visualization dataloader workers."
    )
    parser.add_argument("--epoch-size", type=int, default=2048, help="Size of training epochs.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=24, help="Size of training batch.")
    parser.add_argument(
        "--regression-test-early-stop",
        action="store_true",
        default=False,
        help="When set, run regression test early stop scheme.",
    )
    parser.add_argument(
        "--regression-test-early-stop-test-error-type",
        type=str,
        default="MoN_fde_error",
        help="Specify the type of validation error to test on",
    )
    parser.add_argument(
        "--regression-test-early-stop-test-error",
        type=float,
        default=0.6,
        help="Specify the validation error, when the error is lower, stop training. Default is 0.6",
    )
    parser.add_argument(
        "--profiler-type",
        choices=["none", "nvidia", "pytorch", "wallclock"],
        default=None,
        help="""Specify which profiler to use. By default, no profiling. Note, pytorch profiler requires at least 2 epochs""",
    )
    parser.add_argument(
        "--profiler-duration",
        choices=[
            "none",
            "all",
            "full_epoch",
            "training",
            "dataloading_training",
            "dataloading",
            "generator",
            "costs",
            "optimization",
            "log_stats",
        ],
        default="none",
        help="""Specify what to profile. By default, no profiling.""",
    )
    verify_and_update_arg(
        parser, arg_name="--past-timestep-size", type=float, default=0.3, help="Timestep, i.e., sampling period"
    )
    verify_and_update_arg(
        parser, arg_name="--future-timestep-size", type=float, default=0.3, help="Timestep, i.e., sampling period"
    )
    verify_and_update_arg(parser, arg_name="--past-timesteps", type=int, default=10, help="Number of past timesteps.")
    verify_and_update_arg(
        parser, arg_name="--future-timesteps", type=int, default=10, help="Number of future timesteps."
    )
    verify_and_update_arg(
        parser,
        "--agent-image-mode",
        choices=["all", "custom", "none"],
        default="custom",
        help="""
            Selection mode of agent images. If this is set to 'custom',
            --agent-image-agents and --agent-image-timepoints can be used for 
            fine-grained control of which images to choose. If either of these 
            arguments is not given, the training behaves as if they have been
            set to use all agents or timepoints respectively.
            """,
    )
    verify_and_update_arg(
        parser,
        "--agent-image-timepoints",
        nargs="+",
        type=int,
        default=[0, 4, 8],
        help="""
            List of timepoints for which agent images are used. The values have 
            to be non-negative and smaller than the total number of timesteps
            (which is the sum of --past-timesteps and --future-timesteps). This
            argument is only relevant if --agent-image-mode is 'custom'.
            """,
    )
    verify_and_update_arg(
        parser,
        "--scene-image-mode",
        choices=["all", "custom", "none"],
        default="custom",
        help="""
            Selection mode of scene images. If this is set to 'custom',
            --scene-image-timepoints can be used for fine-grained control of 
            which images to choose. If no timepoints are given, the training 
            injects images at all timepoints.
            """,
    )
    verify_and_update_arg(
        parser,
        "--scene-image-timepoints",
        nargs="+",
        type=int,
        default=[0, 4, 8],
        help="""
            List of timepoints for which scene images are used. The values have 
            to be non-negative and smaller than the total number of timesteps
            (which is the sum of --past-timesteps and --future-timesteps). This
            argument is only relevant if --scene-image-mode is 'custom'.
            """,
    )
    verify_and_update_arg(
        parser,
        "--agent-image-agents",
        nargs="+",
        type=int,
        default=[1],
        help="""
            List of agent indices to be included in agent images. The values 
            have to range between 0 and the value of --max-agents. This
            argument is only relevant if --agent-image-mode is 'custom'.
            """,
    )
    verify_and_update_arg(
        parser, arg_name="--val-epoch-size", type=int, default=1024, help="Size of validation epochs."
    )
    verify_and_update_arg(
        parser,
        arg_name="--input-dir",
        default=[
            "pedestrian_intent/augmented_reharvest/10_1_20_min1p5_leak0/",
            "pedestrian_intent/augmented_reharvest/pedestrian-intent-labeling-1-11-21/",
            "pedestrian_intent/augmented_reharvest/pedestrian-intent-labeling-12-16-20-a-redo_v2/",
            "pedestrian_intent/reharvested/for_annotation/10_1_20_min1p5_leak0/",
            "pedestrian_intent/reharvested/for_annotation/10_1_20_min1p5_leak1e-3/",
            "pedestrian_intent/reharvested/for_annotation/10_1_20_min3_leak0/",
            "pedestrian_intent/reharvested/for_annotation/10_1_20_min4_leak0/",
            "harvests/pedestrian_reharvests_jp6/",
        ],
        nargs="+",
        help="Input directories with protobuf files to read.",
    )

    verify_and_update_arg(
        parser,
        arg_name="--image-dir",
        nargs="+",
        default=["harvests/pedestrian_harvest_7_24/", "harvests/pedestrian_harvest_odaiba_1_30/"],
        help="Image input directories with raw images to read.",
    )

    verify_and_update_arg(
        parser,
        "--datasets-name-lists",
        type=nullable_str,
        default="pedestrian_intent/split/split_10_timesteps.json",
        help="location of json with train/validation protobuf file lists. This will only be set if the default value of 'none' is set in trajectory_prediction_misc.py.",
    )

    return parser


def latent_factors_argument_setter(parser):
    # Base folder of the repository.
    repo_base = Path(__file__).parent.parent.parent

    parser.add_argument(
        "--latent-factors-detach", type=str2bool, default="false", help="Detach latent factors in the decoder."
    )
    parser.add_argument(
        "--latent-factors-drop", type=str2bool, default="false", help="Drop latent factors in the decoder."
    )
    parser.add_argument(
        "--explicit-duration-factors-num-segments",
        type=int,
        default=4,
        help="Explicit duration factors number of segments.",
    )
    parser.add_argument(
        "--explicit-duration-factors-transition-scale",
        type=float,
        default=0.25,
        help="Explicit duration factors timescale for transitions.",
    )
    parser.add_argument(
        "--latent-factors-type",
        type=str,
        default="explicit",
        choices=["none", "const", "explicit", "attention", "causality"],
        help="Type of latent factors.",
    )
    parser.add_argument(
        "--latent-factors-use-linear-layers",
        type=str2bool,
        default="false",
        help="Use linear layers instead of MLP for latent factors layers.",
    )
    parser.add_argument(
        "--latent-factors-internal-dim",
        nargs="+",
        type=int,
        default=[8, 8],
        help="Dimensions of latent factor emitters.",
    )
    parser.add_argument(
        "--latent-factors-internal-dim2",
        nargs="+",
        type=int,
        default=[8, 8],
        help="Dimensions of another set of latent factor layers",
    )
    parser.add_argument(
        "--full-spatiotemporal-attention",
        type=str2bool,
        default="false",
        help="Set the attention to full spatial x agents.",
    )
    parser.add_argument(
        "--latent-factors-output-dim",
        type=int,
        default=1,
        help="Dimension of the latent factor output.",
    )
    parser.add_argument(
        "--latent-factors-attention-dim",
        type=int,
        default=4,
        help="Dimension of the attention latent factor inner dimension.",
    )
    parser.add_argument(
        "--latent-factors-num-attention-heads",
        type=int,
        default=1,
        help="Number of attention heads for the attention latent factor.",
    )
    parser.add_argument(
        "--stdev-regularization-coeff",
        type=float,
        default=1e0,
        help="The coefficient of the standard deviation regularization.",
    )
    parser.add_argument("--semantic-term-coeff", type=float, default=3.0, help="Coefficient for semantic labels.")
    parser.add_argument(
        "--latent-factors-file",
        type=str,
        default=f"{repo_base}/intent/multiagents/latent_factors_subset.json",
        help="Definitions file for latent factors.",
    )
    parser.add_argument("--use-latent-factors", type=str2bool, default="true", help="Use latent factors in prediction.")
    parser.add_argument("--use-semantics", type=str2bool, default="false", help="Use semantics in prediction.")
    # turned on for this case.
    parser.add_argument(
        "--map-sampling-length", type=float, default=1.0, help="Typical length to resample map elements."
    )
    parser.add_argument(
        "--map-sampling-minimum-length", type=float, default=0.5, help="Minimum length to resample map elements."
    )
    parser.add_argument(
        "--runner-test-vehicle-filter",
        type=str2bool,
        default="False",
        help="Use protobuf instance info to filter relevant prediction instance for evaluation.",
    )

    return parser


def add_latent_factors(params):
    """
    Adds latent factors to the parameters
    :param params:
    :return:
    """
    assert "latent_factors_keys" not in params, "add_latent_factors() been called twice"
    latent_factors_type = params["latent_factors_type"].lower()
    if latent_factors_type == "const":
        latent_factors_keys = ["1", "2", "3"]
        latent_factors_generator = constant_latent_factors_generator
        params["latent_factors_keys"] = latent_factors_keys
        params["latent_factors_generator"] = latent_factors_generator
    elif latent_factors_type == "explicit":
        latent_factors_keys = ["1", "2", "3"]
        latent_factors_generator = explicit_duration_factors_generator
        params["latent_factors_keys"] = latent_factors_keys
        params["latent_factors_generator"] = latent_factors_generator
    elif latent_factors_type == "none":
        params["latent_factors_generator"] = None
        params["latent_factors_keys"] = []
    elif latent_factors_type == "attention":
        latent_factors_generator = attention_factors_generator
        params["latent_factors_generator"] = latent_factors_generator
        params["latent_factors_keys"] = [str(i) for i in range(params["latent_factors_num_attention_heads"])]
    elif latent_factors_type == "causality":
        if not causality_factors_generator:
            raise ValueError("Causality factors requires causality_factors_generator")
        params["latent_factors_generator"] = causality_factors_generator
        params["latent_factors_keys"] = ["0"]
    elif latent_factors_type != "custom":
        raise ValueError()

    if params["latent_factors_generator"]:
        params["latent_factors"] = params["latent_factors_generator"](
            params["latent_factors_keys"], params["predictor_hidden_state_dim"], params
        )

    return params


def set_agent_types(params: dict) -> dict:
    """Set agent types based on datasets.

    Parameters
    ----------
    params: dict
        The parameter dictionary.
    """
    if params["use_waymo_dataset"]:
        # Use Waymo-specific agent types.
        params["agent_types"] = WAYMO_AGENT_TYPES

        # Overwrite arguments for Waymo dataset.
        params["err_horizons_timepoints"] = [2.9, 4.9, 7.9]
        # Disable trajectory interpolation for Waymo.
        params["interp_type"] = ["none"]
    else:
        params["agent_types"] = DEFAULT_AGENT_TYPES

    # Specify relevant agent types to train and evaluate, and convert to integers.
    if params["relevant_agent_types"] is not None:
        relevant_agent_types = [AGENT_TYPES_MAP[name] for name in params["relevant_agent_types"]]
        params["relevant_agent_types"] = relevant_agent_types
    return params


@typechecked
def add_env_var_to_params(params: Dict, env_var: str) -> None:
    if "ENV" not in params:
        params["ENV"] = {}
    env = params["ENV"]

    if env_var in env:
        raise ValueError(f"Overriding parameter {env_var}")

    val = os.environ.get(env_var)
    if val:
        env[env_var] = val


@typechecked
def add_aws_env_params(params: Dict) -> None:
    add_env_var_to_params(params, "BATCH_JOB_NAME")
    add_env_var_to_params(params, "AWS_BATCH_JOB_ID")
    add_env_var_to_params(params, "AWS_BATCH_JOB_ARRAY_INDEX")

    env = params["ENV"]
    if "AWS_BATCH_JOB_ID" in env:
        base_url = "https://console.aws.amazon.com/batch/home?region=us-east-1#jobs"
        job_id = env["AWS_BATCH_JOB_ID"]
        if "AWS_BATCH_JOB_ARRAY_INDEX" in env:
            params["batch_parent_job_url"] = f"{base_url}/array-job/{job_id}"
            params["batch_job_url"] = f"{base_url}/detail/{job_id}:{env['AWS_BATCH_JOB_ARRAY_INDEX']}"
        else:
            params["batch_job_url"] = f"{base_url}/detail/{job_id}"


@typechecked
def prepare_pedestrian_model_params(params: dict):
    if params["data_debug_mode"]:
        params["data_debug_mode"] = generate_synthetic_data
    params = set_agent_types(params)
    params = add_latent_factors(params)
    add_aws_env_params(params)

    return params
