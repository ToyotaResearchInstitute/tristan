import functools

from intent.multiagents.hybrid.hybrid_argument_setter import (
    hybrid_argoverse_prediction_argument_setter,
    hybrid_prediction_argument_setter,
)
from intent.multiagents.hybrid.hybrid_models import HybridFactors
from intent.multiagents.hybrid.hybrid_prediction_utils import (
    HybridGenerationCost,
    HybridModelCallback,
    HybridTrainerCallback,
    subsample_hybrid_fps,
    subsample_hybrid_nms,
    subsample_hybrid_random,
    subsample_hybrid_top_k,
)
from intent.multiagents.hybrid.logging_handlers_hybrid import SaveHybridErrorStatistics
from intent.multiagents.hybrid.prediction_dataset_hybrid_handler import DiscreteModeLabelHandler
from intent.multiagents.logging_handlers import LogWorstCases
from intent.multiagents.parameter_utils import get_torch_device, inference_only_setter
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    add_latent_factors,
    latent_factors_argument_setter,
    set_agent_types,
)
from intent.multiagents.training_utils import perform_data_inference, prepare_cache
from intent.trajectory_prediction_misc import str2bool, trajectory_pred_argument_setter
from triceps.protobuf.prediction_dataset_synthetic import generate_synthetic_data
from triceps.protobuf.proto_arguments import expand_user_path_params, parse_arguments


def hybrid_inference_only_setter(parser):
    """
    Add argument setter for hybrid inference.
    """
    parser.add_argument(
        "--hybrid-runner-subsample-size", type=int, default=6, help="Number of subsamples in final predictions."
    )
    parser.add_argument(
        "--hybrid-runner-nms-dist-threshold", type=float, default=2.0, help="Distance threshold used in NMS."
    )
    parser.add_argument(
        "--hybrid-runner-dist-type", type=str, default="final", help="Distance type used in NMS/FPS (avg/final)."
    )
    parser.add_argument(
        "--hybrid-runner-full-dataset", type=str2bool, default="false", help="Whether to train on a full dataset."
    )
    parser.add_argument(
        "--hybrid-runner-visualize", type=str2bool, default="false", help="Whether to visualize examples."
    )
    parser.add_argument(
        "--hybrid-runner-save", type=str2bool, default="false", help="Whether to save prediction results."
    )
    parser.add_argument("--hybrid-runner-fps-only", type=str2bool, default="false", help="Whether to use FPS only.")
    parser.add_argument(
        "--hybrid-runner-save-single",
        type=str2bool,
        default="false",
        help="Whether to save a single file (for Argoverse test).",
    )

    return parser


if __name__ == "__main__":
    args = parse_arguments(
        additional_arguments_setter=[
            trajectory_pred_argument_setter,
            latent_factors_argument_setter,
            hybrid_prediction_argument_setter,
            inference_only_setter,
            hybrid_argoverse_prediction_argument_setter,
            hybrid_inference_only_setter,
        ]
    )
    params = vars(args)
    device = get_torch_device(params)
    expand_user_path_params(params, ["latent_factors_file"])

    if params["data_debug_mode"]:
        params["data_debug_mode"] = generate_synthetic_data

    params = set_agent_types(params)

    # Add latent factors usage
    params = add_latent_factors(params)
    params = prepare_cache(params)

    # Add hybrid specific handlers.
    params["discrete_mode_handler"] = DiscreteModeLabelHandler(params)
    params["additional_structure_callbacks"] = [HybridGenerationCost()]
    params["additional_model_callbacks"] = [HybridModelCallback()]
    params["additional_trainer_callbacks"] = [HybridTrainerCallback(params)]

    # Add hybrid dataset caching parameters.
    cache_param_list = ["compute_maneuvers_online", "hybrid_smooth_mode"]
    params["additional_cache_param_list"] = cache_param_list

    # Add hybrid model.
    def hybrid_factors_generator(keys, intermediate_dim, params):
        return HybridFactors(params)

    params["latent_factors_generator"] = hybrid_factors_generator
    intermediate_dim = params["predictor_hidden_state_dim"]
    params["latent_factors"] = params["latent_factors_generator"](
        params["latent_factors_keys"], intermediate_dim, params
    )

    # Here we add callbacks to log statistics and plot things as we iterate over the dataset.
    dist_type = params["hybrid_runner_dist_type"]
    if params["hybrid_runner_fps_only"]:
        samplers = [functools.partial(subsample_hybrid_fps, dist_type=dist_type)]
    else:
        samplers = [
            functools.partial(subsample_hybrid_fps, dist_type=dist_type),
            functools.partial(subsample_hybrid_nms, dist_threshold=2, dist_type=dist_type),
            functools.partial(subsample_hybrid_nms, dist_threshold=4, dist_type=dist_type),
            subsample_hybrid_top_k,
            subsample_hybrid_random,
        ]
    additional_logging_handlers = [
        SaveHybridErrorStatistics(
            params=params,
            prefix="prefix",
            subsamplers=samplers,
            subsamplers_names=["fps", "nms2", "nms4", "top_k", "random"],
            visualize=args.hybrid_runner_visualize,
            save_results=args.hybrid_runner_save,
            save_single_results=args.hybrid_runner_save_single,
        ),
    ]

    worst_cases_folder = params["runner_output_folder"] if params["logger_type"] == "none" else None
    for agent_idx in range(params["max_agents"]):
        logger_key = f"fde_agent_{agent_idx}"
        additional_logging_handlers.append(LogWorstCases(logger_key, params, worst_cases_folder))
    perform_data_inference(params, device, additional_logging_handlers)
