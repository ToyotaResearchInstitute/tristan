import torch

from intent.multiagents.language.language_argument_setter import language_inference_setter, language_prediction_setter
from intent.multiagents.language.language_utils import create_language_models
from intent.multiagents.language.logging_handlers_language import SaveLanguageErrorStatistics
from intent.multiagents.logging_handlers import LogWorstCases
from intent.multiagents.parameter_utils import inference_only_setter
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    add_latent_factors,
    latent_factors_argument_setter,
    pedestrian_prediction_setter,
    set_agent_types,
)
from intent.multiagents.training_utils import perform_data_inference, prepare_cache
from intent.trajectory_prediction_misc import trajectory_pred_argument_setter
from triceps.protobuf.prediction_dataset_synthetic import generate_synthetic_data
from triceps.protobuf.proto_arguments import expand_user_path_params, parse_arguments

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    args = parse_arguments(
        additional_arguments_setter=[
            trajectory_pred_argument_setter,
            latent_factors_argument_setter,
            pedestrian_prediction_setter,
            inference_only_setter,
            language_prediction_setter,
            language_inference_setter,
        ]
    )
    params = vars(args)

    expand_user_path_params(params, ["latent_factors_file"])

    if params["data_debug_mode"]:
        params["data_debug_mode"] = generate_synthetic_data

    params = set_agent_types(params)

    # Add latent factors usage
    params = add_latent_factors(params)

    params = prepare_cache(params)

    params = create_language_models(params)

    worst_cases_folder = params["runner_output_folder"] if params["logger_type"] == "none" else None

    # Here we add callbacks to log statistics and plot things as we iterate over the dataset.
    additional_logging_handlers = [
        LogWorstCases("g_cost", params, worst_cases_folder),
        LogWorstCases("distance_x0", params, worst_cases_folder),
        SaveLanguageErrorStatistics(params, params["additional_trainer_callbacks"][0], params["output_folder_name"]),
    ]

    for agent_idx in range(params["max_agents"]):
        logger_key = f"fde_agent_{agent_idx}"
        additional_logging_handlers.append(LogWorstCases(logger_key, params, worst_cases_folder))

    perform_data_inference(params, device, additional_logging_handlers)
