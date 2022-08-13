import copy
from collections import OrderedDict

from intent.multiagents.language.language_argument_setter import (
    define_language_pretraining_parameter_sets,
    language_prediction_setter,
)
from intent.multiagents.language.language_utils import create_language_models
from intent.multiagents.language.logging_handlers_language import LanguageVisualizer
from intent.multiagents.parameter_utils import get_torch_device
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    add_latent_factors,
    latent_factors_argument_setter,
    pedestrian_prediction_setter,
    set_agent_types,
)
from intent.multiagents.training_utils import perform_training_schedule, prepare_cache
from intent.trajectory_prediction_misc import trajectory_pred_argument_setter
from triceps.protobuf.prediction_dataset_synthetic import generate_synthetic_data
from triceps.protobuf.proto_arguments import expand_user_path_params, parse_arguments

if __name__ == "__main__":
    args = parse_arguments(
        additional_arguments_setter=[
            trajectory_pred_argument_setter,
            latent_factors_argument_setter,
            pedestrian_prediction_setter,
            language_prediction_setter,
        ]
    )
    params_original = vars(args)
    device = get_torch_device(params_original)

    expand_user_path_params(params_original, ["latent_factors_file"])

    if params_original["data_debug_mode"]:
        params_original["data_debug_mode"] = generate_synthetic_data

    params_original = set_agent_types(params_original)

    # Add latent factors usage
    full_param = copy.deepcopy(params_original)
    full_param = add_latent_factors(full_param)
    full_param = prepare_cache(full_param)

    full_param = create_language_models(full_param)

    param_sets = OrderedDict()
    param_sets["full_params"] = full_param

    # Do not visualize trajectories by tokens if using annotations.
    if full_param["use_annotated_captions"]:
        logging_handlers = []
    else:
        logging_handlers = [LanguageVisualizer(full_param, full_param["additional_trainer_callbacks"][0])]

    if params_original["pretraining_mode"]:
        param_sets = define_language_pretraining_parameter_sets(full_param)

    if params_original["pretraining_relative_lengths"] is not None:
        assert len(params_original["pretraining_relative_lengths"]) == len(param_sets) - 1

    perform_training_schedule(param_sets, device, logging_handlers=logging_handlers)
