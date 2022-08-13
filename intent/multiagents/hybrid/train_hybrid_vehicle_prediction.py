import copy
from collections import OrderedDict

from intent.multiagents.hybrid.hybrid_argument_setter import (
    hybrid_argoverse_prediction_argument_setter,
    hybrid_prediction_argument_setter,
)
from intent.multiagents.hybrid.hybrid_models import HybridFactors
from intent.multiagents.hybrid.hybrid_prediction_utils import (
    HybridGenerationCost,
    HybridModelCallback,
    HybridTrainerCallback,
)
from intent.multiagents.hybrid.prediction_dataset_hybrid_handler import DiscreteModeLabelHandler
from intent.multiagents.parameter_utils import define_pretraining_parameter_sets, get_torch_device
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    add_latent_factors,
    latent_factors_argument_setter,
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
            hybrid_prediction_argument_setter,
            hybrid_argoverse_prediction_argument_setter,
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

    # Add hybrid specific handlers.
    full_param["discrete_mode_handler"] = DiscreteModeLabelHandler(full_param)
    full_param["additional_structure_callbacks"] = [HybridGenerationCost()]
    full_param["additional_model_callbacks"] = [HybridModelCallback()]
    full_param["additional_trainer_callbacks"] = [HybridTrainerCallback(full_param)]

    # Add hybrid dataset caching parameters.
    cache_param_list = ["compute_maneuvers_online", "hybrid_smooth_mode"]
    full_param["additional_cache_param_list"] = cache_param_list

    # Add hybrid model.
    def hybrid_factors_generator(keys, intermediate_dim, params):
        return HybridFactors(params)

    full_param["latent_factors_generator"] = hybrid_factors_generator
    intermediate_dim = full_param["predictor_hidden_state_dim"]
    full_param["latent_factors"] = full_param["latent_factors_generator"](
        full_param["latent_factors_keys"], intermediate_dim, full_param
    )
    param_sets = OrderedDict()
    param_sets["full_params"] = full_param

    if params_original["pretraining_mode"]:
        param_sets = define_pretraining_parameter_sets(full_param)

    if params_original["pretraining_relative_lengths"] is not None:
        assert len(params_original["pretraining_relative_lengths"]) == len(param_sets) - 1

    perform_training_schedule(param_sets, device)
