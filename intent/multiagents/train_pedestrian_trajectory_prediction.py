import importlib

from intent.multiagents.latent_factors import LatentFactorsCallback
from intent.multiagents.parameter_utils import get_torch_device
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    latent_factors_argument_setter,
    pedestrian_prediction_setter,
    prepare_pedestrian_model_params,
)
from intent.multiagents.training_utils import perform_training_schedule
from intent.trajectory_prediction_misc import trajectory_pred_argument_setter
from triceps.protobuf.proto_arguments import args_to_dict, parse_arguments


def main(args):
    full_param = args_to_dict(args, additional_file_path_params=["latent_factors_file"])
    full_param = prepare_pedestrian_model_params(full_param)

    device = get_torch_device(full_param)

    if full_param["use_semantics"]:
        full_param["latent_factors_trainer_callback"] = LatentFactorsCallback(full_param)

    if full_param["pretraining_mode"]:
        assert "pretraining_definition" in full_param and (not full_param["pretraining_definition"] == "")
        tokens = full_param["pretraining_definition"].split(".")
        pretraining_definition_module = ".".join(tokens[:-1])
        pretraining_definition_function = tokens[-1]
        module = importlib.import_module(pretraining_definition_module)
        pretraining_func = getattr(module, pretraining_definition_function)
        param_sets = pretraining_func(full_param)
    else:
        param_sets = {"full_params": full_param}

    return perform_training_schedule(param_sets, device)


def parse_args(args=None, additional_arguments_setter=None):
    if not additional_arguments_setter:
        additional_arguments_setter = []
    try:
        from tristan.predictors.causal import CausalModule

        additional_arguments_setter.append(CausalModule.add_model_specific_args)
    except:
        pass
    return parse_arguments(
        args,
        additional_arguments_setter=[
            *additional_arguments_setter,
            trajectory_pred_argument_setter,
            latent_factors_argument_setter,
            pedestrian_prediction_setter,
        ],
    )


if __name__ == "__main__":
    main(parse_args())
