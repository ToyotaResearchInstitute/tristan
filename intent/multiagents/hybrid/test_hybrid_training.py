import copy
import os
import tempfile
import unittest
from collections import OrderedDict
from typing import Dict

from intent.multiagents.hybrid import logging_handlers_hybrid as hybrid_log
from intent.multiagents.hybrid import run_hybrid_vehicle_trajectory_prediction as hybrid_run
from intent.multiagents.hybrid import train_hybrid_vehicle_prediction as hybrid_train
from intent.multiagents.hybrid.hybrid_argument_setter import (
    hybrid_argoverse_prediction_argument_setter,
    hybrid_prediction_argument_setter,
)
from intent.multiagents.hybrid.hybrid_dataset_utils import generate_hybrid_synthetic_data
from intent.multiagents.hybrid.hybrid_models import HybridFactors
from intent.multiagents.hybrid.hybrid_prediction_utils import (
    HybridGenerationCost,
    HybridModelCallback,
    HybridTrainerCallback,
)
from intent.multiagents.hybrid.prediction_dataset_hybrid_handler import DiscreteModeLabelHandler
from intent.multiagents.parameter_utils import define_pretraining_parameter_sets, get_torch_device
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    DEFAULT_AGENT_TYPES,
    add_latent_factors,
    latent_factors_argument_setter,
)
from intent.multiagents.training_utils import perform_training_schedule, prepare_cache
from intent.trajectory_prediction_misc import trajectory_pred_argument_setter
from radutils.reproducibility import seed_everything
from triceps.protobuf.proto_arguments import expand_user_path_params, parse_arguments


# Add hybrid model.
def hybrid_factors_generator(keys, intermediate_dim, params):
    return HybridFactors(params)


def hybrid_params() -> Dict:
    tmp_log_dir = tempfile.TemporaryDirectory(prefix="rad_test_hybrid_predictor_")

    try:
        args_dir = ["--data-debug-mode", "true"]
    except:
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        input_training_dir = os.path.join(repo_dir, "data_sources", "argoverse", "dummy_data", "training")
        input_validation_dir = os.path.join(repo_dir, "data_sources", "argoverse", "dummy_data", "validation")
        args_dir = [
            "--input-training-dir",
            input_training_dir,
            "--input-validation-dir",
            input_validation_dir,
            "--data-debug-mode",
            "true",
        ]

    args = parse_arguments(
        args=args_dir
        + [
            "--past-timesteps",
            "5",
            "--future-timesteps",
            "5",
            "--past-timestep-size",
            "0.1",
            "--future-timestep-size",
            "0.1",
            "--epoch-size",
            "1",
            "--learn-discrete-proposal",
            "--disable-map-input",
            "--logs-dir",
            tmp_log_dir.name,
            "--vis-epoch-size",
            "1",
            "--val-epoch-size",
            "1",
            "--batch-size",
            "2",
            "--val-batch-size",
            "1",
            "--vis-batch-size",
            "1",
            "--num-epochs",
            "1",
            "--disable-model-saving",
            "true",
            "--use-waymo-dataset",
            "false",
            "--num-visualization-worst-cases",
            "1",
            "--disable-cache",
        ],
        additional_arguments_setter=[
            trajectory_pred_argument_setter,
            latent_factors_argument_setter,
            hybrid_prediction_argument_setter,
            hybrid_argoverse_prediction_argument_setter,
        ],
    )

    params_original = vars(args)
    params_original["agent_types"] = DEFAULT_AGENT_TYPES

    # Make test as deterministic as possible.
    seed_everything(0)

    expand_user_path_params(params_original, ["latent_factors_file"])

    if params_original["data_debug_mode"]:
        params_original["data_debug_mode"] = generate_hybrid_synthetic_data
    params_original = prepare_cache(params_original)

    # Add latent factors usage
    full_param = copy.deepcopy(params_original)
    full_param = add_latent_factors(full_param)

    # Add hybrid specific handlers.
    full_param["discrete_mode_handler"] = DiscreteModeLabelHandler(params_original)
    full_param["additional_structure_callbacks"] = [HybridGenerationCost()]
    full_param["additional_model_callbacks"] = [HybridModelCallback()]
    full_param["additional_trainer_callbacks"] = [HybridTrainerCallback(full_param)]

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

    return param_sets


class TestHybridTrainingPipeline(unittest.TestCase):
    def test_hybrid_training(self):
        """Tests hybrid training on synthetic data."""

        param_sets = hybrid_params()
        # Run test on CUDA if possible.
        device = get_torch_device()
        perform_training_schedule(param_sets, device)


def test_hybrid_files():
    # The most minimal test possible: make sure imports work
    _ = hybrid_log, hybrid_run, hybrid_train


if __name__ == "__main__":
    unittest.main()
