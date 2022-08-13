import copy
import os
import tempfile
import unittest
from collections import OrderedDict

import data_sources.augment_protobuf_with_language as lang_data
from intent.multiagents.language import run_language_trajectory_prediction as lang_run
from intent.multiagents.language import train_language_trajectory_prediction as lang_train
from intent.multiagents.language.language_argument_setter import language_prediction_setter
from intent.multiagents.language.language_utils import create_language_models
from intent.multiagents.language.logging_handlers_language import LanguageVisualizer
from intent.multiagents.parameter_utils import define_pretraining_parameter_sets, get_torch_device
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    DEFAULT_AGENT_TYPES,
    add_latent_factors,
    latent_factors_argument_setter,
)
from intent.multiagents.training_utils import perform_training_schedule, prepare_cache
from intent.trajectory_prediction_misc import trajectory_pred_argument_setter
from radutils.reproducibility import seed_everything
from triceps.protobuf.prediction_dataset_synthetic import generate_synthetic_data
from triceps.protobuf.proto_arguments import expand_user_path_params, parse_arguments


class TestLanguageTrainingPipeline(unittest.TestCase):
    def test_language_training(self):
        """Tests language training on synthetic data."""

        tmp_log_dir = tempfile.TemporaryDirectory(prefix="rad_test_language_predictor_")

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
                "--pretraining-mode",
                "false",
                "--max-files",
                "500",
                "--max-agents",
                "1",
                "--scene-image-mode",
                "none",
                "--agent-image-mode",
                "none",
                "--past-timesteps",
                "5",
                "--future-timesteps",
                "5",
                "--past-timestep-size",
                "0.1",
                "--past-timestep-size",
                "0.1",
                "--epoch-size",
                "1",
                "--raw-l2-for-mon",
                "--ego-agent-only",
                "True",
                "--l2-term-coeff",
                "0.0",
                "--augment-trajectories",
                "False",
                "--MoN-number-samples",
                "6",
                "--mon-term-coeff",
                "1.0",
                "--disable-map-input",
                "--encoder-normalized-trajectory-only",
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
                "--num-visualization-worst-cases",
                "1",
                "--use-language",
                "true",
                "--use-semantics",
                "false",
                "--max-language-tokens",
                "10",
                "--latent-factors-type",
                "none",
                "--disable-cache",
            ],
            additional_arguments_setter=[
                trajectory_pred_argument_setter,
                latent_factors_argument_setter,
                language_prediction_setter,
            ],
        )

        params_original = vars(args)

        # Make test as deterministic as possible.
        seed_everything(0)

        expand_user_path_params(params_original, ["latent_factors_file"])

        if params_original["data_debug_mode"]:
            params_original["data_debug_mode"] = generate_synthetic_data
        params_original["agent_types"] = DEFAULT_AGENT_TYPES

        # Add latent factors usage
        full_param = copy.deepcopy(params_original)
        full_param = add_latent_factors(full_param)

        full_param = prepare_cache(full_param)

        full_param = create_language_models(full_param)

        param_sets = OrderedDict()
        param_sets["full_params"] = full_param

        logging_handlers = [LanguageVisualizer(full_param, full_param["additional_trainer_callbacks"][0])]

        if params_original["pretraining_mode"]:
            param_sets = define_pretraining_parameter_sets(full_param)

        if params_original["pretraining_relative_lengths"] is not None:
            assert len(params_original["pretraining_relative_lengths"]) == len(param_sets) - 1

        # Run test on CUDA if possible.
        device = get_torch_device()
        perform_training_schedule(param_sets, device, logging_handlers=logging_handlers)


def test_language_files():
    # The most minimal test possible: make sure imports work
    _ = lang_data, lang_run, lang_train


if __name__ == "__main__":
    unittest.main()
