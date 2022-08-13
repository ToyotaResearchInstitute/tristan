import copy
import multiprocessing
from collections import OrderedDict
from typing import Optional

import torch

from intent.multiagents.training_utils import MAP_INPUT_SIZE, update_num_epochs_from_list
from intent.trajectory_prediction_misc import str2bool

DEFAULT_NUM_WORKERS = min(32, multiprocessing.cpu_count())


def define_pretraining_parameter_sets_2step(params_original: dict) -> OrderedDict:
    """Defines a list of parameter sets to allow curriculum training. The pretraining starts from a simplified network,
    which are trained with bigger batches and faster. Note that the order of execution is reverse from definition
    (i.e the last phase to train is defined first.)

    Parameters
    ----------
    params_original: dict
        The full, original, parameter dictionary.

    Returns
    -------
    params_sets: OrderedDict
        An ordered dictionary of parameter sets to train.
    """
    param_sets = OrderedDict()
    # Full parameter set
    param_sets["full_params"] = copy.deepcopy(params_original)

    # No image/map, no GAN - just dynamics.
    dynamics_only_param = copy.deepcopy(params_original)
    dynamics_only_param["use_semantics"] = False
    dynamics_only_param["discriminator_term_coeff"] = 0.0
    dynamics_only_param["disable_discriminator_update"] = True
    dynamics_only_param["add_agent_input_stub"] = True
    dynamics_only_param["add_input_stub"] = True
    dynamics_only_param["input_stub_embed_size"] = params_original["image_embed_size"]
    dynamics_only_param["agent_input_stub_embed_size"] = params_original["image_embed_size"]
    dynamics_only_param["agent_image_mode"] = "none"
    dynamics_only_param["scene_image_mode"] = "none"
    dynamics_only_param["disable_map_input"] = True
    dynamics_only_param["add_agent_map_input_stub"] = True
    dynamics_only_param["agent_map_input_stub_dim"] = MAP_INPUT_SIZE
    dynamics_only_param["agent_map_input_stub_embed_dim"] = 32
    update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
    param_key = "no_images_no_map_no_gan"
    param_sets[param_key] = copy.deepcopy(dynamics_only_param)
    param_sets.move_to_end(param_key, False)

    if params_original["pretraining_relative_lengths"] is not None:
        assert len(params_original["pretraining_relative_lengths"]) == len(param_sets) - 1

    return param_sets


def define_pretraining_parameter_sets_2step_map_only(params_original: dict) -> OrderedDict:
    """Defines a list of parameter sets to allow curriculum training using map-only data (no image input).
    The pretraining starts from a simplified network, which are trained with bigger batches and faster.
    Note that the order of execution is reverse from definition (i.e the last phase to train is defined first.)

    Parameters
    ----------
    params_original: dict
        The full, original, parameter dictionary.

    Returns
    -------
    params_sets: OrderedDict
        An ordered dictionary of parameter sets to train.
    """
    param_sets = OrderedDict()
    # Full parameter set
    param_sets["full_params"] = copy.deepcopy(params_original)
    dynamics_only_param = copy.deepcopy(params_original)

    # No cross attention across agent if it is transformer-based model - no attending to other agents
    if params_original["encoder_decoder_type"] == "transformer":
        dynamics_only_param["num_encoder_transformer_agent_attn_skips"] = params_original[
            "num_encoder_transformer_blocks"
        ]
        update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
        param_key = "no_cross_agent"
        param_sets[param_key] = copy.deepcopy(dynamics_only_param)
        param_sets.move_to_end(param_key, False)

    # No map, no GAN - just dynamics.
    dynamics_only_param["discriminator_term_coeff"] = 0.0
    dynamics_only_param["disable_discriminator_update"] = True
    dynamics_only_param["add_input_map_stub"] = True
    dynamics_only_param["input_map_stub_dim"] = MAP_INPUT_SIZE + 2  # map input + traj.
    dynamics_only_param["input_map_stub_embed_size"] = params_original["map_layer_features"][0]
    dynamics_only_param["map_points_max"] = 50  # a dummy number of map segments to stub
    dynamics_only_param["disable_map_input"] = True
    update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
    param_key = "no_map_no_gan"
    param_sets[param_key] = copy.deepcopy(dynamics_only_param)
    param_sets.move_to_end(param_key, False)

    if params_original["pretraining_relative_lengths"] is not None:
        assert len(params_original["pretraining_relative_lengths"]) == len(param_sets) - 1

    return param_sets


def define_pretraining_parameter_sets(params_original: dict) -> OrderedDict:
    """Defines a list of parameter sets to allow curriculum training. The pretraining starts from a simplified network,
    which are trained with bigger batches and faster. Note that the order of execution is reverse from definition
    (i.e the last phase to train is defined first.)

    Parameters
    ----------
    params_original: dict
        The full, original, parameter dictionary.

    Returns
    -------
    params_sets: OrderedDict
        An ordered dictionary of parameter sets to train.
    """
    param_sets = OrderedDict()
    # full parameter set
    param_sets["full_params"] = copy.deepcopy(params_original)

    if params_original["pretraining_relative_lengths"] is not None:
        assert len(params_original["pretraining_relative_lengths"]) == len(param_sets) - 1

    # with gnn edges, maps, but no images
    dynamics_only_param = copy.deepcopy(params_original)
    dynamics_only_param["scene_inputs_detach"] = True
    dynamics_only_param["agent_inputs_detach"] = True
    dynamics_only_param["latent_factors_drop"] = True
    dynamics_only_param["latent_factors_detach"] = True
    update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
    dynamics_only_param["learning_rate"] = 1e-3

    # Note we also change dataloaders, batch sizes, etc, for training efficiency.
    dynamics_only_param["batch_size"] = 64
    dynamics_only_param["val_batch_size"] = 64
    dynamics_only_param["vis_batch_size"] = 16
    dynamics_only_param["add_agent_input_stub"] = True
    dynamics_only_param["add_input_stub"] = True
    dynamics_only_param["input_stub_embed_size"] = params_original["image_embed_size"]
    dynamics_only_param["agent_input_stub_embed_size"] = params_original["image_embed_size"]
    dynamics_only_param["agent_image_mode"] = "none"
    dynamics_only_param["scene_image_mode"] = "none"
    update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
    param_key = "no_images"
    param_sets[param_key] = copy.deepcopy(dynamics_only_param)
    param_sets.move_to_end(param_key, False)

    dynamics_only_param["use_semantics"] = False
    dynamics_only_param["discriminator_term_coeff"] = 0.0
    dynamics_only_param["disable_discriminator_update"] = True
    dynamics_only_param["batch_size"] = 64
    dynamics_only_param["val_batch_size"] = 64
    dynamics_only_param["vis_batch_size"] = 16
    dynamics_only_param["add_agent_input_stub"] = True
    dynamics_only_param["add_input_stub"] = True
    dynamics_only_param["input_stub_embed_size"] = params_original["image_embed_size"]
    dynamics_only_param["agent_input_stub_embed_size"] = params_original["image_embed_size"]
    dynamics_only_param["agent_image_mode"] = "none"
    dynamics_only_param["scene_image_mode"] = "none"
    param_key = "no_images_no_gan"
    param_sets[param_key] = copy.deepcopy(dynamics_only_param)
    param_sets.move_to_end(param_key, False)

    return param_sets


def inference_only_setter(parser):
    parser.add_argument(
        "--disable-optimization", type=str2bool, default="true", help="Do not perform any optimization."
    )
    parser.add_argument(
        "--disable-tensorboard-writer",
        type=str2bool,
        default="true",
        help="Do not save tensorboard logs, no writer initialized.",
    )
    parser.add_argument(
        "--disable-model-saving", type=str2bool, default="true", help="Do not save models (e.g. for statistics runner)."
    )
    parser.add_argument(
        "--use-linear-model",
        type=str2bool,
        default="false",
        help="Instantiate a simple linear velocity projection model.",
    )
    parser.add_argument(
        "--nonstraight-walking-threshold",
        type=float,
        default=0.2,
        help="Threshold to save separately BEV images of non-straight trajectories.",
    )
    parser.add_argument(
        "--nonstraight-walking-degree", type=int, default=1, help="Degree used to fit non-straight trajectories."
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs to collect data.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=4, help="Size of training batch.")
    parser.add_argument(
        "--use-dummy-model",
        default=False,
        action="store_true",
        help="Uses an untrained network for evaluation (mostly useful for debugging).",
    )
    parser.add_argument(
        "--val-num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers."
    )
    parser.add_argument("--val-batch-size", type=int, default=4, help="Size of training batch.")
    parser.add_argument(
        "--vis-num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of dataloader workers."
    )
    parser.add_argument("--vis-batch-size", type=int, default=4, help="Size of training batch.")
    parser.add_argument(
        "--evaluation-dataset",
        type=str,
        choices=["train", "validation"],
        default="validation",
        help="On which dataset to run the evaluation.",
    )

    parser.set_defaults(full_dataset_epochs=True, val_interval=1)

    return parser


def get_torch_device(params: Optional[dict] = None) -> torch.device:
    """Get CUDA torch device if available.

    If the `--cpu-only` flag is passed through `params`, this will skip checking
    for CUDA to avoid memory initialization overhead which is useful for
    debugging and memory profiling.

    Parameters
    ----------
    params: dict or None

        The parameter dictionary, optional.
    """
    if params is None:
        params = {}
    if not params.get("cpu_only") and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device
