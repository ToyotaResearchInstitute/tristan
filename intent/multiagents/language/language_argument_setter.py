import copy
from collections import OrderedDict
from pathlib import Path

from intent.multiagents.training_utils import update_num_epochs_from_list
from intent.trajectory_prediction_misc import str2bool


def language_prediction_setter(parser):
    """Add flags for language-based predictors.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Parser for the arguments.

    Returns
    -------
    parser: argparse.ArgumentParser
        The updated parser with language specific arguments.
    """
    repo_base = Path(__file__).parent.parent.parent.parent
    parser.add_argument("--use-language", type=str2bool, default="true", help="Use language in prediction.")
    parser.add_argument(
        "--use-annotated-captions",
        type=str2bool,
        default="false",
        help="Use annotated captions in the language-based predictor.",
    )
    parser.add_argument(
        "--source-captions-dir",
        type=str,
        default="waymo/captions/",
        help="The path that contains the annotated caption.",
    )
    parser.add_argument(
        "--skip-out-agent-tokens",
        type=str2bool,
        default="true",
        help="Skip the sentence that mentions the agent index exceeding `max-agent` in the handler.",
    )
    parser.add_argument(
        "--token-generator-noise-dim",
        type=int,
        default=1,
        help="The size of noise vector for the language token generator.",
    )
    parser.add_argument(
        "--token-generator-hidden-size",
        type=int,
        default=8,
        help="Dimension of the hidden states for the token generator.",
    )
    parser.add_argument(
        "--token-generator-coeff", type=float, default=1.0, help="Weight for the token generation loss."
    )
    parser.add_argument(
        "--token-encoder-input-size", type=int, default=8, help="Embedding dimension for the token encoder."
    )
    parser.add_argument(
        "--token-encoder-hidden-size",
        type=int,
        default=8,
        help="Dimension of the hidden states for the token encoder.",
    )
    parser.add_argument(
        "--unfreeze-token-embedding-epoch",
        type=int,
        default=-1,
        help="The epoch number to unfreeze the pretrained token embedding parameters. Do not unfreeze if set to -1.",
    )
    parser.add_argument(
        "--language-layer-norm",
        type=str2bool,
        default="false",
        help="Use layer norm in token generator and encoder LSTMs.",
    )
    parser.add_argument(
        "--language-use-mlp",
        type=str2bool,
        default="false",
        help="Use MLP instead of linear layers in token generator and encoder.",
    )
    parser.add_argument(
        "--language-dropout-ratio",
        type=float,
        default=0.0,
        help="The dropout ratio in the language factors.",
    )
    parser.add_argument(
        "--language-ablate-attention",
        type=str2bool,
        default="false",
        help="Do not use attention in language factors.",
    )
    parser.add_argument(
        "--drop-language-output",
        type=str2bool,
        default="false",
        help="Drop output of language factors in inference time.",
    )
    parser.add_argument(
        "--language-ablate-agent-attention",
        type=str2bool,
        default="false",
        help="Do not use attention for agents in language factors.",
    )
    parser.add_argument(
        "--language-attention-size",
        type=int,
        default=5,
        help="Size of language attention layers.",
    )
    parser.add_argument(
        "--language-input-train-ratio",
        type=float,
        default=1.0,
        help="Percentage of language groundtruth used in training.",
    )
    parser.add_argument(
        "--caption-typo-csv",
        type=str,
        default=f"{repo_base}/intent/multiagents/language/data/typo_corrections.csv",
        help="CSV files to correct typos for language captions.",
    )
    return parser


def language_inference_setter(parser):
    """Add flags for language-based predictors (inference-only).

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Parser for the arguments.

    Returns
    -------
    parser: argparse.ArgumentParser
        The updated parser with language specific arguments.
    """
    parser.add_argument(
        "--visualize-prediction", type=str2bool, default="false", help="Visualize predictions for language attentions."
    )
    parser.add_argument(
        "--compute-information-gain", type=str2bool, default="false", help="Compute information gain for language."
    )
    parser.add_argument(
        "--output-folder-name",
        type=str,
        default="intent/data_runner_results/output",
        help="The output folder name for the images and other data.",
    )
    return parser


def define_language_pretraining_parameter_sets(params_original: dict) -> OrderedDict:
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

    # with gnn edges, maps, but no language factor
    dynamics_only_param = copy.deepcopy(params_original)
    dynamics_only_param["latent_factors_detach"] = True
    update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
    param_key = "no_maps"
    param_sets[param_key] = copy.deepcopy(dynamics_only_param)
    param_sets.move_to_end(param_key, False)
    dynamics_only_param["learning_rate"] = 1e-3

    # Note we also change dataloaders, batch sizes, etc, for training efficiency.
    dynamics_only_param["batch_size"] = 64
    dynamics_only_param["val_batch_size"] = 64
    dynamics_only_param["vis_batch_size"] = 16
    dynamics_only_param["disable_map_input"] = True
    dynamics_only_param["add_agent_map_input_stub"] = True
    dynamics_only_param["agent_map_input_stub_dim"] = 7  # map point attention
    dynamics_only_param["agent_map_input_stub_embed_dim"] = 32  # map-layer-features
    update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
    param_key = "no_maps_no_lang"
    param_sets[param_key] = copy.deepcopy(dynamics_only_param)
    param_sets.move_to_end(param_key, False)

    dynamics_only_param["discriminator_term_coeff"] = 0.0
    dynamics_only_param["disable_discriminator_update"] = True
    dynamics_only_param["batch_size"] = 64
    dynamics_only_param["val_batch_size"] = 64
    dynamics_only_param["vis_batch_size"] = 16
    update_num_epochs_from_list(dynamics_only_param, params_original, param_sets, 16)
    param_key = "no_maps_no_gan"
    param_sets[param_key] = copy.deepcopy(dynamics_only_param)
    param_sets.move_to_end(param_key, False)

    return param_sets
