import copy
import random
from collections import OrderedDict, defaultdict
from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.axes
import numpy as np
import torch
import tqdm
from torch import nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

import triceps
from intent.multiagents.additional_callback import AdditionalTrainerCallback, MatplotlibColor
from intent.multiagents.additional_costs import AdditionalCostCallback
from intent.multiagents.cache_utils import split_reading_hash
from intent.multiagents.language.vocab import SPECIALS, clean_caption, get_caption_vocab, get_synthetic_vocab
from intent.multiagents.latent_factors import LatentFactorsTrainerCallback
from intent.multiagents.trainer_logging import TrainingLogger
from intent.multiagents.trainer_visualization import wrap_text
from model_zoo.intent.language_models import LanguageFactors
from radutils.misc import parse_protobuf_timestamp
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_auxiliary import InputsHandler
from triceps.protobuf.prediction_dataset_cache import CacheElement
from triceps.protobuf.prediction_dataset_semantic_handler import TYPE_LANGUAGE_TOKENS


def language_factors_generator(keys, intermediate_dim, params):
    model_param = {
        "max_token_length": params["max_language_tokens"],
        "agent_state_size": params["predictor_hidden_state_dim"] + params["latent_factors_output_dim"],
        "generator_input_size": params["predictor_hidden_state_dim"],
        "generator_hidden_size": params["token_generator_hidden_size"],
        "generator_noise_dim": params["token_generator_noise_dim"],
        "encoder_input_size": params["token_encoder_input_size"],
        "encoder_hidden_size": params["token_encoder_hidden_size"],
        "max_agents": params["max_agents"],
        "use_pretrained_word_embedding": params["use_annotated_captions"],
        "use_accelerated_decoder": params["use_multiagent_accelerated_decoder"],
        "use_layer_norm": params["language_layer_norm"],
        "use_mlp": params["language_use_mlp"],
        "dropout_ratio": params["language_dropout_ratio"],
        "ablate_attention": params["language_ablate_attention"],
        "drop_output": params["drop_language_output"],
        "ablate_agent_attention": params["language_ablate_agent_attention"],
        "attention_size": params["language_attention_size"],
        "compute_information_gain": False,
    }
    if "compute_information_gain" in params:
        model_param["compute_information_gain"] = params["compute_information_gain"]
    return LanguageFactors(model_param, params["language_vocab"])


def create_language_models(full_param) -> dict:
    """Create and initialize the required language models and handlers.

    Parameters
    ----------
    full_param: dict
        The parameter dictionary.

    Returns
    -------
    dict
        The updated dict containing the latent factors and callbacks.
    """
    # Generate vocabulary.
    if full_param["use_annotated_captions"]:
        full_param["language_vocab"], full_param["language_word_map"] = get_caption_vocab(
            full_param["source_captions_dir"], full_param["caption_typo_csv"], full_param["max_agents"]
        )
    else:
        full_param["language_vocab"] = get_synthetic_vocab(full_param["max_agents"])

    # Add language model.
    full_param["latent_factors_keys"] = ["language_factors"]
    full_param["latent_factors_output_dim"] = full_param["token_encoder_hidden_size"]
    full_param["latent_factors_generator"] = language_factors_generator
    intermediate_dim = full_param["predictor_hidden_state_dim"]
    full_param["latent_factors"] = full_param["latent_factors_generator"](
        full_param["latent_factors_keys"], intermediate_dim, full_param
    )
    full_param["latent_factors_trainer_callback"] = LanguageFactorsTrainerCallback(full_param)

    # Add language specific handlers.
    full_param["language_token_handler"] = LanguageTokensHandler(full_param)
    full_param["additional_structure_callbacks"] = [TokenGenerationCost()]
    full_param["additional_trainer_callbacks"] = [LanguageTrainerCallback(full_param)]
    return full_param


class LanguageTokensHandler(InputsHandler):
    """Process and convert the language tokens into tensors.

    Parameters
    ----------
    params: dict
        trainer parameters.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.vocab = params["language_vocab"]
        if params["use_annotated_captions"]:
            self.word_map = params["language_word_map"]
        self.tokenizer = get_tokenizer("basic_english")

    def select_semantic_target_to_token_ids(
        self, semantic_targets: list, params: dict, agent_ids: list, selected_agent_idx: list
    ):
        """Random select a semantic target from a list to convert to a token id list.

        Parameters
        ----------
        semantic_targets: list
            The candidate semantic targets.
        params: dict
            The dictionary of parameters.
        agent_ids: np.ndarray
            List of agent ids in a prediction instance.
        selected_agent_idx: list
            List of index of selected agents.

        Returns
        -------
        out_vocabs: list
            The token id list.
        """
        out_vocabs = []
        semantic_target = None
        while len(semantic_targets) > 0 and len(out_vocabs) == 0:
            semantic_target: dict = random.choice(semantic_targets)
            if params["use_annotated_captions"]:
                caption = clean_caption(semantic_target["value"], self.word_map)
                tokens = self.tokenizer(caption)
                out_tokens = []
                for token in tokens:
                    if token.isnumeric():
                        if int(token) not in selected_agent_idx:
                            token = "agent"
                        else:
                            token = str(selected_agent_idx.index(int(token)))
                    out_tokens.append(self.vocab.get_stoi()[token])
                if len(out_tokens) > 0:
                    out_tokens = [self.vocab.get_stoi()["<bos>"]] + out_tokens + [self.vocab.get_stoi()["<eos>"]]
                    out_vocabs.extend(out_tokens)
            else:
                other_agent_idx = None
                tokens = semantic_target["value"].split()
                label = tokens[0]
                if len(tokens) > 1:
                    try:
                        agent_idx = np.argwhere(agent_ids == tokens[1])[0][0]
                    except:
                        continue
                    if params["skip_out_agent_tokens"] and agent_idx not in selected_agent_idx:
                        semantic_targets.remove(semantic_target)
                        continue
                    other_agent_idx = str(selected_agent_idx.index(agent_idx))
                out_vocabs.append(self.vocab.get_stoi()[label])
                if other_agent_idx is not None:
                    out_vocabs.append(self.vocab.get_stoi()[other_agent_idx])
        if len(out_vocabs) == 0:
            semantic_target = None
        return out_vocabs, semantic_target

    def get_hash_param_keys(self) -> List[str]:
        return [
            "use_annotated_captions",
            "skip_out_agent_tokens",
            "max_agents",
            "use_annotated_captions",
            "max_language_tokens",
            "source_captions_dir",
        ]

    def _get_params_for_hash(self, params: dict) -> dict:
        p = super()._get_params_for_hash(params)
        # Only keep folder name to make cache transferable.
        p["source_captions_dir"] = p["source_captions_dir"].split("/")[-1]
        p = dict(sorted(p.items()))
        return p

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):
        """Create vectors for the input language tokens.

        Returns
        ----------
        dict
            Update the "language_tokens" field in result_dict.
            The dimension of the vector is (num_agents, num_lang_tokens) that stores token IDs.
            The tokens include <bos> and <eos> to indicate start and end of a sentence.
        """
        selected_agent_idx = list(
            result_dict[ProtobufPredictionDataset.DATASET_KEY_AGENT_IDX]
        )  # index in the original list
        agent_ids = result_dict[ProtobufPredictionDataset.DATASET_KEY_DOT_KEYS]
        agent_raw_tokens = dict()
        agent_tokens = np.ones((params["max_agents"], params["max_language_tokens"])) * -1
        agent_token_cnts = np.zeros(params["max_agents"])
        prediction_timestamp = (
            result_dict[ProtobufPredictionDataset.DATASET_KEY_PREDICTION_TIMESTAMP] * 1e9
        )  # To nanoseconds
        # The overall number of entries in the semantic tensor.
        if result_dict[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_TARGETS]:
            # Each token sequence has the <bos>, <eos>, <pad> tokens to indicate the start, end, and padding.
            for semantic_target in result_dict[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_TARGETS]:
                try:
                    # Get the new index in the agent tensor.
                    agent_idx = np.argwhere(agent_ids == int(semantic_target["agentId"]))[0][0]
                except:
                    # Skip if the DOT key is not available
                    continue
                if params["use_annotated_captions"]:
                    tokens = clean_caption(semantic_target["value"], self.word_map).split()
                else:
                    tokens = semantic_target["value"].split()
                # Only consider future language tokens and cap the number of agents to consider.
                if (
                    semantic_target["typeId"] != TYPE_LANGUAGE_TOKENS
                    or parse_protobuf_timestamp(semantic_target["timestampEnd"]).ToNanoseconds() <= prediction_timestamp
                    or (len(tokens) > 0 and tokens[0] not in self.vocab.get_stoi())
                    or random.uniform(0, 1) > params["language_input_train_ratio"]  # sample data for training
                ):
                    continue
                start_time = max(
                    prediction_timestamp, parse_protobuf_timestamp(semantic_target["timestampStart"]).ToNanoseconds()
                )
                if agent_idx not in agent_raw_tokens:
                    agent_raw_tokens[agent_idx] = defaultdict(list)
                agent_raw_tokens[agent_idx][start_time].append(semantic_target)
        # Random sample a token if they start at the same time.
        for agent_idx, tokens in agent_raw_tokens.items():
            current_end_time = -1
            sorted_tokens = OrderedDict(sorted(tokens.items()))
            for start_time, semantic_targets in sorted_tokens.items():
                if start_time < current_end_time or int(agent_token_cnts[agent_idx]) >= params["max_language_tokens"]:
                    continue
                selected_vocabs, semantic_target = self.select_semantic_target_to_token_ids(
                    semantic_targets, params, agent_ids, selected_agent_idx
                )
                for vocab_id in selected_vocabs:
                    if int(agent_token_cnts[agent_idx]) >= params["max_language_tokens"]:
                        break
                    agent_tokens[agent_idx][int(agent_token_cnts[agent_idx])] = vocab_id
                    agent_token_cnts[agent_idx] += 1
                current_end_time = parse_protobuf_timestamp(semantic_target["timestampEnd"]).ToNanoseconds()
        # Go over all agents to build the output tensor.
        token_vec = np.ones((params["max_agents"], params["max_language_tokens"])) * self.vocab.get_stoi()["<pad>"]
        for agent_idx, agent_token in enumerate(agent_tokens):
            n_tokens = np.count_nonzero(agent_token)
            # Skip an agent if the agent doesn't have any token.
            if n_tokens < 1:
                continue
            has_token = False
            last_idx = 0
            for token_idx, token in enumerate(agent_token):
                if token >= 0:
                    if not params["use_annotated_captions"]:
                        if token_idx + 1 < params["max_language_tokens"] - 1:
                            token_vec[agent_idx][token_idx + 1] = token
                            last_idx = token_idx + 1
                    else:
                        token_vec[agent_idx][token_idx] = token
                        last_idx = token_idx
                    has_token = True
            if has_token and not params["use_annotated_captions"]:
                token_vec[agent_idx][0] = self.vocab.get_stoi()["<bos>"]
                token_vec[agent_idx][last_idx + 1] = self.vocab.get_stoi()["<eos>"]
        ret = {ProtobufPredictionDataset.DATASET_KEY_LANGUAGE_TOKENS: token_vec}
        return ret


class TokenGenerationCost(AdditionalCostCallback):
    def __init__(self):
        """Module for computing the supervised loss for the token generator."""
        super().__init__()

    def update_costs(self, additional_stats, params, predicted, expected, future_encoding, extra_context):
        """Compute the cross-entropy loss of the generated token sequence.

        Parameters
        ----------
        additional_stats : dict
            A dictionary output variable, add 'token_cost' value with the added cost.
        params : dict
            A parameters dictionary.
        predicted : dict
            A dictionary with 'tokens', a tensor value of size B x A x max_token_length x vocab_size.
        expected : dict
            A dictionary with expected 'tokens', a tensor value of size B x A x max_token_length x vocab_size.
        future_encoding : dict
            A dictionary with additional encoding information for the future prediction.
        extra_context : dict
            A dictionary that provides extra context for computing costs.
        """
        vocab = params["language_vocab"]
        predicted_tokens = predicted["language_tokens"]
        n_samples, n_agents, n_tokens, batch_size, vocab_size = predicted_tokens.shape
        # Compute token cost if there is ground-truth language tokens or not in ablation
        if "language_tokens" in expected and not params["latent_factors_detach"]:
            total_size = n_samples * n_agents * n_tokens * batch_size
            # Permute to be (n_tokens, batch_size, n_agents) and repeat for n_samples
            gt_tokens = expected["language_tokens"].long().permute(2, 0, 1).repeat(n_samples, 1, 1, 1)
            gt_tokens = gt_tokens.view(total_size)
            predicted_tokens = predicted_tokens.view(total_size, vocab_size)
            cross_entropy = nn.CrossEntropyLoss(ignore_index=vocab.get_stoi()["<pad>"])
            additional_stats["token_cost"] = cross_entropy(predicted_tokens, gt_tokens)
        else:
            additional_stats["token_cost"] = torch.tensor(0.0)


class LanguageTrainerCallback(AdditionalTrainerCallback):
    def __init__(self, params):
        """Module for handling additional language callbacks in the trainer script."""
        super().__init__(params)

    def epoch_update(self, epoch: int, params: dict):
        """Unfreeze the pretrained word embeddings if needed."""
        if epoch == params["unfreeze_token_embedding_epoch"]:
            for param in params["latent_factors"].token_embedding.embed.parameters():
                param.requires_grad = True

    def update_statistic_keys(self, statistic_keys: list) -> None:
        """
        Callback function to add additional keys to the last of statistic_keys.

        Parameters
        ----------
        statistic_keys : list
            Statistic keys.
        """
        statistic_keys.append("token_cost")

    def update_decoding(self, data: dict, stats_list: list):
        """
        Callback function to add additional decoding to decoder inputs.

        Parameters
        ----------
        data : dict
            Dictionary of data.
        stats_list : list
            List of stats dictionaries.
        """
        # Stack preictions from all samples
        data["language_tokens"] = torch.stack([stats["language_tokens"] for stats in stats_list])
        # Only take the first sample for visualization
        stats = stats_list[0]
        data["decoded_tokens"] = stats["decoded_tokens"]
        data["token_attention_weights"] = stats["attention_weights"]
        data["token_agent_attention_weights"] = stats["agent_attention_weights"]

    def update_expected_results(self, expected: dict, batch_itm: dict, num_future_timepoints: int):
        """
        Callback function to add additional data to expected states.

        Parameters
        ----------
        expected : dict
            Dictionary of expected values.
        batch_itm : dict
            Dictionary of batch item.
        num_future_timepoints : int
            Number of future time steps.
        """
        if ProtobufPredictionDataset.DATASET_KEY_LANGUAGE_TOKENS in batch_itm:
            expected["language_tokens"] = batch_itm[ProtobufPredictionDataset.DATASET_KEY_LANGUAGE_TOKENS]

    def update_solution(self, solution: dict, predicted: dict, batch_idx: int):
        """
        Callback function to add predicted language tokens to solution.

        Parameters
        ----------
        solution : dict
            Dictionary of solution.
        predicted : dict
            Dictionary of prediction.
        batch_idx : int
            Batch index.
        """
        solution["token_attention_weights"] = []
        solution["token_agent_attention_weights"] = []
        solution["predicted_tokens"] = []
        for agent_i in range(len(predicted["decoded_tokens"])):
            solution["predicted_tokens"].append(predicted["decoded_tokens"][agent_i][batch_idx])
            if len(predicted["token_attention_weights"].keys()) == 0:
                continue
            attn_maps = predicted["token_attention_weights"][agent_i]
            solution["token_attention_weights"].append([att[batch_idx] for att in attn_maps])
            solution["token_agent_attention_weights"].append({})
            for token_idx in predicted["token_agent_attention_weights"].keys():
                solution["token_agent_attention_weights"][agent_i][token_idx] = []
                if agent_i in predicted["token_agent_attention_weights"][token_idx]:
                    for att_t in predicted["token_agent_attention_weights"][token_idx][agent_i]:
                        solution["token_agent_attention_weights"][agent_i][token_idx].append(att_t[batch_idx])

    def visualize_agent_additional_info(
        self,
        agent_id: int,
        is_future_valid: torch.Tensor,
        sample_i: int,
        solution: dict,
        ax: matplotlib.axes.Axes,
        predicted_x: Union[torch.Tensor, np.ndarray],
        predicted_y: Union[torch.Tensor, np.ndarray],
        agent_color_predicted: MatplotlibColor,
    ):
        if sample_i == 0:
            ax.text(
                predicted_x[0], predicted_y[0], "Agent {}".format(agent_id), color=agent_color_predicted, fontsize=3
            )

    def update_visualization_text(
        self, cost_str: str, solution: dict, batch: dict, batch_idx: int, num_past_points: int
    ) -> str:
        """
        Update visualization text.
        Parameters
        ----------
        cost_str : str
            Cost string.
        solution : dict
            Solution dictionary.
        batch : dict
            Data batch.
        batch_idx : int
            Batch index.
        num_past_points : int
            Number of past points.
        """
        text = "cost " + cost_str
        if "predicted_tokens" in solution and ProtobufPredictionDataset.DATASET_KEY_LANGUAGE_TOKENS in batch:
            vocab = self.params["language_vocab"]
            # Only show the groud truth and prediction of the first agent, may include other agents later.
            gt_tokens = (
                batch[ProtobufPredictionDataset.DATASET_KEY_LANGUAGE_TOKENS][batch_idx][0].detach().cpu().numpy()
            )
            gt_tokens_str = []
            for token_idx in gt_tokens:
                token = vocab.lookup_token(int(token_idx))
                if token in SPECIALS:
                    continue
                gt_tokens_str.append(token)
            text += "\n\ntoken seqs ="
            text += "\n\n GT: " + wrap_text(str(",".join(gt_tokens_str)), length_line=25)
            text += "\n\n predicted: " + wrap_text(str(",".join(solution["predicted_tokens"][0])), length_line=25)
        return text

    def visualize_additional_info(self, solution: dict, fig: mpl.figure.Figure):
        """
        Visualize additional info, e.g. attention

        Parameters
        ----------
        solution : dict
            Solution dictionary.
        fig : matplotlib.figure.Figure
            Figure instance.
        """
        if len(solution["token_attention_weights"]) == 0:
            return

        def viz_attention(title, attn, tokens, y=0.7, dx=0.06):
            x = 0.01
            for i, token in enumerate(tokens):
                fig.text(x, y, title, fontsize=4)
                text = fig.text(x + i * dx, y - 0.01, token, fontsize=4)
                text.set_bbox(dict(facecolor="red", alpha=attn[i].detach().cpu().item(), edgecolor="none"))

        # Currently only plot the first agent, may include other agents later.
        # Plot attention weight at different time steps: begin, mid, end
        attention_weights = solution["token_attention_weights"][0]
        predicted_tokens = solution["predicted_tokens"][0]
        num_time_steps = len(attention_weights)
        mid = int(num_time_steps / 2)
        viz_attention("t=0", attention_weights[0], predicted_tokens, y=0.35)
        viz_attention("t=" + str(mid), attention_weights[mid], predicted_tokens, y=0.3)
        viz_attention("t=" + str(num_time_steps - 1), attention_weights[-1], predicted_tokens, y=0.25)


class LanguageFactorsTrainerCallback(LatentFactorsTrainerCallback):
    """Trainer callback to balance examples with language tokens."""

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.vocab = params["language_vocab"]
        self.trainer_params = params
        self.rebalance_labels = None

    def trainer_init(self, datasets: dict, logger: TrainingLogger):
        cache_id = split_reading_hash(params=self.trainer_params, postfix="language_factor_balancing")
        if self.trainer_params["cache_latent_factors"]:
            cache_element = CacheElement(
                folder=self.trainer_params["cache_dir"],
                id=cache_id,
                ext="pkl",
                should_lock=self.trainer_params["use_cache_lock"],
                read_only=self.trainer_params["cache_read_only"],
                disable_cache=self.trainer_params["disable_cache"],
            )
            print("cache_id: {}".format(cache_id))
            cached_data = cache_element.load()
            if cached_data is not None:
                rebalance_labels = cached_data
                print("Loading class indices from cache")
            else:
                rebalance_labels = self.get_class_labels(datasets)
                cache_element.save((rebalance_labels))
        else:
            rebalance_labels = self.get_class_labels(datasets)
        self.rebalance_labels = rebalance_labels

    def get_class_labels(self, datasets: dict):
        """Get the class labels of the examples that have tokens and do not have tokens

        Parameters
        ----------
        datasets: dict
            Object containing the dataset to count or rebalance.

        Returns
        -------
        rebalance_class_indices: dict
            The dictionary that contains class to indices mappings.
        """
        num_rebalance_workers = self.trainer_params["num_rebalance_workers"]
        rebalance_labels = {}
        for key in datasets:
            dataset = copy.copy(datasets[key])
            # Remove unused data transforms.
            old_dts = []
            for ds_i, ds in enumerate(dataset.datasets):
                old_dts.append(ds.data_transforms)
                new_dt = []
                for dtransform in ds.data_transforms:
                    if (
                        str(type(dtransform))
                        == "<class 'triceps.protobuf.prediction_dataset_auxiliary.GlobalImageHandler'>"
                        or str(type(dtransform))
                        == "<class 'triceps.protobuf.prediction_dataset_auxiliary.AgentImageHandler'>"
                        or str(type(dtransform))
                        == "<class 'triceps.protobuf.prediction_dataset_map_handlers.PointMapHandler'>"
                    ):
                        continue
                    new_dt.append(dtransform)
                ds.data_transforms = new_dt
            has_tokens = []
            # Go over the data to get class labels. Uses a bigger batch size than parameters, as this is running without
            # any heavy transforms.
            for item in tqdm.tqdm(
                DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_rebalance_workers),
                desc="Get token class indices",
            ):
                tokens = (
                    item[ProtobufPredictionDataset.DATASET_KEY_LANGUAGE_TOKENS] == self.vocab.get_stoi()["<pad>"]
                ).int()
                num_all_pads = (tokens.sum(2) >= self.trainer_params["max_language_tokens"]).int()
                has_tokens.append(num_all_pads.sum(1) != self.trainer_params["max_agents"])
            has_tokens = torch.cat(has_tokens).int().cpu().detach().numpy()
            for ds_i, ds in enumerate(dataset.datasets):
                ds.data_transforms = old_dts[ds_i]
            rebalance_labels[key] = has_tokens
        return rebalance_labels

    def get_balanced_indices(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_key: str,
        epoch_size: int,
        batch_size: int,
        balance_classes: Optional[dict],
    ):
        indices_range = np.array(range(len(dataset)))
        if self.rebalance_labels is not None:
            rebalance_class_indices = {}
            class_ids = np.unique(self.rebalance_labels[dataset_key])
            for cls_id in class_ids:
                class_idxs = indices_range[self.rebalance_labels[dataset_key] == cls_id]
                rebalance_class_indices[cls_id] = class_idxs
        else:
            rebalance_class_indices = {"all": indices_range}

        # Balance examples that have language tokens and do not have language tokens.
        corrected_epoch_size = max(epoch_size, batch_size * 2)
        if not self.trainer_params["full_dataset_epochs"]:
            balanced_indices = []
            for cls_id in rebalance_class_indices:
                class_idxs = rebalance_class_indices[cls_id]
                if len(class_idxs) == 0:
                    continue
                balanced_indices.append(
                    np.random.choice(
                        a=class_idxs,
                        size=np.int64(np.ceil(corrected_epoch_size / len(rebalance_class_indices))),
                        replace=True,
                    )
                )
            balanced_indices = np.concatenate(balanced_indices)
        else:
            balanced_indices = list(range(len(dataset)))

        return balanced_indices
