import copy
import json
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorboardX
import torch
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from intent.multiagents.cache_utils import split_reading_hash
from intent.multiagents.trainer_logging import TrainingLogger
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset

# TODO(guy.rosman): move the cache elements to a utils package.
from triceps.protobuf.prediction_dataset_cache import CacheElement
from triceps.protobuf.prediction_dataset_semantic_handler import (
    SEMANTIC_HANDLER_FINISH_IDX,
    SEMANTIC_HANDLER_START_IDX,
    SEMANTIC_HANDLER_TYPE_IDX,
    SEMANTIC_HANDLER_VALID_IDX,
    SEMANTIC_HANDLER_VALUE_IDX,
)


def load_latent_factor_info(latent_factors_file: str, replace_ids: bool = False) -> list:
    """Loads the json info of the latent factors, including semantics.

    Parameters
    ----------
    latent_factors_file
        The file location
    replace_ids
        Whether or not to replace ids after loading.

    returns:
        label definitions from latent_factors_file
    """
    with open(latent_factors_file) as fp:
        label_definitions = json.load(fp)
    if replace_ids:
        for i, itm in enumerate(label_definitions):
            itm["id"] = i
    return label_definitions


class LatentFactorsTrainerCallback(ABC):
    """
    Interface for latent factors related operations in the trainer.
    """

    @abstractmethod
    def __init__(self, params: dict):
        """
        Initialize the callback.

        Parameters
        ----------
        params: dict
            trainer parameters.
        """
        super().__init__()

    def trainer_init(self, datasets: dict, logger: TrainingLogger):
        """Run at the init phase of a trainer.

        Parameters
        ----------
        datasets: dict
            a dictionary from dataset_key to Pytorch dataset
        logger: TrainingLogger
            The logger to save training statistics and info.
        """

    def get_semantic_label_weights(
        self, semantic_labels: torch.Tensor, dataloader_type: str, device: torch.device
    ) -> torch.Tensor:
        """Compute the weights of the given semantic labels

        Parameters
        ----------
        semantic_labels: torch.Tensor
            The semantic labels to be considered.
        dataloader_type: str
            Type of the data loader.
        device: torch.device
            Device for the tensors.

        Returns
        -------
        torch.Tensor
            The weights for semantic labels.
        """

    @abstractmethod
    def get_balanced_indices(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_key: str,
        epoch_size: int,
        batch_size: int,
        balance_classes: list,
    ) -> list:
        """Get a balanced set of indices from the dataset, to sample from during training.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            pytorch dataset.
        dataset_key : string
            basestring
        epoch_size: int
            size of desired epoch
        batch_size: intself.deviceself.deviceself.deviceself.device
            size of the batch, for verifying the epoch size
        balance_classes: list
            list of relevant classes.

        Returns
        -------
        balanced_indices: list
            List of indices for relevant classes
        """


class LatentFactorsCallback(LatentFactorsTrainerCallback):
    """Encapsulate functionality related to latent factors and semantic labels."""

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.label_pos_weights = None
        self.rebalance_labels = None
        self.trainer_param = params
        self.latent_factors_info = load_latent_factor_info(self.trainer_param["latent_factors_file"])

    def get_semantic_pos_weights(self, datasets: dict, trainer_param: dict, logger: TrainingLogger):
        """
        Get positive weights which can be used for weighted BCE loss
        :return: positive weights, size [N_semantics]
        """
        if len(datasets["train"]) and ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_LABELS in datasets["train"][0]:
            print("computing semantic label weights")
        else:
            return None

        cache_id = split_reading_hash(params=trainer_param, postfix="latent_factor_balancing")
        if trainer_param["cache_latent_factors"]:
            cache_element = CacheElement(
                folder=trainer_param["cache_dir"],
                id=cache_id,
                ext="pkl",
                read_only=trainer_param["cache_read_only"],
                disable_cache=trainer_param["disable_cache"],
                should_lock=trainer_param["use_cache_lock"],
            )
            print("cache_id: {}".format(cache_id))
            cached_data = cache_element.load()
            if cached_data is not None:
                neg_num, pos_num, rebalance_labels = cached_data
                print("loading balancing counts")
            else:
                neg_num, pos_num, rebalance_labels = self.calculate_nums_labels(trainer_param, datasets)
                cache_element.save((neg_num, pos_num, rebalance_labels))
        else:
            neg_num, pos_num, rebalance_labels = self.calculate_nums_labels(trainer_param, datasets)

        # pos_weight for BCEWithLogitsLoss
        label_pos_weights = neg_num / (pos_num + 1e-3)
        if self.trainer_param["semantic_labels_balance_cap"] is not None:
            semantic_cap = trainer_param["semantic_labels_balance_cap"]
            label_pos_weights = label_pos_weights.clamp(1.0 / semantic_cap, semantic_cap)
        # TODO(rui.yu): add label name
        text_weight = ""
        for i in range(len(label_pos_weights)):
            text_weight += "Label {}, {}: positive_weight = {}; [{}] positives; [{}] negatives.  \n".format(
                i, self.latent_factors_info[i]["name"], label_pos_weights[i], pos_num[i], neg_num[i]
            )
        if logger:
            logger.add_text("semantic label weights", text_weight)
        return label_pos_weights, rebalance_labels

    def trainer_init(self, datasets: dict, logger: TrainingLogger):
        self.label_pos_weights, self.rebalance_labels = self.get_semantic_pos_weights(
            datasets=datasets, trainer_param=self.trainer_param, logger=logger
        )

    @staticmethod
    def calculate_nums_labels(trainer_param, datasets) -> Tuple:
        """Calculate the number of each label and allow for rebalancing the dataset.

        Parameters
        ----------
        trainer_param
            All params for this run.
        datasets
            Object containing the dataset to count or rebalance.
        """
        pos_num = 0
        neg_num = 0

        num_rebalance_workers = trainer_param["num_rebalance_workers"]
        rebalance_labels = {}
        # TODO(guy.rosman): replace with dataloader to make it faster.
        for key in datasets:
            dataset = copy.copy(datasets[key])
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
            is_labeled = []
            # Go over the data to get class labels. Uses a bigger batch size than parameters, as this is running without
            # any heavy transforms.
            for itm in tqdm.tqdm(
                DataLoader(dataset, num_workers=num_rebalance_workers, batch_size=64),
                desc="get_semantic_pos_weights",
            ):
                labels_i = itm[ProtobufPredictionDataset.DATASET_KEY_SEMANTIC_LABELS]
                pos_num += (
                    labels_i[:, :, SEMANTIC_HANDLER_VALID_IDX] * (labels_i[:, :, SEMANTIC_HANDLER_VALUE_IDX] > 0)
                ).sum(0)
                neg_num += (
                    labels_i[:, :, SEMANTIC_HANDLER_VALID_IDX] * (labels_i[:, :, SEMANTIC_HANDLER_VALUE_IDX] < 0)
                ).sum(0)
                is_labeled.append(labels_i[:, :, SEMANTIC_HANDLER_VALID_IDX].sum(1) > 0)
                assert all(neg_num >= 0)
            is_labeled = torch.cat(is_labeled).int().cpu().detach().numpy()
            for ds_i, ds in enumerate(dataset.datasets):
                ds.data_transforms = old_dts[ds_i]
            rebalance_labels[key] = is_labeled
        return neg_num, pos_num, rebalance_labels

    def get_semantic_label_weights(self, semantic_labels, dataloader_type, device):
        semantic_label_weights = self.label_pos_weights.to(device)
        if (semantic_labels is not None) and (semantic_labels**2).sum() > 0:
            num_semantic_entries = semantic_labels.shape[1]
            if (dataloader_type != "train") or (
                dataloader_type == "train" and self.trainer_param["disable_label_weights"]
            ):
                semantic_label_weights = torch.ones(num_semantic_entries).to(device)

        return semantic_label_weights

    def get_balanced_indices(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_key: str,
        epoch_size: int,
        batch_size: int,
        balance_classes: dict = None,
    ):
        if balance_classes is None:
            balance_classes = self.rebalance_labels

        indices_range = np.array(range(len(dataset)))
        if balance_classes is not None:
            rebalance_class_indices = {}
            class_ids = np.unique(balance_classes[dataset_key])
            for cls_id in class_ids:
                class_idxs = indices_range[balance_classes[dataset_key] == cls_id]
                # TODO(guy.rosman): place better conditions here on what makes sense to rebalance or is "too unbalanced"
                assert len(class_idxs) > 0
                rebalance_class_indices[cls_id] = class_idxs
        else:
            rebalance_class_indices = {"all": indices_range}

        # Balance classes - create the indices with each class,
        # concatenate to create an index set to sample from.
        corrected_epoch_size = max(epoch_size, batch_size * 2)
        if not self.trainer_param["full_dataset_epochs"]:
            balanced_indices = []
            for cls_id in rebalance_class_indices:
                class_idxs = rebalance_class_indices[cls_id]
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


def compute_semantic_costs(
    additional_stats,
    params,
    timestamps,
    prediction_timestamp,
    future_encoding,
    semantic_emissions,
    semantic_labels,
    semantic_index,
    label_weights,
):
    """
    Update additional stats to add "semantic_cost" and "semantic_costs/<key>"
    :param additional_stats: Dictionary of statistics, costs, etc.
    :param params: The parameters dictionary.
    :param timestamps:
    :param prediction_timestamp:
    :param future_encoding:
    :param semantic_labels:
    :param semantic_index:
    :return:
    """
    # Semantic costs are computed for each batch-entry individually (shape batch_size)
    semantic_costs = torch.zeros_like(timestamps[:, 0])
    if semantic_labels is not None:
        additional_stats["predicted_semantics"] = torch.zeros_like(semantic_labels)
    if (semantic_labels is not None) and (semantic_labels**2).sum() > 0:
        num_semantic_entries = semantic_labels.shape[1]
        num_timestamps = future_encoding.shape[1]
        num_valid_entries = semantic_labels[:, :, SEMANTIC_HANDLER_VALID_IDX].sum(dim=1)
        for se_i in range(num_semantic_entries):
            # if the label tensor of a sample includes all zeros, it is not annotated (augmented).
            is_annotated = semantic_labels.sum(dim=2).sum(dim=1).bool()  # batch_size
            # the id of the semantic label in semantic entry se_i
            ids = semantic_labels[:, se_i, SEMANTIC_HANDLER_TYPE_IDX]  # batch_size
            start_times = semantic_labels[:, se_i, SEMANTIC_HANDLER_START_IDX]  # batch_size
            end_times = semantic_labels[:, se_i, SEMANTIC_HANDLER_FINISH_IDX]
            # TODO(guy.rosman): fix times and prediction_timestamp.
            full_timestamps = prediction_timestamp.unsqueeze(1) + timestamps[:, -num_timestamps:]
            value = semantic_labels[:, se_i, SEMANTIC_HANDLER_VALUE_IDX]
            validity = semantic_labels[:, se_i, SEMANTIC_HANDLER_VALID_IDX]
            for s_id in semantic_index:
                # Assume the semantic IDs of the valid instances are consistent in a batch
                if s_id != ids[is_annotated][0]:
                    continue
                s_res = semantic_emissions[semantic_index[s_id]](future_encoding[:, :, :, 0]).squeeze(2)
                # This is old t_weight, being replaced by the following one (uniform in an interval)
                # t_weight = ((full_timestamps-times.unsqueeze(1)).abs()+5e-2)**-2
                t_weight = (full_timestamps >= start_times.unsqueeze(1)) * (full_timestamps <= end_times.unsqueeze(1))
                normalized_t_weight = t_weight.float() / (t_weight.float().sum(dim=1, keepdim=True) + 1e-4)
                additional_stats["predicted_semantics"][:, se_i, SEMANTIC_HANDLER_VALUE_IDX] = torch.sigmoid(
                    (s_res * normalized_t_weight).sum(dim=1)
                )
                additional_stats["predicted_semantics"][:, se_i, SEMANTIC_HANDLER_VALID_IDX] = validity
                key_name = str(s_id) + "_" + semantic_index[s_id]
                if validity.sum() == 0:
                    continue
                # For BCE loss, clamp target values from [-1, 1] to [0, 1]
                if label_weights is not None:
                    err_cost = F.binary_cross_entropy_with_logits(
                        s_res,
                        value.clamp(min=0).unsqueeze(1).repeat(1, num_timestamps),
                        reduction="none",
                        pos_weight=label_weights[se_i],
                    )
                    err_cost /= 1 + label_weights[se_i]
                else:
                    err_cost = F.binary_cross_entropy_with_logits(
                        s_res, value.clamp(min=0).unsqueeze(1).repeat(1, num_timestamps), reduction="none"
                    )

                weighted_err_cost = (err_cost * normalized_t_weight).sum(dim=1)
                semantic_costs += weighted_err_cost * validity
                additional_stats["semantic_costs/" + key_name] = weighted_err_cost[validity > 1e-5]
        # Compute the average cost of all semantics, but not including the instances with no valid semantics
        instance_validity = (num_valid_entries > 0).float()
        semantic_costs = semantic_costs / (num_valid_entries + 1e-4)
        semantic_costs *= instance_validity
    else:
        for s_id in semantic_index:
            key_name = str(s_id) + "_" + semantic_index[s_id]
    additional_stats["semantic_cost"] = semantic_costs * params["semantic_term_coeff"]
