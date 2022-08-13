import copy
import glob
import hashlib
import json
import os
import pickle
import random
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Type

import numpy as np
import torch
import tqdm
import tqdm.asyncio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from intent.multiagents.additional_costs import TrajectoryRegularizationCost
from intent.multiagents.cache_utils import (
    DATA_AND_HANDLER_VERSION,
    PARAM_HASH_LIST,
    SPLIT_HASH_DIRS,
    compute_hash,
    compute_param_hash,
    download_caches_from_s3,
    get_dir_names,
    get_hashing_params,
    get_input_dirs_for_dataset,
)
from intent.multiagents.logging_handlers import ImageStatsLogHandler, LogWorstCases, WaymoLogHandler
from intent.multiagents.prediction_trainer import PredictionProtobufTrainer
from model_zoo.intent.batch_graph_encoder import BatchGraphEncoder
from model_zoo.intent.create_networks import CNNModel
from model_zoo.intent.graph_encoder import GraphEncoder

try:
    from model_zoo.intent.linear_prediction_model import LinearPredictionModel
except ModuleNotFoundError:
    LinearPredictionModel = None
from model_zoo.intent.prediction_model_interface import PredictionModelInterface
from radutils import profiler

# Handle files not found in CI tests.
try:
    from model_zoo.intent.lstm_encoder_net import LSTMEncoder
    from model_zoo.intent.mlp_decoder import MLPDecoder
    from model_zoo.intent.prediction_util_classes import MLPEncoder
    from model_zoo.intent.transformer import TransformerDecoder, TransformerEncoder
except ImportError:
    MLPDecoder = None
    TransformerDecoder = None
    TransformerEncoder = None
    LSTMEncoder = None
    MLPEncoder = None

import radutils.misc as rad_misc
from model_zoo.intent.multiagent_decoder import AugmentedMultiAgentDecoder
from model_zoo.intent.multiagent_decoder_batch import AugmentedMultiAgentDecoderAccelerated

try:
    from model_zoo.intent.poly_decoder import PolynomialMultiAgentDecoder
except ModuleNotFoundError:
    PolynomialMultiAgentDecoder = None
try:
    from model_zoo.intent.poly_encoder import PolynomialEncoder
except ModuleNotFoundError:
    PolynomialEncoder = None
from model_zoo.intent.prediction_model import PredictionModel
from model_zoo.intent.prediction_model_codec import PredictionModelCodec
from model_zoo.intent.prediction_util_classes import (
    ImageEncoder,
    MapAttentionEncoder,
    MapPointGNNEncoder,
    MapPointMLPEncoder,
    StubEncoder,
)
from triceps.protobuf.prediction_dataset import (
    DatasetValidation_Dataset,
    PredictionDataset,
    ProtobufPredictionDataset,
    dataset_collate_fn,
)
from triceps.protobuf.prediction_dataset_auxiliary import (
    AgentImageHandler,
    GlobalImageHandler,
    ImageProcessor,
    WaymoStateHandler,
    interpolate_trajectory,
    select_agent,
)
from triceps.protobuf.prediction_dataset_cache import CacheElement
from triceps.protobuf.prediction_dataset_map_handlers import (
    PointMapHandler,
    RasterMapHandler,
    add_polynomial_features,
    normalize_agent_additional_inputs,
)
from triceps.protobuf.prediction_dataset_semantic_handler import SemanticLabelsHandler
from triceps.protobuf.proto_arguments import normalize_path
from triceps.protobuf.protobuf_training_parameter_names import PARAM_FUTURE_TIMESTEPS, PARAM_PAST_TIMESTEPS

# Map input dim includes (x, y, point type, sin(theta), cos(theta), cos(theta), -sin(theta)).
MAP_INPUT_SIZE = 7


class DataSplitPolicy:
    """Creates a split ensuring that datapoints in train / validation do not overlap.

    Parameters
    ----------
    labels : dictionary
        A dictionary from label name to the label fraction. Fractions should sum up to 1.
    params : dictionary
        The training parameters dict. We use the dataset_collision_inteval to set the resolution for different
        train/test segments.
    """

    def __init__(self, labels: dict = None, params: dict = None) -> None:
        if labels is None:
            labels = {"train": 0.7, "validation": 0.3}
        self.split_map = {}
        self.trainer_params = params
        self.labels = OrderedDict(labels)
        assert np.sum(list(self.labels.values())) == 1.0
        self.cumulative_prob = np.cumsum(list(self.labels.values()))
        self.cache_hits = 0
        self.cache_misses = 0
        self.label_assignments = {k: 0 for k in self.labels.keys()}

    def get_dataset_bin_hash(self, dataset, item=None):
        """Put similar data (from same tlog, same time period) in the same bin. Then compute it's hash as id.
        Avoid put different similar data into both training and validation dataset.

        Parameters
        ----------
        dataset: ProtobufPredictionDataset
            A protobuf dataset. Assuming either 1 prediction instance per dataset, or that the prediction
            instances are similar in tlog/timestamp.

        Returns
        -------
        hash: str
            The bin hash of the dataset
        """
        if item is None:
            item = dataset[0]
        assert "instance_info" in item and item["instance_info"] is not None
        instance_info = json.loads(item["instance_info"])
        _, tlog_hash = os.path.split(instance_info["json_dir"])
        source_tlog = instance_info["source_tlog"]
        # The time bin
        time_bin = int(instance_info["timestamp"] // self.trainer_params["dataset_collision_inteval"])
        # Put examples from same tlog, same time period in the same name
        bin_signature = str(tlog_hash) + "_" + source_tlog + "_" + str(time_bin)
        return hashlib.md5(str(bin_signature).encode()).hexdigest()

    def get_assignment(self, dataset_hash: str) -> str:
        """Get the dataset assignment for a given dataset (e.g. does it belong in training/validation?).

        Parameters
        ----------
        dataset_hash: str
            The hash of the dataset

        Returns
        -------
        label: str
            The label for that dataset (training, validation, depends on the labels used).

        """
        if dataset_hash in self.split_map:
            self.cache_hits += 1
            return self.split_map[dataset_hash]
        else:
            self.cache_misses += 1
            draw = np.random.uniform()
            idx = int(np.sum(self.cumulative_prob < draw))
            label = list(self.labels.keys())[idx]
            self.label_assignments[label] += 1
            self.split_map[dataset_hash] = label
            return label


def get_param_hash(params: Dict, data_handlers):
    """Compute the hash for given parameters."""
    cache_modularity = "whole" if params["cache_dataset_item"] else "individual"
    param_list = list(PARAM_HASH_LIST)
    if params["cache_dataset_item"]:
        assert data_handlers is not None, "When params['cache_dataset_item'] is used, must pass in data_handlers"
        for handler in data_handlers:
            if handler is None:
                continue
            param_list += handler.get_hash_param_keys()
        param_list = sorted(list(set(param_list)))
    return f"data_ver_{DATA_AND_HANDLER_VERSION}-{cache_modularity}-param_{compute_param_hash(params, param_list)}"


def get_split_hash(params):
    training_set_ratio = params["training_set_ratio"]
    # Cache split at the base dir (assuming already append [param_hash] to params["cache_dir"])
    dir_base_names = get_dir_names(params)
    # Define split hash based on base directory names and split percentage.
    split_hash = (
        f"{params['main_param_hash']}-dataset-{compute_hash(str(dir_base_names))}-{int(training_set_ratio * 100)}-split"
    )
    return split_hash


def write_split_info(params, split_hash):
    info = {dir_: None if params[dir_] is None else sorted(params[dir_]) for dir_ in SPLIT_HASH_DIRS}
    info["main_param_hash"] = params["main_param_hash"]
    info["handlers_param_hash"] = params["handlers_param_hash"]
    info["NOTE"] = (
        "Only the base name of the dirs affect the hash of the split. But when loading the "
        "pb files, the relative filename (relative to --input-dir-prefix, which default to ~/) is used. "
        "So the pb files must present at the relative input_dir, for example, "
        "at ${input_dir_prefix}/pedestrian_intent/reharvested/for_annotation/"
    )
    with open(os.path.join(params["split_cache_dir"], split_hash + ".json"), "w") as f:
        json.dump(info, f, indent=4)


def get_dataloader(dataset, params):
    num_workers = params["num_workers"]
    if num_workers == 0:
        persistent_workers = False
        pre_fetch = 2
    else:
        persistent_workers = True
        pre_fetch = 4
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        prefetch_factor=pre_fetch,
        persistent_workers=persistent_workers,
        collate_fn=dataset_collate_fn,
    )
    return dataloader


def set_datasets_item_post_processors(datasets, processors: List[Callable[[Dict, Dict], Dict]]):
    """This function is used to set the item_post_processors to the dataset.
    Must be used like this:
        def functor(item, params):
            return item

        datasets, example_dataset, _ = load_datasets(params, data_transforms=data_handlers)
        set_datasets_item_post_processors(datasets, [functor])
    """
    for _, concat_dataset in datasets.items():
        for dataset in concat_dataset.datasets:
            dataset.item_post_processors = processors


def load_split(
    params,
    split_data_policy,
    datasets_name_lists,
    data_transforms,
    trajectory_handler,
    prediction_instance_cls=ProtobufPredictionDataset,
):
    num_all_instances = 0
    num_valid_instances = 0

    # Remove all torch modules/handlers/callbacks from the params for prediction dataset, it will break multiprocessing.
    dataset_params = copy.copy(params)
    for key, v in params.items():
        if isinstance(params[key], torch.nn.Module) or "_callback" in key or "_handler" in key:
            dataset_params.pop(key)
    try:
        # Test if they are picklable
        pickle.dumps(dataset_params)
    except pickle.PicklingError as e:
        raise RuntimeError("dataset_params must be pickable. Remove them from dataset_params, like lambdas")

    train_datasets_list = []
    validation_datasets_list = []
    train_datasets_names_list = []
    validation_datasets_names_list = []
    if params["data_debug_mode"]:

        def create_dataset(dataset_name, num_instance):
            data_list = []
            data_name_list = []
            for i in range(num_instance):
                d = prediction_instance_cls(
                    f"/synthetic/{dataset_name}/synthetic_instance_{i}",
                    data_transforms=data_transforms,
                    trajectory_handler=trajectory_handler,
                    agent_selection_handler=select_agent,
                    mock_data_mode=params["data_debug_mode"],
                    additional_input_stub=params["add_input_stub"],
                    input_stub_dim=params["input_stub_dim"],
                    additional_agent_input_stub=params["add_agent_input_stub"],
                    agent_input_stub_dim=params["agent_input_stub_dim"],
                    params=dataset_params,
                    check_valid_instances=False,
                )
                data_list.append(d)
                pb_file_only = os.path.basename(d.filename)

                data_name_list.append(pb_file_only)
            return data_list, data_name_list

        train_datasets_list, train_datasets_names_list = create_dataset("training", params["epoch_size"])
        validation_datasets_list, validation_datasets_names_list = create_dataset(
            "validation", max(1, params["epoch_size"] // 10)
        )
        num_all_instances = num_valid_instances = (
            len(train_datasets_list) + len(validation_datasets_list)
        ) * train_datasets_list[0].num_total_instances
        return (
            train_datasets_list,
            validation_datasets_list,
            train_datasets_names_list,
            validation_datasets_names_list,
            num_all_instances,
            num_valid_instances,
        )

    split_hash = get_split_hash(params)
    split_cache_element = CacheElement(
        folder=params["split_cache_dir"],
        id=split_hash,
        ext="pkl",
        read_only=params["cache_read_only"],
        disable_cache=params["disable_cache"] or not params["cache_splits"],
        should_lock=params["use_cache_lock"],
    )
    split_cache = split_cache_element.load()
    print(f"Looking for split at {split_cache_element.filename}")
    if split_cache:
        print("Read split from cache: {}".format(split_cache_element.filename))
        # Update the params in the cached dataset.
        (
            train_datasets_list,
            validation_datasets_list,
            train_datasets_names_list,
            validation_datasets_names_list,
            num_all_instances,
            num_valid_instances,
        ) = split_cache
    else:  # No cached split
        print(f"\nCache split not found, parsing folders\n{params['input_dir']},{params['input_validation_dir']}")
        # Cached split not available.  Generate new split.
        is_validation_dir = params["input_validation_dir"] and len(params["input_validation_dir"]) > 0

        files_to_parse = []
        for input_dir in params["input_dir"]:
            if not os.path.exists(input_dir):
                raise IOError("Missing input folder: {}".format(input_dir))
            files = glob.glob(os.path.join(input_dir, "**/*.pb"), recursive=True)
            files = sorted(files)
            if len(files) == 0:
                warnings.warn("Empty input folder: {}".format(input_dir))

            for filename in files:
                if is_validation_dir:
                    # Assumes we're with homogeneous datasets, need to check whether we're in training/validation and fill it out.
                    if (
                        input_dir in params["input_validation_dir"]
                        and len(validation_datasets_list) >= params["max_files_count"]
                    ):
                        break
                    if (
                        input_dir in params["input_training_dir"]
                        and len(train_datasets_list) >= params["max_files_count"]
                    ):
                        break
                else:
                    # Assumes we're with heterogeneous datasets, need to fill out both datasets.
                    if (
                        len(train_datasets_list) >= params["max_files_count"]
                        and len(validation_datasets_list) >= params["max_files_count"]
                    ):
                        break
                    # When data split is not provided.
                    # This may cause an empty training/validation set, if so, increase the max_files_count.
                    if len(files_to_parse) >= params["max_files_count"]:
                        break
                files_to_parse.append(filename)

        # Validate datasets in parallel
        files_to_parse = sorted(files_to_parse)
        prediction_datasets = [
            # Do not validate instance here, do it later in parallel.
            prediction_instance_cls(
                filename,
                data_transforms=data_transforms,
                trajectory_handler=trajectory_handler,
                agent_selection_handler=select_agent,
                mock_data_mode=params["data_debug_mode"],
                additional_input_stub=params["add_input_stub"],
                input_stub_dim=params["input_stub_dim"],
                additional_agent_input_stub=params["add_agent_input_stub"],
                agent_input_stub_dim=params["agent_input_stub_dim"],
                params=dataset_params,
                check_valid_instances=False,
            )
            for filename in files_to_parse
        ]

        to_validate_data_items = [DatasetValidation_Dataset(d) for d in prediction_datasets]
        to_validate_dataset = torch.utils.data.ConcatDataset(to_validate_data_items)
        dataloader = get_dataloader(to_validate_dataset, params)
        print(f"files to parse: {len(files_to_parse)}")
        dataloader = tqdm.tqdm(dataloader, desc="Validating protobuf files", miniters=1)  # params["disable_tqdm"])
        for checked_datasets in dataloader:
            # protobuf parsed and validated.
            for d_i in range(len(checked_datasets["valid"])):
                dataset = checked_datasets["dataset"][d_i]
                is_valid = checked_datasets["valid"][d_i]
                num_all_instances += dataset.num_total_instances
                num_valid_instances += len(dataset.valid_instances)
                if not is_valid:
                    continue
                item = checked_datasets["result_items"][d_i]

                pb_filename = dataset.filename
                pb_file_only = os.path.basename(dataset.filename)
                if datasets_name_lists is not None:
                    if pb_file_only in datasets_name_lists["train_datasets_names_list"]:
                        train_datasets_list.append(dataset)
                        train_datasets_names_list.append(pb_file_only)
                    elif pb_file_only in datasets_name_lists["validation_datasets_names_list"]:
                        validation_datasets_list.append(dataset)
                        validation_datasets_names_list.append(pb_file_only)
                    # In data debug mode, we do not care about this as we do not use any splits.
                    # Instead, we use a synthetic dataset, thus an error is only raised when we work
                    # on real data.
                    elif not params["data_debug_mode"]:
                        raise ValueError("File {} missing from datasets_name_lists.".format(pb_filename))
                else:
                    # Determine if the scenario is unique enough.
                    try:
                        dataset.params = params
                        dataset_bin_hash = split_data_policy.get_dataset_bin_hash(dataset, item)
                    except ValueError:
                        print(f"ValueError when reading file at {pb_file_only}")
                        continue
                    assignment = split_data_policy.get_assignment(dataset_bin_hash)
                    if assignment == "train":
                        train_datasets_list.append(dataset)
                        train_datasets_names_list.append(pb_file_only)
                    else:
                        validation_datasets_list.append(dataset)
                        validation_datasets_names_list.append(pb_file_only)

        print(
            "protobuf parse {}, after consolidation: {}, train set: {}, validation set: {}".format(
                split_data_policy.cache_hits + split_data_policy.cache_misses,
                split_data_policy.cache_hits,
                len(train_datasets_list),
                len(validation_datasets_list),
            )
        )
        if len(train_datasets_list) == 0:
            print("training set is empty. Increase the protobuf dataset or --max-files-count")
            raise RuntimeError("training set is empty.")
        if len(validation_datasets_list) == 0:
            print("validation set is empty. Increase the protobuf dataset or --max-files-count")
            raise RuntimeError("validation set is empty.")
        # Save cache_splits when it's None
        if params["cache_splits"] and not split_cache_element.read_only:
            split_cache_element.save(
                (
                    train_datasets_list,
                    validation_datasets_list,
                    train_datasets_names_list,
                    validation_datasets_names_list,
                    num_all_instances,
                    num_valid_instances,
                )
            )
            # Write split info to JSON.
            write_split_info(params, split_hash)

    # Re init PredictionDataset params, see
    for t_list in (train_datasets_list, validation_datasets_list):
        for t in t_list:
            t.init_params(
                params,
                data_transforms=data_transforms,
                trajectory_handler=trajectory_handler,
                agent_selection_handler=select_agent,
            )
    return (
        train_datasets_list,
        validation_datasets_list,
        train_datasets_names_list,
        validation_datasets_names_list,
        num_all_instances,
        num_valid_instances,
    )


def get_datasets_name_lists(params):
    # Replace input_dir as a combination of input_training_dir and input_validation_dir if they are provided.
    if params["input_training_dir"] and params["input_validation_dir"]:
        params["input_dir"] = params["input_training_dir"] + params["input_validation_dir"]
    is_training_dir = params["input_training_dir"] and len(params["input_training_dir"]) > 0
    is_validation_dir = params["input_validation_dir"] and len(params["input_validation_dir"]) > 0
    assert is_validation_dir == is_training_dir

    # Load existing split. Check first if --dataset-name-lists was set. If this is not the case
    # see if we are resuming a session and try to use the split from the resumed session.
    datasets_name_lists = None
    last_session_split = f"{params['model_load_folder']}/{params['resume_session_name']}/data_files.json"
    if "datasets_name_lists" in params and params["datasets_name_lists"] is not None:
        datasets_name_lists = params["datasets_name_lists"]

    # Create dataset name lists if separate training and validation dirs are provided.
    elif params["input_training_dir"] and params["input_validation_dir"]:
        datasets_name_lists = {}
        train_datasets_names_list_input = []
        for input_training_dir in params["input_training_dir"]:
            if not os.path.exists(input_training_dir):
                raise IOError("Missing input folder: {}".format(input_training_dir))
            training_files = sorted(glob.glob(os.path.join(input_training_dir, "**/*.pb"), recursive=True))
            train_datasets_names_list_input += [os.path.basename(pb_filename) for pb_filename in training_files]
        datasets_name_lists["train_datasets_names_list"] = train_datasets_names_list_input

        validation_datasets_names_list_input = []
        for input_validation_dir in params["input_validation_dir"]:
            if not os.path.exists(input_validation_dir):
                raise IOError("Missing input folder: {}".format(input_validation_dir))
            validation_files = sorted(glob.glob(os.path.join(input_validation_dir, "**/*.pb"), recursive=True))
            validation_datasets_names_list_input += [os.path.basename(pb_filename) for pb_filename in validation_files]
        datasets_name_lists["validation_datasets_names_list"] = validation_datasets_names_list_input

    elif os.path.exists(last_session_split):
        datasets_name_lists = last_session_split

    if datasets_name_lists is not None:
        if isinstance(datasets_name_lists, str):
            print(f"Loading dataset split from {datasets_name_lists}")
            with open(datasets_name_lists, "r") as fp:
                datasets_name_lists = json.load(fp)

    return datasets_name_lists


def load_datasets(
    params: dict,
    data_transforms: list,
    dataset_cls: Type[torch.utils.data.Dataset] = PredictionDataset,
    prediction_instance_cls=ProtobufPredictionDataset,
):
    """
    :param params: Parameters dictionary, includes input_dir, max_agents,timestep, past_timesteps,
    future_timesteps,cache_dir
    (see argument parsers for documentation)
    :param data_transforms: Data transforms for the datasets, to update the fields.
    :return: datasets (a dictionary with 'train', 'validation' datasets),
    """
    datasets_name_lists = get_datasets_name_lists(params)

    split_data_policy = DataSplitPolicy(
        labels={"train": params["training_set_ratio"], "validation": 1.0 - params["training_set_ratio"]}, params=params
    )

    # Pull the mapping of train/validation from protobuf names
    if params["interp_type"][0].lower() == "none":
        trajectory_handler = None
    else:
        trajectory_handler = interpolate_trajectory
    (
        train_datasets_list,
        validation_datasets_list,
        train_datasets_names_list,
        validation_datasets_names_list,
        num_all_instances,
        num_valid_instances,
        *_,  # Older versions of the cache return an unused variable
    ) = load_split(
        params, split_data_policy, datasets_name_lists, data_transforms, trajectory_handler, prediction_instance_cls
    )

    print("Split dataset policy label assignments", split_data_policy.label_assignments)
    if datasets_name_lists is not None and (
        len(train_datasets_list) < len(datasets_name_lists["train_datasets_names_list"])
        or len(validation_datasets_list) < len(datasets_name_lists["train_datasets_names_list"])
    ):
        print("Warning: Some of the requested dataset files could not be loaded.")

    print(
        "finished reading files, got {} valid instances from total {} instances. Training set: {}, validation set: {}, "
        "(timesteps={},{})".format(
            num_valid_instances,
            num_all_instances,
            len(train_datasets_list),
            len(validation_datasets_list),
            params[PARAM_PAST_TIMESTEPS],
            params[PARAM_FUTURE_TIMESTEPS],
        )
    )

    if params["serialize_trajectories"]:
        # filter validation datasets by scenarios specified with the --serialize-trajectories flag
        validation_datasets_names_list = [
            fn for fn in validation_datasets_names_list if fn in params["serialize_trajectories"]
        ]
        validation_datasets_list = [
            ds
            for ds in validation_datasets_list
            if os.path.basename(ds.rel_filename) in params["serialize_trajectories"]
        ]

    train_dataset = dataset_cls(train_datasets_list, params, "training")
    validation_dataset = dataset_cls(validation_datasets_list, params, "validation")
    datasets = {"train": train_dataset, "validation": validation_dataset}
    datasets_name_lists = {
        "train_datasets_names_list": train_datasets_names_list,
        "validation_datasets_names_list": validation_datasets_names_list,
    }
    return datasets, datasets_name_lists


def create_map_handlers(params: dict):
    # Dataset handler for maps.
    if params["disable_map_input"]:
        map_handler, agent_map_handlers = None, []
    elif params["map_input_type"] == "point":
        map_handler = PointMapHandler(
            params,
            max_point_num=params["map_points_max"],
            sampling_length=params["map_sampling_length"],
            sampling_minimum_length=params["map_sampling_minimum_length"],
        )

        if params["use_global_map"]:
            agent_map_handlers = []
        else:
            agent_map_handlers = [normalize_agent_additional_inputs]
            if params["map_polyline_feature_degree"] > 0:
                agent_map_handlers.append(add_polynomial_features)
    else:
        map_handler = RasterMapHandler(
            params,
            halfwidth=params["map_halfwidth"],
            halfheight=params["map_halfheight"],
            map_scale=params["map_scale"],
        )
        agent_map_handlers = []

    return map_handler, agent_map_handlers


def create_image_handlers(params: dict):
    global_transform = transforms.Compose(
        [transforms.Resize((params["img_height"], params["img_width"])), transforms.ToTensor()]
    )

    # Dataset handler for global images.
    if params["scene_image_mode"] == "none":
        global_image_handler = None
    else:
        time_points = params["scene_image_timepoints"] if params["scene_image_mode"] == "custom" else None
        global_image_handler = GlobalImageHandler(
            params,
            img_dir=params["image_dir"],
            height=params["img_height"],
            width=params["img_width"],
            total_timesteps=params["past_timesteps"] + params["future_timesteps"],
            timepoints=time_points,
            transform=global_transform,
            image_processor=None,
        )

    agent_transform = transforms.Compose(
        [
            transforms.Resize((params["agent_img_height"], params["agent_img_width"])),
            transforms.ToTensor(),
        ]
    )
    # Dataset handler for agent images.
    if params["agent_image_mode"] == "none":
        agent_image_handler = None
    else:
        agents = params["agent_image_agents"] if params["agent_image_mode"] == "custom" else None
        time_points = params["agent_image_timepoints"] if params["agent_image_mode"] == "custom" else None

        if params["agent_image_processor"] == "pa":
            from data_sources.pedestrians.perceptive_automata.pa_wrapper import PerceptiveAutomataWrapper

            image_processor = ImageProcessor(PerceptiveAutomataWrapper(params), 10, params)
        else:
            image_processor = None

        agent_image_handler = AgentImageHandler(
            params,
            img_dir=params["image_dir"],
            height=params["agent_img_height"],
            width=params["agent_img_width"],
            max_agents=params["max_agents"],
            total_timesteps=params["past_timesteps"] + params["future_timesteps"],
            agents=agents,
            timepoints=time_points,
            transform=agent_transform,
            image_processor=image_processor,
        )

    return global_image_handler, agent_image_handler


def create_semantic_handler(params: dict):
    # Dataset handler for semantic labels.
    if not params["use_semantics"]:
        return None
    else:
        return SemanticLabelsHandler(params)


def create_discrete_handler(params: dict):
    # Dataset handler for discrete modes.
    if "discrete_mode_handler" in params and params["discrete_mode_handler"]:
        return params["discrete_mode_handler"]
    else:
        return None


def create_language_handler(params: dict):
    # Dataset handler for language tokens.
    if "language_token_handler" in params and params["language_token_handler"]:
        return params["language_token_handler"]
    else:
        return None


def create_waymo_handler(params: dict):
    # Dataset handler for Waymo extra states.
    if params["use_waymo_dataset"]:
        return WaymoStateHandler(params)
    else:
        return None


def create_handlers(params: dict):
    # Read protobufs, merge into train/validation datasets.
    map_handler, agent_map_handlers = create_map_handlers(params)
    global_image_handler, agent_image_handler = create_image_handlers(params)

    data_handlers = [
        map_handler,
        global_image_handler,
        agent_image_handler,
        create_semantic_handler(params),
        create_discrete_handler(params),
        create_language_handler(params),
        create_waymo_handler(params),
    ]
    params["handlers_param_hash"] = {}
    for handler in data_handlers:
        if hasattr(handler, "handler_param_hash"):
            params["handlers_param_hash"][handler.__class__.__name__] = handler.handler_param_hash

    return data_handlers, agent_map_handlers


def create_input_encoders(params: dict, device: torch.device):
    input_encoders = nn.ModuleDict({})

    if params["add_input_stub"]:
        input_encoders[ProtobufPredictionDataset.DATASET_KEY_STUB] = StubEncoder(
            input_size=params["input_stub_dim"],
            embed_size=params["input_stub_embed_size"],
            requires_grad=params["stub_requires_grads"],
        )

    if params["scene_image_mode"] != "none":
        # Load the parameters for the scene image. Note: the parameters for each image type can differ.
        network_params = {"layer_features": params["scene_image_layer_features"]}
        channels = 3
        if params["use_scene_semantic_masks"]:
            channels += 3

        input_encoders[ProtobufPredictionDataset.DATASET_KEY_IMAGES] = ImageEncoder(
            embed_size=params["image_embed_size"],
            width=params["img_width"],
            height=params["img_height"],
            channels=channels,
            backbone_model=CNNModel(params["scene_image_model"]),
            frozen_params=params["scene_image_frozen_layers"],
            pretrained=params["pretrained_image_encoder"],
            use_checkpoint=False,
            fc_widths=params["scene_image_fc_widths"],
            params=network_params,
            nan_result_retries=params["image_encoder_nan_retries"],
        )

    if params["add_input_map_stub"]:
        input_encoders[ProtobufPredictionDataset.DATASET_KEY_MAP_STUB] = StubEncoder(
            input_size=params["input_map_stub_dim"],
            embed_size=params["input_map_stub_embed_size"],
            requires_grad=params["stub_requires_grads"],
        )

    if not params["disable_map_input"] and params["use_global_map"]:
        input_encoders[ProtobufPredictionDataset.DATASET_KEY_MAP] = MapAttentionEncoder(
            map_input_dim=MAP_INPUT_SIZE,
            traj_input_dim=3,
            embed_size=params["map_layer_features"][0],
            device=device,
            params=params,
        )

    return input_encoders


def create_agent_input_encoders(params: dict, map_handler: callable, device: torch.device):
    agent_input_encoders = nn.ModuleDict({})

    # Load the parameters for the agent image and map rasters (if used).
    if params["add_agent_input_stub"]:
        agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_AGENT_STUB] = StubEncoder(
            input_size=params["agent_input_stub_dim"],
            embed_size=params["agent_input_stub_embed_size"],
            requires_grad=params["stub_requires_grads"],
        )
    if params["add_agent_map_input_stub"]:
        agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_AGENT_MAP_STUB] = StubEncoder(
            input_size=params["agent_map_input_stub_dim"],
            embed_size=params["agent_map_input_stub_embed_dim"],
            requires_grad=params["stub_requires_grads"],
        )
    # Only encode the map in the agent frame if it is not disabled or set to use global frame.
    if not params["disable_map_input"] and not params["use_global_map"]:
        if params["map_input_type"] == "point":
            if params["map_encoder_type"] == "gnn":
                # Map input dim includes (x, y, point type, sin(theta), cos(theta), cos(theta), -sin(theta)).
                agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_MAP] = MapPointGNNEncoder(
                    map_input_dim=MAP_INPUT_SIZE,
                    traj_input_dim=3,
                    embed_size=params["map_layer_features"][0],
                    device=device,
                    params=params,
                )
            elif params["map_encoder_type"] == "attention":
                # Map input dim includes (x, y, point type, sin(theta), cos(theta), cos(theta), -sin(theta)).
                agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_MAP] = MapAttentionEncoder(
                    map_input_dim=MAP_INPUT_SIZE,
                    traj_input_dim=3,
                    embed_size=params["map_layer_features"][0],
                    device=device,
                    params=params,
                )
            else:
                agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_MAP] = MapPointMLPEncoder(
                    embed_size=params["map_layer_features"]
                )
        else:
            network_params = {"layer_features": params["raster_map_layer_features"]}

            agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_MAP] = ImageEncoder(
                embed_size=params["image_embed_size"],
                width=map_handler.pixel_width,
                height=map_handler.pixel_height,
                frozen_params=params["raster_map_frozen_layers"],
                pretrained=params["pretrained_image_encoder"],
                params=network_params,
            )
    if params["agent_image_mode"] != "none":
        network_params = {"layer_features": params["agent_image_layer_features"]}
        channels = 3
        if params["use_agent_semantic_masks"]:
            channels += 3
        if params["use_agent_pose_estimates"]:
            channels += 3

        agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES] = ImageEncoder(
            embed_size=params["image_embed_size"],
            width=params["agent_img_width"],
            height=params["agent_img_height"],
            channels=channels,
            backbone_model=CNNModel(params["agent_image_model"]),
            pretrained=params["pretrained_image_encoder"],
            frozen_params=params["agent_image_frozen_layers"],
            use_checkpoint=False,
            fc_widths=params["agent_image_fc_widths"],
            params=network_params,
        )
    if params["use_waymo_dataset"]:
        assert MLPEncoder is not None, "MLPEncoder needs to be provided."
        # Auxiliary state only supports MLP decoder.
        agent_input_encoders[ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE] = MLPEncoder(
            input_size=6,
            embed_size=params["predictor_hidden_state_dim"],
        )

    return agent_input_encoders


def prepare_cache_and_encoders(params):
    data_handlers, _ = create_handlers(params)
    params = prepare_cache(params, data_handlers)
    data_handlers, agent_map_handlers = create_handlers(params)
    return params, data_handlers, agent_map_handlers


def create_encoders(params: dict, map_handler: callable, device: torch.device):
    input_encoders = create_input_encoders(params, device)
    agent_input_encoders = create_agent_input_encoders(params, map_handler, device)
    return (
        input_encoders,
        agent_input_encoders,
    )


def update_num_epochs_from_list(output_param, param_original, current_param_sets, default_size, invert=False):
    """
    Update the num_epochs field from the original params, assuming we're now building the next parameter set to add to current_param_sets.
    If param_original['pretraining_relative_lengths'] is None, use a default size.
    :param output_param: the parameter set to be updated.
    :param param_original: the original parameter set of the training program.
    :param current_param_sets: the current set of parameter sets.
    :param default_size: the default size.
    :param invert: should we use param_original['pretraining_relative_lengths'] in a reverse order.
    :return: updates output_param's num_epoch field.
    """
    if param_original["pretraining_relative_lengths"] is not None:
        if invert:
            idx = len(current_param_sets) - 1
        else:
            idx = len(param_original["pretraining_relative_lengths"]) - (len(current_param_sets))
        output_param["num_epochs"] = int(
            param_original["pretraining_relative_lengths"][idx] * param_original["pretraining_timescale"]
        )
    else:
        output_param["num_epochs"] = int(default_size * param_original["pretraining_timescale"])


def prepare_cache(params, data_handlers=None):
    main_param_hash = get_param_hash(params, data_handlers)
    if "cache_dir_original" not in params:
        params["cache_dir_original"] = params["cache_dir"]

    # Override the cache_dir to put data in sub-dir of [param_hash]/
    params["cache_dir"] = os.path.join(params["cache_dir_original"], main_param_hash)
    print("cache_dir:", params["cache_dir"])
    params["split_cache_dir"] = os.path.join(params["cache_dir"], "splits")
    params["main_param_hash"] = main_param_hash
    if not params["cache_read_only"]:
        os.makedirs(params["split_cache_dir"], exist_ok=True)
        os.makedirs(params["cache_dir"], exist_ok=True)
    elif not os.path.exists(params["cache_dir"]):
        if params["dataset_names"] is not None:
            download_caches_from_s3(
                main_param_hash,
                params,
            )
    if params["input_dir_names"] is not None:
        input_dirs = []
        for name in params["input_dir_names"]:
            input_dirs.extend(get_input_dirs_for_dataset(name))
        params["input_dir"] = [normalize_path(path, params["input_dir_prefix"]) for path in input_dirs]
        print(f"For dataset {params['dataset_names']} set input-dir names to", params["input_dir"])

    hashing_params = get_hashing_params(params)
    print("hashing_params", hashing_params)

    # Write the params to the cached dir.
    params_filename = os.path.join(params["cache_dir"], "params.json")
    if not params["cache_read_only"] and not os.path.exists(params_filename):
        hashing_params = get_hashing_params(params)
        with open(params_filename, "w") as f:
            json.dump(hashing_params, f, indent=2)

    return params


def perform_training_schedule(
    param_sets,
    device: torch.device,
    logging_handlers=None,
    dataset_cls: Callable = PredictionDataset,
    prediction_instance_cls=ProtobufPredictionDataset,
    model_cls: Type[PredictionModelInterface] = PredictionModel,
    trainer_cls=PredictionProtobufTrainer,
):
    """
    Performs training based on the parameter set, propagating session name and dataset selection between
    training sessions.

    Parameters
    ----------
    param_sets: dict[dict]
        a list of parameter dictionaries.
    device: torch.device
        torch device to use
    logging_handlers: list
        List of additional logging handlers.

    Returns
    -------
    sessions:
        Training sessions
    """
    current_session_name = None
    optimizer_state = None
    datasets_name_lists = None
    sessions = []

    for params_level_i, params_key in enumerate(param_sets):
        params = param_sets[params_key]

        params["params_level"] = params_level_i
        params["params_key"] = params_key

        start_level = params["params_start_level"] or rad_misc.get_param_level_from_session_id(
            params["resume_session_name"]
        )

        if params_level_i < start_level:
            continue

        print("Starting parameters set {}: {}, for {} epochs".format(params_level_i, params_key, params["num_epochs"]))
        if current_session_name is not None:
            params["resume_session_name"] = current_session_name
        if optimizer_state is not None:
            params["resume_optimizer_state"] = optimizer_state

        if datasets_name_lists is not None:
            params["datasets_name_lists"] = datasets_name_lists

        if params["seed"] is not None:
            seed_everything(params["seed"])

        params, data_handlers, agent_map_handlers = prepare_cache_and_encoders(params)
        datasets, datasets_name_lists = load_datasets(
            params,
            data_transforms=data_handlers,
            dataset_cls=dataset_cls,
            prediction_instance_cls=prediction_instance_cls,
        )
        params["datasets_name_lists"] = datasets_name_lists  # Make the loaded datasets available to trainer logging

        # Setting the seed a second time ensures that we start training with the same seed independent of whether
        # load_datasets has created a new split (which is randomized) or simply used a loaded one.
        if params["seed"] is not None:
            seed_everything(params["seed"])

        # Create training model. input_dim is the past trajectories of the ado-agents, as everything else is optional.
        # agent_type_dim is the number of agent types (e.g., car, bicycle, motorcycle, pedestrian, large-vehicle, truck)

        worst_cases_folder = params["runner_output_folder"] if params["logger_type"] == "none" else None
        additional_logging_handlers = [
            ImageStatsLogHandler(params, ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES),
            LogWorstCases("g_cost", params, worst_cases_folder),
            LogWorstCases("distance_x0", params, worst_cases_folder),
        ]
        if logging_handlers is not None:
            additional_logging_handlers.extend(logging_handlers)

        # Add Waymo loggers for if Waymo dataset is used.
        if params["use_waymo_dataset"] and params["report_waymo_metrics"]:
            additional_logging_handlers.append(WaymoLogHandler(params))

        # We assume relevant_agents to contain the ego vehicle and one pedestrian for now.
        for agent_idx in range(params["max_agents"]):
            logger_key = f"fde_agent_{agent_idx}"
            additional_logging_handlers.append(LogWorstCases(logger_key, params, worst_cases_folder))

        prediction_model = create_prediction_model(
            params,
            device,
            agent_map_handlers,
            data_map_handler=data_handlers[0],
            prediction_model_cls=model_cls,
        )

        trainer = trainer_cls(datasets, prediction_model, params, device=device)
        profile_type = params.get("profiler_type", None)
        profile_duration = params.get("profiler_duration", None)
        current_session_name = trainer.get_session_name()
        trace_folder = os.path.join(params["logs_dir"], current_session_name, "profile")
        os.makedirs(trace_folder, exist_ok=True)

        with profiler.create_profiler(profile_type, profile_duration, params["num_epochs"], trace_folder) as profile:
            trainer.profiler = profile
            # Step now as the 1st wait schedule
            profile.step("trainer_start")
            trainer.train(additional_logging_handlers=additional_logging_handlers)
            profile.step("trainer_end")
        profiler.print_profile_result(profile)
        current_session_name = trainer.get_session_name()
        sessions.append(current_session_name)
        if params["pretraining_resume_optimizer"]:
            optimizer_state = trainer.get_optimizer_state()

        del prediction_model
        del trainer
    return sessions


def create_prediction_model(
    params,
    device,
    agent_map_handlers=None,
    data_map_handler=None,
    prediction_model_cls: type(PredictionModelInterface) = PredictionModel,
) -> PredictionModelInterface:
    """Factory method for creating the prediction model.

    Parameters
    ----------
    params : dict
        The global parameters dictionary.
    device : torch.device
        The device on which the model is to be stored.

    Returns
    -------
    torch.nn.Module
        The created prediction model.
    """
    if "learn_reward_model" in params.keys():

        def import_from_str(name):
            components = name.split(".")
            mod = __import__(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod

        prediction_model_cls = import_from_str("model_zoo.intent.prediction_reward_model.PredictionRewardModel")

    if params.get("use_linear_model", False):
        if not LinearPredictionModel:
            raise ValueError("Cannot use linear model without including it in the file list")
        return LinearPredictionModel(params=params)

    input_dim = 2
    node_hidden_dim = params["predictor_hidden_state_dim"]
    edge_hidden_dim = params["edge_hidden_state_dim"]
    intermediate_dim = params["predictor_hidden_state_dim"]
    dropout_ratio = params["dropout_ratio"]
    total_scene_dim = 0

    (
        input_encoders,
        agent_input_encoders,
    ) = create_encoders(params, data_map_handler, device)

    for input_encoder in input_encoders.values():
        total_scene_dim += input_encoder.out_features
    total_agent_dim = 0
    for agent_input_encoder in agent_input_encoders.values():
        total_agent_dim += agent_input_encoder.out_features
    print("Total agent dim: {}".format(total_agent_dim))
    agent_type_dim = len(params["agent_types"])

    if params["encoder_decoder_type"] == "polynomial":
        if not PolynomialEncoder or not PolynomialMultiAgentDecoder:
            raise ValueError(
                "Polynomial decoder requires PolynomialEncoder and PolynomialMultiAgentDecoder in the file list"
            )
        encoder_model = PolynomialEncoder(total_num_timepoints=params["past_timesteps"] + params["future_timesteps"])
        decoder_model = PolynomialMultiAgentDecoder(intermediate_dim, coordinate_dim=2, params=params)
        models = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        assert not params[
            "use_discriminator"
        ], "Use of a discriminator currently not supported for a polynomial encoder-decoder model."

    elif params["encoder_decoder_type"] == "lstm_mlp":
        encoder_model = LSTMEncoder(
            input_dim=input_dim,
            type_dim=agent_type_dim,
            traj_embedding_dim=intermediate_dim,
            h_dim=intermediate_dim,
            agent_dim=total_agent_dim,
            map_embedding_dim=params["map_layer_features"][0] * 2,
            device=device,
            params=params,
        )
        decoder_model = MLPDecoder(
            intermediate_dim,
            coordinate_dim=2,
            cumulative_decoding=params["cumulative_decoding"],
            truncated_steps=params["decoder_tbptt"],
            params=params,
        )
        models = {"encoder_model": encoder_model, "decoder_model": decoder_model}

    elif params["encoder_decoder_type"] == "gnn":

        if params["use_batch_graph_encoder"]:
            graph_encoder_class = BatchGraphEncoder
            if params["use_mlp_decoder"] and MLPDecoder is not None:
                decoder_class = MLPDecoder
            else:
                decoder_class = AugmentedMultiAgentDecoderAccelerated
                assert not params.get(
                    "use_hybrid_outputs", False
                ), "hybrid_outputs is not implemented in accelerated decoder, which is used with batch_graph_encoder"
        elif "use_multiagent_accelerated_decoder" in params and params["use_multiagent_accelerated_decoder"]:
            assert not params.get(
                "use_hybrid_outputs", False
            ), "hybrid_outputs is not implemented in accelerated decoder"
            graph_encoder_class = GraphEncoder
            decoder_class = AugmentedMultiAgentDecoderAccelerated
        else:
            graph_encoder_class = GraphEncoder
            decoder_class = AugmentedMultiAgentDecoder

        encoder_model = graph_encoder_class(
            input_dim,
            node_hidden_dim,
            edge_hidden_dim,
            intermediate_dim,
            agent_type_dim,
            scene_dim=total_scene_dim,
            agent_dim=total_agent_dim,
            params=params,
            nullify_nan_inputs=True,
            temporal_truncated_steps=params["encoder_tbptt"],
            leaky_relu=params["leaky_generator"],
        )
        decoder_model = decoder_class(
            intermediate_dim,
            coordinate_dim=2,
            params=params,
            cumulative_decoding=params["cumulative_decoding"],
            truncated_steps=params["decoder_tbptt"],
        )
        models = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        if params.get("weighted_samples"):
            generator_head_layers = [
                nn.Linear((intermediate_dim + agent_type_dim) * params["MoN_number_samples"], node_hidden_dim)
            ]
            generator_head_layers.extend(
                [nn.LeakyReLU(), nn.Dropout(p=dropout_ratio), nn.Linear(node_hidden_dim, params["MoN_number_samples"])]
            )
            generator_head = nn.Sequential(*generator_head_layers)
            models.update({"generator_head": generator_head})
        if params["use_discriminator"]:
            if params.get("learn_reward_model"):
                total_scene_dim = 0
                total_agent_dim = total_agent_dim
            discriminator_encoder = graph_encoder_class(
                input_dim,
                node_hidden_dim,
                edge_hidden_dim,
                intermediate_dim,
                agent_type_dim,
                scene_dim=total_scene_dim,
                agent_dim=total_agent_dim,
                params=params,
                nullify_nan_inputs=True,
                temporal_truncated_steps=params["encoder_tbptt"],
                leaky_relu=params["leaky_discriminator"],
            )
            # This should be different (e.g no future image inputs)
            discriminator_future_encoder = graph_encoder_class(
                input_dim,
                node_hidden_dim,
                edge_hidden_dim,
                intermediate_dim,
                agent_type_dim,
                scene_dim=0,
                agent_dim=0,
                params=params,
                nullify_nan_inputs=True,
                temporal_truncated_steps=-1,
                leaky_relu=params["leaky_discriminator"],
            )

            discriminator_head_layers = [nn.Linear(node_hidden_dim, node_hidden_dim)]
            if params["predictor_batch_norm"]:
                discriminator_head_layers.append(nn.BatchNorm1d(node_hidden_dim))

            discriminator_head_layers.extend(
                [nn.LeakyReLU(), nn.Dropout(p=dropout_ratio), nn.Linear(node_hidden_dim, 1)]
            )

            discriminator_head = nn.Sequential(*discriminator_head_layers)

            models.update(
                {
                    "discriminator_encoder": discriminator_encoder,
                    "discriminator_future_encoder": discriminator_future_encoder,
                    "discriminator_head": discriminator_head,
                }
            )
    elif params["encoder_decoder_type"] == "transformer" and TransformerEncoder is not None:
        encoder_model = TransformerEncoder(
            input_dim,
            params["map_layer_features"][0],
            params["predictor_hidden_state_dim"],
            intermediate_dim,
            params["num_encoder_transformer_heads"],
            params=params,
            num_transformer_block=params["num_encoder_transformer_blocks"],
            nullify_nan_inputs=True,
        )
        # Only use the MLP decoder when specified.
        if params["use_mlp_decoder"] and MLPDecoder is not None:
            decoder_class = MLPDecoder
        else:
            decoder_class = TransformerDecoder
        decoder_model = decoder_class(
            intermediate_dim,
            coordinate_dim=2,
            cumulative_decoding=params["cumulative_decoding"],
            truncated_steps=params["decoder_tbptt"],
            params=params,
        )
        models = {"encoder_model": encoder_model, "decoder_model": decoder_model}
        assert not params["use_discriminator"], "Transformer encoder in discriminator is not supported"
    else:
        raise ValueError("Not supported encoder/decoder type")

    additional_structure_callbacks = []
    additional_model_callbacks = []
    if params["trajectory_regularization_cost"] > 0:
        additional_structure_callbacks.append(TrajectoryRegularizationCost())
    if "additional_structure_callbacks" in params and params["additional_structure_callbacks"]:
        additional_structure_callbacks.extend(params["additional_structure_callbacks"])
    if "additional_model_callbacks" in params and params["additional_model_callbacks"]:
        additional_model_callbacks.extend(params["additional_model_callbacks"])

    model_encap = PredictionModelCodec(
        models=models,
        params=params,
        intermediate_dim=intermediate_dim,
        additional_structure_callbacks=additional_structure_callbacks,
        additional_model_callbacks=additional_model_callbacks,
    ).to(device)
    models.update({"model_encap": model_encap})
    prediction_model = prediction_model_cls(
        models=models,
        params=params,
        device=device,
        input_encoders=input_encoders,
        agent_input_encoders=agent_input_encoders,
        agent_input_handlers=agent_map_handlers,
        additional_model_callbacks=additional_model_callbacks,
    )

    return prediction_model


def perform_data_inference(params, device, additional_logging_handlers: list = None):
    if params["use_dummy_model"]:
        params["params_level"] = 0
    elif params["resume_session_name"] is not None:
        params["params_level"] = rad_misc.get_param_level_from_session_id(params["resume_session_name"])
    else:
        raise ValueError("Provide --resume-session-name or --use-dummy-model")

    if params["params_start_level"] is not None:
        params["params_level"] = params["params_start_level"]

    if params["seed"] is not None:
        seed_everything(params["seed"])

    params, data_handlers, agent_map_handlers = prepare_cache_and_encoders(params)

    datasets, _ = load_datasets(params, data_transforms=data_handlers)

    # Create training model. input_dim is the past trajectories of the ado-agents, as everything else is optional.
    # agent_type_dim is the number of agent types (e.g., car, bicycle, motorcycle, pedestrian, large-vehicle, truck)

    prediction_model = create_prediction_model(params, device, agent_map_handlers, data_map_handler=data_handlers[0])
    trainer = PredictionProtobufTrainer(datasets, prediction_model, params, device=device)

    profile_type = params.get("profiler_type", None)
    profile_duration = params.get("profiler_duration", None)
    current_session_name = trainer.get_session_name()
    trace_folder = os.path.join(params["logs_dir"], current_session_name, "profile")
    os.makedirs(trace_folder, exist_ok=True)

    with profiler.create_profiler(profile_type, profile_duration, params["num_epochs"], trace_folder) as profile:
        trainer.profiler = profile
        # Step now as the 1st wait schedule
        profile.step("trainer_start")
        trainer.train(additional_logging_handlers=additional_logging_handlers)
        profile.step("trainer_end")
    profiler.print_profile_result(profile)

    del prediction_model
    del trainer


def seed_everything(seed: int):
    """Manually set all relevant random seeds.

    Based on https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.8.3/pytorch_lightning/utilities/seed.py

    Args:
    :param seed: The random seed to be used in all random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingContext:
    params: dict
    dataloader_type: str = ""
    epoch_size: int = 0
    epoch: int = 0
    batch_itm_idx: int = 0
    global_batch_cnt: int = 0
    time_intervals: dict = field(default_factory=dict)

    @property
    def is_generator_update(self) -> bool:
        if not self.dataloader_type in ("train", "validation"):
            return False
        return self.batch_itm_idx % 2 == 0 or not self.params["use_discriminator"]

    @property
    def skip_visualization(self) -> bool:
        offset_epoch = self.epoch + self.params["vis_interval_offset"]
        return self.params["disable_visualizations"] or not offset_epoch % self.params["vis_interval"] == 0

    @property
    def skip_validation(self) -> bool:
        offset_epoch = self.epoch + self.params["val_interval_offset"]
        return self.params["disable_validation"] or not offset_epoch % self.params["val_interval"] == 0

    @staticmethod
    def from_checkpoint(params: dict) -> "TrainingContext":
        if resume_session_name := params.get("resume_session_name"):
            load_folder = os.path.join(params["model_load_folder"], resume_session_name)
            checkpoint = torch.load(os.path.join(load_folder, "checkpoint.tar"))
            ctx = TrainingContext(
                epoch=checkpoint["epoch"],
                global_batch_cnt=checkpoint["global_batch_cnt"],
                params=params,
            )
        else:
            ctx = TrainingContext(params=params)
        return ctx


class TrainerDebugger:
    """
    Simple class to pickle intermediate values during training.

    Useful for debugging and regression tests for training.
    """

    OUTPUT_FILENAME = "debug.pkl"

    def __init__(self) -> None:
        self.data = defaultdict(list)

    def record_value(self, key: str, value) -> None:
        if isinstance(value, dict):
            self.data[key].append(copy.deepcopy(value))
        elif isinstance(value, torch.Tensor):
            self.data[key].append(value.detach().cpu().numpy())
        else:
            self.data[key].append(value)

    def write_out(self, output_dir: str) -> None:
        file_path = os.path.join(output_dir, self.OUTPUT_FILENAME)
        with open(file_path, "wb") as fh:
            pickle.dump(self.data, fh)
