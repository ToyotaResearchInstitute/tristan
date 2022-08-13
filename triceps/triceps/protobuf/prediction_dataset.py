# Copyright 2020 Toyota Research Institute. All rights reserved.
import logging
import os
import pathlib

import google.protobuf
import numpy as np
from google.protobuf.json_format import MessageToDict
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_collate

from intent.multiagents.cache_utils import compute_hash, split_reading_hash
from radutils.misc import remove_prefix
from triceps.protobuf.prediction_dataset_cache import CacheElement
from triceps.protobuf.prediction_training_pb2 import PredictionInstance, PredictionSet
from triceps.protobuf.protobuf_data_names import (
    ADDITIONAL_INPUT_EGOVEHICLE,
    ADDITIONAL_INPUT_INTERACTIVE,
    ADDITIONAL_INPUT_RELEVANT_PEDESTRIAN,
)

# A parameter key for a map from source dataset, parameters to which prediction instances are valid.
from triceps.protobuf.protobuf_training_parameter_names import (
    PARAM_FUTURE_TIMESTEP_SIZE,
    PARAM_FUTURE_TIMESTEPS,
    PARAM_MAX_AGENTS,
    PARAM_PAST_TIMESTEP_SIZE,
    PARAM_PAST_TIMESTEPS,
)


class ProtobufDatasetParsingError(RuntimeError):
    pass


class InvalidProtoPredictionInstance(RuntimeError):
    pass


class ProtobufPredictionDataset(Dataset):
    # Timestamps for the prediction instance.
    DATASET_KEY_TIMESTAMPS = "timestamps"
    # The timestamp that distinguishes past from future.
    DATASET_KEY_PREDICTION_TIMESTAMP = "prediction_timestamp"
    # Positions of prediction agents.
    # [batch, agents, timestamps, (x,y,validity)]
    DATASET_KEY_POSITIONS = "positions"

    # Keys / IDs of the agents. dot id from tlog.
    # [batch, agents]
    DATASET_KEY_DOT_KEYS = "dot_keys"
    # Is the agent the vehicle from which measurements were taken?
    # [batch, agents]
    DATASET_KEY_IS_EGO_VEHICLE = "is_ego_vehicle"

    # Is this the main agent being predicted?
    # [batch, agents]
    DATASET_KEY_IS_RELEVANT_AGENT = "is_relevant_agent"
    # Which type of agent/vehicle is it? One-hot encoding
    # [batch, agents, 7 or 3], 7 or 3 types of agents defined in DEFAULT_AGENT_TYPES or WAYMO_AGENT_TYPES
    DATASET_KEY_AGENT_TYPE = "agent_type"

    # Additional inputs consumed downstream
    DATASET_KEY_ADDITIONAL_INPUTS = "additional_inputs__processing_only"
    DATASET_KEY_TRAJECTORIES = "trajectories__processing_only"
    DATASET_KEY_MAP_INFORMATION = "map_information__processing_only"
    DATASET_KEY_SEMANTIC_TARGETS = "semantic_targets__processing_only"

    # Selected agent index.
    # ??
    DATASET_KEY_AGENT_IDX = "agent_index"

    # In global frame, result of scale map.
    # [batch, num_map_points, 9(x, y, validity, type, tan_x, tan_y, normal_x, normal_y, id)]
    DATASET_KEY_MAP = "map"
    # Reuslt of scale_maps, maps in global frame
    # [batch, num_map_points, 2(x, y)]
    DATASET_KEY_MAP_COORDINATES = "map_coordinates"
    # How many points are considered past points?
    # [batch]
    DATASET_KEY_NUM_PAST_POINTS = "num_past_points"
    # Index of instance within the protobuf file
    # From resampling_filter
    DATASET_KEY_INSTANCE_INDEX = "instance_index"
    # A list of auxiliary information about the instance. JSON in String
    # [batch]
    DATASET_KEY_INSTANCE_INFO = "instance_info"
    # Saved global images for this prediction instance.
    DATASET_KEY_IMAGES = "images"
    # [batch, ] ??
    DATASET_KEY_IMAGES_MAPPING = DATASET_KEY_IMAGES + "_mapping"
    # Saved cropped images of the relevant agent.
    # [batch, ] ??
    DATASET_KEY_AGENT_IMAGES = "agent_images"
    DATASET_KEY_AGENT_IMAGES_STATS = DATASET_KEY_AGENT_IMAGES + "_stats"
    # Stores which image belongs to which agent at what timepoint.
    DATASET_KEY_AGENT_IMAGES_MAPPING = DATASET_KEY_AGENT_IMAGES + "_mapping"
    # Stub additional input.
    DATASET_KEY_STUB = "stub"
    # Stub additional map input.
    DATASET_KEY_MAP_STUB = "map_stub"
    # Stub additional agent input.
    DATASET_KEY_AGENT_STUB = "agent_stub"
    # Stub additional agent map input.
    DATASET_KEY_AGENT_MAP_STUB = "agent_map_stub"
    # Headings of agents.
    DATASET_KEY_HEADINGS = "headings"
    # Name of the corresponding protobuf file.
    DATASET_KEY_PROTOBUF_FILE = "protobuf_file"
    # Semantic labels / output keys.
    DATASET_KEY_SEMANTIC_LABELS = "semantic_labels"
    # Width length height output keys
    DATASET_KEY_WLHS = "wlhs"
    # Velocity output keys
    DATASET_KEY_VELOCITIES = "velocities"
    # Collision detected output keys.
    DATASET_KEY_COLLISION = "collision"
    # Language tokens input.
    DATASET_KEY_LANGUAGE_TOKENS = "language_tokens"
    # Auxiliary additional states input, including (length, width, heading, velocity_x, velocity_y).
    DATASET_KEY_AUXILIARY_STATE = "auxiliary_state"

    # [For hybrid prediction] High level modes of prediction agents.
    # [batch, agents, timestamps, maneuver_size]
    DATASET_KEY_MANEUVERS = "maneuvers"
    DATASET_KEY_MANEUVERS_PAST = "maneuvers_past"
    DATASET_KEY_MANEUVERS_FUTURE = "maneuvers_future"
    DATASET_KEY_LANE_CHANGES = "lane_changes"
    DATASET_KEY_LANE_CHANGES_PAST = "lane_changes_past"
    DATASET_KEY_LANE_CHANGES_FUTURE = "lane_changes_future"
    DATASET_KEY_LANE_INDICES = "lane_indices"
    DATASET_KEY_LANE_INDICES_PAST = "lane_indices_past"
    DATASET_KEY_LANE_INDICES_FUTURE = "lane_indices_future"

    def __init__(
        self,
        filename,
        data_transforms: list = None,
        trajectory_handler=None,
        agent_selection_handler=None,
        mock_data_mode=None,
        additional_input_stub=False,
        input_stub_dim=64,
        additional_agent_input_stub=False,
        agent_input_stub_dim=64,
        params=None,
        check_valid_instances=False,
    ):
        """
        Dataset for loading a protobuf for prediction.
        :param filename: protobuf data .pb filename. This data file will contain a PredictionSet.
        :param data_transforms: A list of transforms to the data, e.g map representation loading.
        :param trajectory_handler: A handler for trajectories, e.g. filtering, etc.
        :param agent_selection_handler: For subselecting agents.
        :param mock_data_mode: Use original data if this is set to None or False. Otherwise, this has to be a callable
            that is used to generate the mock data.
        :param additional_input_stub: If true, add an additional inputs vector with synthetic data.
        :param input_stub_dim: The dimension of the stub additional input vector.
        :param additional_agent_input_stub: If true, add an additional agent inputs vector with synthetic data.
        :param agent_input_stub_dim: The dimension of the sub additional agent input vector.
        :param params: Additional parameters. This includes:

            - max_agents: Number of agents to allow / pad for in the data item.
            - past_timesteps: Number of past timesteps to allow / pad for in the data item.
            - future_timesteps: Number of future timesteps to allow / pad for in the data item.
            - past_timestep_size: Past timestep size. Data can be resampled to that time step after loading.
            - future_timestep_size: Future timestep size. Data can be resampled to that time step after loading.
        :param check_valid_instances: When True, check for valid instance immediately, False, will defer it when __getitem__
        """
        self.main_param_hash = params["main_param_hash"]
        self.init_params(params)

        self.transpose_agent_times = True
        self.data_transforms = data_transforms or []
        self.trajectory_handler = trajectory_handler
        self.agent_selection_handler = agent_selection_handler
        self.data = None
        self.debug_mode = mock_data_mode
        self.additional_input_stub = additional_input_stub
        # TODO(guy.rosman): Take from params.
        self.input_stub_dim = input_stub_dim
        self.additional_agent_input_stub = additional_agent_input_stub
        self.agent_input_stub_dim = agent_input_stub_dim

        if params is None:
            params = {}
        self.filename = filename
        # Remove log dir prefix (/home/user), so rel_filename is the same across difference machine/accounts.
        self.rel_filename = remove_prefix(filename, self.input_dir_prefix)
        self.dataitem_hash = self.compute_dataitem_hash()
        # Uninitialized state.
        self.num_total_instances = None

        if self.debug_mode:
            self.valid_instances = range(10)
            self.num_total_instances = 10
        elif check_valid_instances:
            self._check_valid_instances()

    def init_params(self, params, data_transforms=None, trajectory_handler=None, agent_selection_handler=None):
        """Init the params for the Dataset. Should be called when reading the cached dataset,
        since the self.params is save from the previous run.
        The parameters affecting the data, should be the same, only 'cache_dir' and model related param would change.

        params: dict
            New params set to this Dataset.
        """
        if hasattr(self, "params") and self.params is not None:
            assert params["agent_types"] == self.params["agent_types"]
            assert params[PARAM_MAX_AGENTS] == self.params[PARAM_MAX_AGENTS]
            assert params[PARAM_PAST_TIMESTEPS] == self.params[PARAM_PAST_TIMESTEPS]
            assert params[PARAM_FUTURE_TIMESTEPS] == self.params[PARAM_FUTURE_TIMESTEPS]
            assert params[PARAM_PAST_TIMESTEP_SIZE] == self.params[PARAM_PAST_TIMESTEP_SIZE]
            assert params[PARAM_FUTURE_TIMESTEP_SIZE] == self.params[PARAM_FUTURE_TIMESTEP_SIZE]
            if PARAM_PAST_TIMESTEP_SIZE not in params:
                params["timestep_size"] = params[PARAM_PAST_TIMESTEP_SIZE]
        self.input_dir_prefix = params["input_dir_prefix"]
        self.cache_folder = params.get("cache_dir", None)
        self.max_agents = params[PARAM_MAX_AGENTS]
        self.past_timesteps = params[PARAM_PAST_TIMESTEPS]
        self.future_timesteps = params[PARAM_FUTURE_TIMESTEPS]
        self.total_timesteps = self.future_timesteps + self.past_timesteps
        self.past_delta_t = params[PARAM_PAST_TIMESTEP_SIZE]
        self.future_delta_t = params[PARAM_FUTURE_TIMESTEP_SIZE]
        self.inference_mode = params["inference_mode"] if "inference_mode" in params else False

        self.debug_mode = params["data_debug_mode"]
        self.additional_input_stub = params["add_input_stub"]
        self.input_stub_dim = params["input_stub_dim"]
        self.agent_map_input_stub = params["add_agent_map_input_stub"]
        self.agent_map_input_stub_dim = params["agent_map_input_stub_dim"]

        self.additional_map_input_stub = params["add_input_map_stub"]
        self.input_map_stub_dim = params["input_map_stub_dim"]
        self.max_map_points = params["map_points_max"]

        self.additional_agent_input_stub = params["add_agent_input_stub"]
        self.agent_input_stub_dim = params["agent_input_stub_dim"]

        self.item_post_processors = params.get("item_post_processors", [])

        self.agent_types = params.get("agent_types")
        self.params = params
        if data_transforms is not None:
            self.data_transforms = data_transforms
        if trajectory_handler is not None:
            self.trajectory_handler = trajectory_handler
        if agent_selection_handler is not None:
            self.agent_selection_handler = agent_selection_handler

        if hasattr(self, "rel_filename") and self.rel_filename is not None:
            # When loading cache, replace the dir prefix with the dir prefix of this machine.
            if "test_veh_data" in self.rel_filename:
                # test_veh_data is stored separately from the main input directories
                self.filename = os.path.expanduser(os.path.join(params["global_dir_prefix"], self.rel_filename))
            else:
                self.filename = os.path.join(self.input_dir_prefix, self.rel_filename)

    def check_if_instance_exists(self):
        """When using cache per data handler, the pb instance must be present.
        This function checks if it is present on this machine.
        """
        if self.debug_mode:
            return
        assert pathlib.Path(self.filename).is_file(), (
            f"When not using whole cache item, the pb file must be available. "
            f"can't find file prefix: {self.input_dir_prefix}, rel_filename: {self.rel_filename}"
        )

    def read_data(self) -> PredictionSet:
        """Reads the protobuf data.
        Returns
        -------
        data: PredictionSet
            A protobuf PredictionSet.
        """
        with open(self.filename, "rb") as fp:
            data = PredictionSet()
            try:
                data.ParseFromString(fp.read())
            except google.protobuf.message.DecodeError as e:
                print(f"Failed to parse {self.filename}")
                raise e
        return data

    def _check_valid_instances(self):
        """Check which examples from the file are valid.

        Parameters
        ----------
        data: PredictionSet
            A protobuf PredictionSet, the data from the loaded protobuf.
        Returns
        -------
        valid_instances: list
            A list of valid prediction instance indices from the PredictionSet.
        """
        cache_name = compute_hash(self.main_param_hash + f"parse_{self.rel_filename}") + "_instance_counts"
        instance_cache_element = CacheElement(
            os.path.join(self.cache_folder, "main_cache"),
            cache_name,
            "pkl",
            should_lock=self.params["use_cache_lock"],
            read_only=self.params["cache_read_only"],
            disable_cache=self.params["disable_cache"],
        )
        if instance_counts := instance_cache_element.load():
            valid_instances, num_total_instances = instance_counts
        else:
            protobuf_data = self.read_data()
            num_total_instances = len(protobuf_data.prediction_instances)
            valid_instances = []
            should_cache_whole_item = self.params["cache_dataset_item"]

            if (self.debug_mode is not None and self.debug_mode) or self.inference_mode:
                for i, instance in enumerate(protobuf_data.prediction_instances):  # pylint: disable=unused-variable
                    # Check whether the instance is valid -- right now we assume all samples are valid, unless
                    # reading them failed.
                    valid_instances.append(i)
            else:
                for i, instance in enumerate(protobuf_data.prediction_instances):
                    # See that we can parse this instance and it's valid.
                    try:
                        return_item, _, is_cached = self.parse_instance(
                            i, i, should_cache_whole_item, proto_prediction_instance=instance
                        )
                    except InvalidProtoPredictionInstance:
                        logging.warning(f"protobuf instance is invalid, skipping, {self.filename}, instance_idx: {i}")
                        continue
                    except ValueError as err:
                        raise ValueError(f"Error when parsing instance: {self.filename}, i: {i}, error: {err}")
                    valid_instances.append(i)

            instance_cache_element.save((valid_instances, num_total_instances))

        self.num_total_instances = num_total_instances
        self.valid_instances = valid_instances

    def __len__(self):
        return len(self.valid_instances)

    def parse_instance(
        self,
        item: int,
        instance_idx: int,
        should_cache_whole_item: bool,
        precheck: bool = False,
        proto_prediction_instance: PredictionInstance = None,
    ) -> tuple:
        # Read the protobuf, parse and transform it.
        # TODO(guy.rosman): extend the cache to the transformations, avoid any parsing
        cache_name = compute_hash(self.main_param_hash + f"parse_{self.rel_filename}_{item}")
        cache_element = CacheElement(
            os.path.join(self.cache_folder, "main_cache"),
            cache_name,
            "pkl",
            should_lock=self.params["use_cache_lock"],
            read_only=self.params["cache_read_only"],
            disable_cache=self.params["disable_cache"],
        )
        cached_data = cache_element.load()

        if should_cache_whole_item and cached_data is not None:
            # Needs to add stubs for pretraining.
            cached_data.update(self.process_stubs())
            # If we cached the whole item, then just return it
            return cached_data, cache_element, True

        # Each data handler is cached separately.
        if cached_data is not None:
            return_item = cached_data
        else:
            if proto_prediction_instance is None:
                data = self.read_data()
                instance: PredictionInstance = data.prediction_instances[instance_idx]
            else:
                instance = proto_prediction_instance
            return_item = self.post_parsing_process(*self._parse_instance(instance, precheck))
            return_item[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INDEX] = instance_idx
            if not should_cache_whole_item:
                cache_element.save(return_item)
                cached_data = return_item
        is_cached = cached_data is not None
        return return_item, cache_element, is_cached

    def _parse_instance(self, instance: PredictionInstance, precheck: bool = False) -> tuple:
        """Read the protobuf prediction instance, handle the trajectory, select which agents, and return a tuple of
        all the data.

        Parameters
        ----------
        instance: triceps.protobuf.prediction_training_pb2.PredictionInstance
            The prediction instance.
        precheck: bool
            Whether we're prechecking the data or actually loading for training/inference.

        Returns
        -------
        instance_info:
            The prediction instance info.
        positions
        timestamps
        prediction_timestamp
        dot_keys: list
            The keys for the different agents loaded in the prediction instance.
        is_ego_vehicle,
        is_relevant_agent
        agent_type_vector
        idx: list
            The agent indices list.
        is_invalid_instance: bool
            is the instance invalid.
        """
        # TODO(guy.rosman) Complete documentation of the outputs.
        # TODO(guy.rosman) Check loading efficiency.
        positions_list = []
        dot_key_list = []

        instance_as_dict = MessageToDict(instance)
        additional_inputs_full = {
            key: traj["additionalInputs"] for key, traj in instance_as_dict["agentTrajectories"].items()
        }
        trajectory_data_full = {key: traj["trajectory"] for key, traj in instance_as_dict["agentTrajectories"].items()}

        is_ego_vehicle_list = []
        is_relevant_agent_list = []
        agent_type_vector_list = []
        additional_inputs = []
        trajectories = []
        instance_info = instance.prediction_instance_info
        prediction_timestamp = np.float32(instance.prediction_time.ToNanoseconds()) / 1e9
        is_invalid_instance = False
        positions = None
        dot_keys = None
        is_ego_vehicle = None
        is_relevant_agent = None
        agent_type_vector = None

        idx = None
        if self.params["augmentation_timestamp_scale"] > 0 and not precheck:
            import scipy.stats

            ndt = scipy.stats.truncnorm.rvs(-1, 1, size=1) * self.params["augmentation_timestamp_scale"]
        else:
            ndt = 0.0
        for k_i, k in enumerate(sorted(instance.agent_trajectories.keys())):
            traj = instance.agent_trajectories[k]
            positions_k = []
            timestamps_k = []
            dot_key_k = 0
            is_ego_vehicle_k = 0
            is_relevant_agent_k = 0
            agent_type_vector_k = np.zeros(len(self.agent_types))
            is_too_short = False
            is_inaccurate = False
            is_crucially_too_short = False
            for point in traj.trajectory:
                is_position_valid = abs(point.position.x) > 0 or abs(point.position.y) > 0
                positions_k.append([point.position.x, point.position.y, float(is_position_valid)])
                timestamps_k.append(point.timestamp.ToNanoseconds() / 1e9 - prediction_timestamp)
            positions_k = np.array(positions_k)
            timestamps = np.array(timestamps_k)
            additional_inputs.append(additional_inputs_full[k])
            trajectories.append(trajectory_data_full[k])
            if traj.additional_inputs:
                for inp in traj.additional_inputs:
                    if (
                        hasattr(inp, "vector_input_type_id")
                        and inp.vector_input_type_id == "additional_input_key_tri_dot_id"
                    ):
                        dot_key_k = inp.vector_input[0]
                    if hasattr(inp, "vector_input_type_id") and inp.vector_input_type_id == ADDITIONAL_INPUT_EGOVEHICLE:
                        # set ego-vehicle key to -2.
                        dot_key_k = -2
                        is_ego_vehicle_k = 1
                    if self.params["train_waymo_interactive_agents"]:
                        # Label relevant agents as interactive agents.
                        if (
                            hasattr(inp, "vector_input_type_id")
                            and inp.vector_input_type_id == ADDITIONAL_INPUT_INTERACTIVE
                        ):
                            is_relevant_agent_k = 1
                    else:
                        if (
                            hasattr(inp, "vector_input_type_id")
                            and inp.vector_input_type_id == ADDITIONAL_INPUT_RELEVANT_PEDESTRIAN
                        ):
                            is_relevant_agent_k = 1
                    if hasattr(inp, "vector_input_type_id") and inp.vector_input_type_id == "additional_input_key_type":
                        try:
                            type_idx = self.agent_types.index(int(inp.vector_input[0]))
                        except IndexError:
                            import IPython

                            IPython.embed(header="agent type error")
                        agent_type_vector_k[type_idx] = 1
            else:
                require_additional_inputs = False
                if require_additional_inputs:
                    import IPython

                    IPython.embed(header="handle missing additional_inputs")

            # Use agent id as dot key for language-based predictor since we need all agent ids.
            if "use_language" in self.params and self.params["use_language"]:
                dot_key_k = k

            is_ego_or_relevant = is_ego_vehicle_k + is_relevant_agent_k

            # TODO(blaise.bruer) We may actually need a trajectory_handler for inference_mode, but for rapid development
            # TODO  reasons we are skipping it in inference_mode.
            if self.trajectory_handler is None and not self.inference_mode:
                positions_k = positions_k[: self.total_timesteps, :]
                timestamps = timestamps[: self.total_timesteps]
            else:
                handler_params = {
                    "is_ego_vehicle": is_ego_vehicle_k,
                    "is_relevant_agent": is_relevant_agent_k,
                    "is_precheck": precheck,
                }
                positions_k, timestamps, is_too_short, is_crucially_too_short, is_inaccurate = self.trajectory_handler(
                    positions_k, timestamps + ndt, self.params, k_i, self.rel_filename, handler_params
                )

            positions_k = positions_k[: self.total_timesteps, :]
            timestamps = timestamps[: self.total_timesteps]
            if len(traj.trajectory) == 0:
                print("Skipping empty trajectory")
                continue
            if is_too_short or is_crucially_too_short or is_inaccurate:
                # Return immediately if a relevant or ego vehicle is bad.
                if is_ego_or_relevant and not self.inference_mode:
                    is_invalid_instance = True
                    return (
                        instance_info,
                        positions,
                        timestamps,
                        prediction_timestamp,
                        dot_keys,
                        is_ego_vehicle,
                        is_relevant_agent,
                        agent_type_vector,
                        idx,
                        is_invalid_instance,
                    )
                # Otherwise skip.
                else:
                    # if is_invalid_trajectory -- do not populate dot_valid_key_list
                    continue

            # TODO(guy.rosman): populate partially for invalid trajectories, rule them out later.
            assert len(positions_k) == self.total_timesteps, "k_i = {}, filename = {}".format(k_i, self.filename)
            assert len(timestamps) == self.total_timesteps, "k_i = {}, filename = {}".format(k_i, self.filename)
            positions_list.append(positions_k)
            dot_key_list.append(dot_key_k)
            is_ego_vehicle_list.append(is_ego_vehicle_k)
            is_relevant_agent_list.append(is_relevant_agent_k)
            agent_type_vector_list.append(agent_type_vector_k)
        timestamps += prediction_timestamp

        if is_invalid_instance:
            raise InvalidProtoPredictionInstance()
        try:
            positions = np.array(positions_list).transpose((1, 0, 2))
        except ValueError as e:
            print("Positions size: {}".format(positions.shape))
            raise e
        # dim of positions is: [n_timestamps,n_agents,3]
        if not positions.shape[0] == timestamps.size == self.total_timesteps:
            import IPython

            IPython.embed(
                header="incorrect number of timestamps in trajectories: {}".format(
                    [positions.shape[0], timestamps.size, self.total_timesteps]
                )
            )
        dot_keys = np.array(dot_key_list)
        is_ego_vehicle = np.array(is_ego_vehicle_list)
        is_relevant_agent = np.array(is_relevant_agent_list)
        agent_type_vector = np.array(agent_type_vector_list)

        # Constrain the (steps, agents) to (max_t, max_k) by choosing the nearest agents with some handler.
        # import IPython;IPython.embed(header='before selector')
        if self.agent_selection_handler:

            # TODO(guy.rosman): have the trajectory handler get dot_valid_key_list and use it to prune
            (
                positions,
                is_ego_vehicle,
                is_relevant_agent,
                dot_keys,
                agent_type_vector,
                additional_inputs,
                trajectories,
                idx,
            ) = self.agent_selection_handler(
                self.params,
                positions,
                is_ego_vehicle,
                is_relevant_agent,
                dot_keys,
                agent_type_vector,
                additional_inputs,
                trajectories,
                self.params["ego_agent_only"],
                self.params["ignore_ego"],
                filename=self.rel_filename,
            )
        else:
            idx = range(len(dot_keys))
        is_ego_vehicle = is_ego_vehicle.astype(int)
        is_relevant_agent = is_relevant_agent.astype(int)
        dot_keys = dot_keys.astype(int)
        idx = np.array(idx)
        return (
            instance_info,
            positions,
            timestamps,
            prediction_timestamp,
            dot_keys,
            is_ego_vehicle,
            is_relevant_agent,
            agent_type_vector,
            additional_inputs,
            trajectories,
            instance_as_dict["mapInformation"],
            instance_as_dict.get("semanticTargets", None),
            idx,
            is_invalid_instance,
        )

    def post_parsing_process(
        self,
        instance_info: str,
        positions: np.array,
        timestamps: np.array,
        prediction_timestamp: np.float64,
        dot_keys: np.array,
        is_ego_vehicle: np.array,
        is_relevant_agent: np.array,
        agent_type_vector: np.array,
        additional_inputs: list,
        trajectories: list,
        map_information: dict,
        semantic_targets,
        idx: np.array,
        is_invalid_instance: bool,
    ) -> dict:
        """Process the data item after protobuf parsing.

        Parameters
        ----------
        instance_info: str
            The instance info string.
        positions: np.array
            The (num_agents, num_timepoints, 3) array of (x,y,validity) positions.
        timestamps: np.array
            The timepoints for each position, a num_timepoints-long array
        prediction_timestamp: np.float64
            The point in time that presents the "current time".
        dot_keys: np.array
            A num_agents-long np.array with agent ids.
        is_ego_vehicle: np.array
            A num_agents-long np.array with boolean - is the agent an ego-vehicle?
        is_relevant_agent: np.array
            A num_agents-long np.array with boolean - is the agent a relevant agent
            (to be predicted and compute cost on)?
        agent_type_vector: np.array
            A (num_agents,num_agent_types) np.array with a one-hot vector representation for which agent type it is.
        idx: np.array
            A num_agents long np.array with the indices of the agents in the protobuf.
        is_invalid_instance
            A boolean - is the instance valid or not. Some instances may not be able to be processed correctly and
            should be invalidated during processing.

        Returns
        -------
        return_item: dict
            A dictionary with all of the processed positions, maps, images, ready to be tensorized.
            See PredictionDataset.DATASET_KEY_XXX for difference keys.

        """
        if self.transpose_agent_times and not is_invalid_instance:
            try:
                positions = np.transpose(positions, [1, 0, 2])
            except ValueError as e:
                import IPython

                IPython.embed()
                if positions is None:
                    print("Positions is none")
                else:
                    print("Positions size: {}".format(positions.shape))
                raise e

        # Add dimension to idx if it is a single value.
        idx = np.array(idx)
        if not idx.shape:
            idx = idx[np.newaxis]

        # Initialize agent index with -1's, as the valid index should be non-negative.
        agent_idx_full = -1 * np.ones_like(is_relevant_agent)
        agent_idx_full[: idx.shape[0]] = idx

        # Basic return dictionary.
        return_item = {
            self.DATASET_KEY_TIMESTAMPS: timestamps,
            # prediction_timestamp is the first time step to be predicted.
            self.DATASET_KEY_PREDICTION_TIMESTAMP: prediction_timestamp,
            self.DATASET_KEY_POSITIONS: positions,
            self.DATASET_KEY_DOT_KEYS: dot_keys,
            self.DATASET_KEY_IS_EGO_VEHICLE: is_ego_vehicle,
            self.DATASET_KEY_IS_RELEVANT_AGENT: is_relevant_agent,
            self.DATASET_KEY_AGENT_IDX: agent_idx_full,
            self.DATASET_KEY_AGENT_TYPE: agent_type_vector,
            self.DATASET_KEY_ADDITIONAL_INPUTS: additional_inputs,
            self.DATASET_KEY_TRAJECTORIES: trajectories,
            self.DATASET_KEY_MAP_INFORMATION: map_information,
            self.DATASET_KEY_SEMANTIC_TARGETS: semantic_targets,
            self.DATASET_KEY_NUM_PAST_POINTS: self.params[PARAM_PAST_TIMESTEPS],
            self.DATASET_KEY_INSTANCE_INFO: instance_info,
            self.DATASET_KEY_PROTOBUF_FILE: self.rel_filename,
        }

        # Update the stubs if needed.
        return_item.update(self.process_stubs())

        return return_item

    def process_stubs(self) -> dict:
        """Add input stubs.

        Returns
        -------
        return_item: dict
            A dictionary with all of the stub input.
            See PredictionDataset.DATASET_KEY_XXX for difference keys.

        """
        return_item = {}
        if self.additional_input_stub:
            return_item[self.DATASET_KEY_STUB] = np.float32(
                np.random.normal(size=(self.total_timesteps, self.input_stub_dim))
            )
        if self.additional_map_input_stub:
            return_item[self.DATASET_KEY_MAP_STUB] = np.float32(
                np.random.normal(size=(self.max_map_points, self.input_map_stub_dim))
            )

        # Add additional agent input stub, for testing purposes.
        if self.additional_agent_input_stub:
            return_item[self.DATASET_KEY_AGENT_STUB] = np.float32(
                np.random.normal(size=(self.max_agents, self.total_timesteps, self.agent_input_stub_dim))
            )
        if self.agent_map_input_stub:
            return_item[self.DATASET_KEY_AGENT_MAP_STUB] = np.float32(
                np.random.normal(size=(self.max_agents, self.total_timesteps, self.agent_map_input_stub_dim))
            )
        return return_item

    def compute_dataitem_hash(self):
        dataitem_hash = compute_hash(f"parse_{self.rel_filename}")
        return dataitem_hash

    def __getitem__(self, item):
        if self.num_total_instances is None:
            # Haven't initialized/checked instance validity yet.
            self._check_valid_instances()
        instance_idx = self.valid_instances[item]
        should_cache_whole_item = self.params["cache_dataset_item"]
        if self.debug_mode is not None and self.debug_mode:
            # Mode for fast-and-fake data.
            synthetic_data_dictionary = self.debug_mode(
                self.params, self.total_timesteps, self.max_agents, item, self.agent_types, self.filename
            )
            positions = synthetic_data_dictionary["positions"]
            timestamps = synthetic_data_dictionary["timestamps"]
            prediction_timestamp = synthetic_data_dictionary["prediction_timestamp"]
            dot_keys = synthetic_data_dictionary["dot_keys"]
            is_relevant_agent = synthetic_data_dictionary["is_relevant_pedestrian"]
            is_ego_vehicle = synthetic_data_dictionary["is_ego_vehicle"]
            agent_type_vector = synthetic_data_dictionary["agent_type_vector"]
            instance_info = synthetic_data_dictionary["instance_info"]
            idx = synthetic_data_dictionary["idx"]

            return_item = self.post_parsing_process(
                instance_info,
                positions,
                timestamps,
                prediction_timestamp,
                dot_keys,
                is_ego_vehicle,
                is_relevant_agent,
                agent_type_vector,
                None,
                None,
                None,
                None,
                idx,
                False,
            )
            return_item[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INDEX] = instance_idx
        else:
            return_item, cache_element, is_cached = self.parse_instance(item, instance_idx, should_cache_whole_item)

        try:
            self._run_data_handler(return_item, item)
        except Exception as e:
            print(f"error {e} for file {self.rel_filename}")
            raise e
        self._remove_unused_items(return_item)

        return_item.update(self.process_stubs())

        # Do filtering/update with custom processors.
        # Note: the process function must be picklable since this is used by torch.Dataloader.
        for processor in self.item_post_processors:
            return_item = processor(return_item, self.params)

        if should_cache_whole_item and "cache_element" in locals() and not is_cached:
            # Save the whole item after all processing has been done.
            cache_element.save(return_item)
        return return_item

    @staticmethod
    def _remove_unused_items(return_item):
        """Remove things that are not needed during training and cannot be collated by pytorch"""
        # Remove any keys related to cache
        # This is set in the GlobalImageHandler/AgentImageHandler
        for key in list(return_item.keys()):
            if key.endswith("__processing_only"):
                return_item.pop(key)

    def remove_params(self):
        """Remove the params before it get serialized.
        Very IMPORTANT!! Don't let the params to be serialized.
          Params shouldn't be serialized within individual which will cause
          1. Memory leaks on the main process during the multiprocessing load function. This is impossible to
              track with tracemalloc, which only tracks memory created by Python API, not C functions.
              Related issue: https://github.com/pytorch/pytorch/issues/13246
          2. increase over all size of the cache.

          And be careful NOT to serialize handlers, callbacks in params.
        """
        self.params = None
        self.data_transforms = None

    def _run_data_handler(self, return_item, item):
        # Do data transforms -- extract more things from the protobuf instance into the return dictionary.
        for tform in self.data_transforms:
            if tform is not None:
                return_item = tform.process(return_item, self.params, self.rel_filename, item)

    def __iter__(self):
        def generator():
            for i in range(len(self.valid_instances)):
                yield self.__getitem__(i)

        return generator()


class PredictionDataset(ConcatDataset):
    def __init__(self, datasets, params, dataset_label=None):
        # Params and dataset_label are expected to be used by subclasses
        self.params = params
        self.dataset_label = dataset_label
        super().__init__(datasets)


def dataset_collate_fn(data):
    ret = {}

    DATASET_KEY = "dataset"
    ITEMS_KEY = "result_items"
    if isinstance(data[0], dict):
        if DATASET_KEY in data[0] and isinstance(data[0][DATASET_KEY], ProtobufPredictionDataset):
            ret = {DATASET_KEY: [d.pop(DATASET_KEY, None) for d in data]}
    if isinstance(data[0], dict):
        if ITEMS_KEY in data[0] and isinstance(data[0][ITEMS_KEY], dict):
            ret[ITEMS_KEY] = [d.pop(ITEMS_KEY, None) for d in data]
    default_ret = default_collate(data)
    default_ret.update(ret)
    return default_ret


class DatasetValidation_Dataset(Dataset):
    """This dataset is used to run parallel validation onf the ProtobufPredictionDataset"""

    def __init__(self, prediction_dataset: ProtobufPredictionDataset):
        self.prediction_dataset = prediction_dataset

    def __len__(self):
        # This function is called before self.prediction_dataset._check_valid_instances(), where
        #   self.prediction_dataset.valid_instances is initialized.
        # So setting it to 1 and handle multiple instances outside this class, in the dataloading loop.
        return 1

    def __getitem__(self, index):
        ret = {"filename": self.prediction_dataset.rel_filename, "index": index, "dataset": self.prediction_dataset}
        try:
            self.prediction_dataset._check_valid_instances()
        except ProtobufDatasetParsingError as e:
            logging.error(f"error parsing file: {ret}, error: {str(e)}")
            ret["valid"] = False
            return
        # Assuming one instance per prediction dataset
        ret["valid"] = True
        if len(self.prediction_dataset.valid_instances) == 0:
            # No valid instance
            ret["valid"] = False
        else:
            if len(self.prediction_dataset.valid_instances) > 1:
                raise RuntimeError("We currently assume one instance per prediction_dataset")
            for idx in self.prediction_dataset.valid_instances:
                # __getitem__ on each instance which will trigger cache generation.
                item = self.prediction_dataset.__getitem__(idx)
                # Assuming one instance per prediction dataset
                ret["result_items"] = item

        self.prediction_dataset.remove_params()
        return ret

    def __iter__(self):
        def generator():
            for i in range(1):
                yield self.__getitem__(i)
