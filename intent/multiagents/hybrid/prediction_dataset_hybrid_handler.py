import copy
from typing import List

import numpy as np

import triceps.protobuf
from data_sources.argoverse.argoverse_hybrid_utils import MODE_DICT_MANEUVER_TO_IDX, MODE_DICT_MANEUVER_TO_IDX_ONLINE
from data_sources.argoverse.create_adovehicle_maneuver_labels_argoverse import label_driving_mode_sequence
from intent.multiagents.hybrid.hybrid_prediction_utils import hybrid_label_smooth_filter
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_auxiliary import InputsHandler
from triceps.protobuf.protobuf_training_parameter_names import (
    PARAM_FUTURE_TIMESTEP_SIZE,
    PARAM_FUTURE_TIMESTEPS,
    PARAM_MAX_AGENTS,
    PARAM_PAST_TIMESTEP_SIZE,
    PARAM_PAST_TIMESTEPS,
)


class DiscreteModeLabelHandler(InputsHandler):
    """
    Adds discrete mode label to the dataset taken from prediction_instance_info.
    Note: Discrete mode is usually assumed to be a hidden variable and not (or partially) known.
    This is intended to validate hybrid prediction performance at test time, but
    should not (in most cases) be used in training time as supervisory cues.
    """

    def __init__(self, params):
        super().__init__(params)
        # Whether to smooth lane indices.
        self.smooth_lane_indices = True

    def smooth_index_label(self, lane_indices):
        """
        Smooth a list of indices, such that there is no subsequences that start and end with the same index, but has
        different indices in between.
        This is used to provide cleaner labels when an agent follows a lane but crosses other lanes.

        Parameters
        ----------
        lane_indices: numpy.ndarray
            Input lane indices.

        Returns
        -------
        lane_indices: numpy.ndarray
            Smoothed lane indices.

        """
        lane_indices = copy.deepcopy(lane_indices)
        start_index = lane_indices[0]
        for i in range(1, len(lane_indices)):
            if lane_indices[i] == start_index:
                start_index = lane_indices[i]
            else:
                noise = False
                for j in range(i + 1, len(lane_indices)):
                    if lane_indices[j] == start_index:
                        noise = True
                        for k in range(i, j):
                            lane_indices[k] = start_index
                        i = j
                        break
                if not noise:
                    start_index = lane_indices[i]
        return lane_indices

    def get_hash_param_keys(self) -> List[str]:
        return [
            PARAM_PAST_TIMESTEPS,
            PARAM_FUTURE_TIMESTEPS,
            PARAM_MAX_AGENTS,
            "agent_types",
            "compute_maneuvers_online",
            "hybrid_smooth_mode",
            PARAM_PAST_TIMESTEP_SIZE,
            PARAM_FUTURE_TIMESTEP_SIZE,
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):
        """
        Label maneuvers given input trajectories.

        Parameters
        ----------
        result_dict:
            Input data.
        params:
            Parameters.
        filename:
            Filename of data example.
        index:
            Index of data example.

        Returns
        -------
        A dictionary containing labeled maneuvers.

        """
        maneuver_label_list = None
        if params["data_debug_mode"]:
            debug_data = params["data_debug_mode"](
                params,
                params[PARAM_PAST_TIMESTEPS] + params[PARAM_FUTURE_TIMESTEPS],
                params[PARAM_MAX_AGENTS],
                index,
                params["agent_types"],
                filename,
            )
            maneuver_label_list = debug_data["modes"]
            maneuver_label_list = np.array(maneuver_label_list)[:, 0]
            # NOTE: This assumes maneuver is only given for the first agent (i.e. Argoverse).
            maneuver_label_list = np.expand_dims(maneuver_label_list, axis=0)
        else:
            # Always compute maneuvers online.
            agent_positions = result_dict[ProtobufPredictionDataset.DATASET_KEY_POSITIONS]
            num_agents = agent_positions.shape[0]
            maneuver_label_list = []
            maneuver_map = MODE_DICT_MANEUVER_TO_IDX_ONLINE

            for i in range(num_agents):
                maneuver_label = label_driving_mode_sequence(agent_positions[i, :, :2])
                maneuver_label = [maneuver_map[maneuver_label[i]] for i in range(len(maneuver_label))]
                maneuver_label_list.append(maneuver_label)
            maneuver_label_list = np.array(maneuver_label_list)

            # Reduce label size if necessary.
            if params["discrete_label_size"] == 5:
                maneuver_label_list[maneuver_label_list > 4] = 4
            elif params["discrete_label_size"] == 4:
                maneuver_label_list[maneuver_label_list > 3] = 3

        # Remove noise in the mode label.
        if params["hybrid_smooth_mode"]:
            maneuver_label_list = hybrid_label_smooth_filter(maneuver_label_list)

        return {ProtobufPredictionDataset.DATASET_KEY_MANEUVERS: maneuver_label_list}
