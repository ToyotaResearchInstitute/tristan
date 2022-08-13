import copy
import json
import os
import pickle
import subprocess
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Hashable, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from intent.multiagents.logging_handlers_utils import (
    MISSES_FULL_LIST,
    MISSES_PARTIAL,
    MISSES_PARTIAL_LIST,
    MISSES_STAT_NAME,
    plot_statistics_plots,
)
from intent.multiagents.trainer_visualization import plot_roc_curve, visualize_histogram, visualize_prediction
from loaders.ado_key_names import AGENT_TYPE_PEDESTRIAN
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_semantic_handler import (
    SEMANTIC_HANDLER_TYPE_IDX,
    SEMANTIC_HANDLER_VALID_IDX,
    SEMANTIC_HANDLER_VALUE_IDX,
)
from util.prediction_metrics import calculate_traversal_error, displacement_errors


def compute_valid_test_instances(batch_itm: dict, time_before: float = 5.0, time_after: float = 2.0) -> Optional[list]:
    """Compute which instances are pivot examples

    Parameters
    ----------
    batch_itm : dict
        A batch items dictionary from the dataloader/trainer.
    time_before : float
        The time before the pivot
    time_after : float
        The time after the pivot

    Returns
    -------
    valid_instances
        A list of whether the example is close to a pivot, or None otherwise.
    """
    batch_size = batch_itm["timestamps"].shape[0]
    if ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO in batch_itm:
        valid_instances = []
        for b in range(batch_size):
            instance_info = json.loads(batch_itm[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO][b])
            if "pivot_events_validation_only" in instance_info:
                key = list(instance_info["pivot_events_validation_only"].keys())[0]
                time_to_event = instance_info["pivot_events_validation_only"][key]["timestamp"]
                #
                if time_to_event >= -time_after and time_to_event < time_before:
                    valid_instances.append(True)
                else:
                    valid_instances.append(False)
            else:
                valid_instances.append(False)
        return valid_instances
    else:
        return None


def get_relevant_agent_idx(batch_itm, batch_idx):
    """
    Return the relevant agent index, None if there is no relevant agent.
    :param batch_itm:
    :param batch_idx:
    :param param:
    :return:
    """
    is_rel_ped = batch_itm[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT][batch_idx, :]
    relevant_idx = np.nonzero(is_rel_ped.cpu().numpy())
    if len(relevant_idx) == 0 or len(relevant_idx[0]) == 0:
        # No relevant pedestrian to speak of.
        return None
    else:
        return relevant_idx[0][0]


def filter_nonstraight_trajectories(batch_itm, batch_idx, param):
    """
    Returns true of this is a non-straight trajectory for the relevant agent.
    :param batch_itm:
    :param batch_idx:
    :param param:
    :return:
    """
    positions = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][batch_idx, :, :, :2].float()
    is_rel_ped = batch_itm[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT][batch_idx, :]
    timestamps = batch_itm[ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS][batch_idx, ...].cpu().numpy()
    relevant_idx = np.nonzero(is_rel_ped.cpu().numpy())
    if len(relevant_idx) == 0 or len(relevant_idx[0]) == 0:
        # No relevant pedestrian to speak of.
        return False
    relevant_idx = relevant_idx[0][0]
    degree = param["nonstraight_walking_degree"]
    relevant_trajectory = positions[relevant_idx, ...].cpu().numpy()
    relevant_trajectory = relevant_trajectory - relevant_trajectory.mean(0)
    total_sq_err = 0.0
    for d in range(2):
        _, residuals, *_ = np.polyfit(timestamps, relevant_trajectory[:, d], degree, full=True)
        total_sq_err += residuals[0]

    # Use relative squared error, with compensation for small trajectories (to ignore tracking noise level).
    noise_level = 2.0
    relative_error = total_sq_err / (np.sum(relevant_trajectory**2) + noise_level**2 * relevant_trajectory.shape[0])
    return relative_error > param["nonstraight_walking_threshold"]


def search_tlog_in_s3(tlog_recent_path, params):
    """Search for a tlog based on a partial postfix of its pathname, in the source_tlog_paths list. Searching is done
    via an AWS client and is not cached.

    Parameters
    ----------
    tlog_recent_path : str
      The tlog and possibly recent parents in its known path.
      e.g '1603243073.5359151/1603243073.5359151_pa_reprocess_dot_reprocess_pa_reprocess.tlog'
    params : dict
      Our parameter dictionary. Should include 'source_tlog_paths', a list of s3 prefixes to search in

    Returns
    -------
    result : dict
      A dictionary with 'full_filepath' of the tlog, 'full_folder_path' for the folder containing the tlog.
      None if no match was found.
    """
    for tlog_path in params["source_tlog_paths"]:
        cmd = ["aws", "s3", "ls", "--recursive", tlog_path]
        aws_ls_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        cut_process = subprocess.Popen(["grep", tlog_recent_path], stdin=aws_ls_process.stdout, stdout=subprocess.PIPE)
        aws_ls_process.stdout.close()
        for line in cut_process.stdout.readlines():
            if len(line) > 0:
                file_path = " ".join(line.decode().split(" ")[3:]).strip()
                folder_path, _ = os.path.split(file_path)
                full_filepath = os.path.join(tlog_path, file_path)
                full_folder_path = os.path.join(tlog_path, folder_path)
                result = {"full_filepath": full_filepath, "full_folder_path": full_folder_path}
                return result

    # Didn't find a match
    return None


class LoggingHandler(ABC):
    """Generic Logging Interface

    Parameters
    ----------
    params : dict
        The command line parameters dictionary.
    """

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params
        self.writer = None
        self.dataloader_type = None

    def initialize_training(self, writer):
        self.writer = writer

    @abstractmethod
    def epoch_start(self, dataloader_type: str) -> None:
        """Logger logic triggered at the beginning of an epoch.

        Parameters
        ----------
        dataloader_type : str
            The dataloader type.
        """
        self.dataloader_type = dataloader_type

    @abstractmethod
    def iteration_update(self, data_dictionary: dict, stats_dict: dict) -> None:
        """Logger logic triggered at each training iteration.

        Parameters
        ----------
        data_dictionary : dict
            A dictionary of data that the logging handler can use.
            This dictionary should contain the batch item and a subset of the following keys:
                additional_inputs -
                agent_additional_inputs -
                batch_agent_type -
                batch_itm - The batch item (dict) from the dataset.
                batch_cost - The cost criterion.
                batch_positions_tensor -
                dataloader_type - A string describing the dataloader type.
                expected_trajectories -
                input_tensor - Input torch.Tensor to the log aggregator.
                is_future_valid - A torch.Tensor - which of the observations are valid?
                is_valid -
                map_coordinates - A torch.Tensor of the positions of the map elements of shape (batch_size,
                    max_element_num, max_point_num, 2).
                map_others - A torch.Tensor with point type and tangent information for each point of shape (batch_size,
                    max_element_num, max_point_num, 3). The elements in the last 3 dimensions are structured as follows
                    (point type, sin(theta), cos(theta))), where point type is corresponding to the integer in
                    MapPointType.
                map_validity - A torch.Tensor of the positions of the map elements of shape (batch_size,
                    max_element_num, max_point_num).
                offset_x - The x offset of the predictions.
                offset_y - The y offset of the predictions.
                param - A dict of trainer params.
                predicted_semantics - Predicted semantics with model.
                predicted_trajectories - A torch.Tensor of predicted trajectories.
                relevant_agents -
                semantic_labels - Annotated semantic label (ground truth).
        stats_dict : dict
            A dictionary of statistics to be aggregated and logged.
            This dictionary will contain a subset of the following keys:
                g_stats - Generator statistics.
                num_future_points - Number of future timepoints.
        """

    @abstractmethod
    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool) -> None:
        """Logger logic triggered at the end of an epoch.

        Parameters
        ----------
        idx : int
            Current epoch id.

        global_batch_cnt : int
            Number of batches processed so far.

        skip_visualization: bool
            Flag to toggle running of visualization code.
        """


class SaveFDEHistogramPlots(LoggingHandler):
    """Logs histograms of the MoN (Min over N samples) FDE.
    Aggregates histograms over one epoch and creates a plot in the data runner folder.

    Parameters
    ----------
    params : dict
        The command line parameters dictionary.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.histogram_values = {}
        self.fdes = None
        self.img_ext = params["visualization_image_format"]

    def epoch_start(self, dataloader_type: str) -> None:
        """Logger logic triggered at the beginning of an epoch.

        Parameters
        ----------
        dataloader_type : str
            The dataloader type.
        """
        super().epoch_start(dataloader_type)
        self.fdes = []
        self.horizon_fdes = {}

    def iteration_update(self, data_dictionary: dict, stats_dict: dict) -> None:
        """Logger logic triggered at each training iteration.

        Parameters
        ----------
        data_dictionary : dict
            A dictionary of data that the logging handler can use.
            This logger requires the following keys be present:
            **No keys required from this dict**

        stats_dict : dict
            A dictionary of statistics to be aggregated and logged.
            This logger requires the following keys be present:
                g_stats
        """
        super().iteration_update(data_dictionary, stats_dict)

        # Discriminator updates do not evaluate the generator.
        if "g_stats" not in stats_dict or stats_dict["g_stats"] is None:
            return

        cur_fdes = stats_dict["g_stats"]["agent_mon_fde"].detach()
        self.fdes.append(cur_fdes[~cur_fdes.isnan()].cpu().numpy())

        for key in stats_dict["g_stats"]["agent_mon_fdes_partial"]:
            if key not in self.horizon_fdes:
                self.horizon_fdes[key] = []

            cur_fdes = stats_dict["g_stats"]["agent_mon_fdes_partial"][key].detach()
            self.horizon_fdes[key].append(cur_fdes[~cur_fdes.isnan()].cpu().numpy())

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool = False) -> None:
        """Creates the acutal histogram plot at the end of the epoch.

        Parameters
        ----------
        idx : int
            Current epoch id.

        global_batch_cnt : int
            Number of batches processed so far.
        """
        if len(self.fdes) == 0:
            raise ValueError("No FDE stats collected.")

        fdes_concatenated = np.concatenate(self.fdes)

        img = visualize_histogram(fdes_concatenated, "MoN FDE", image_format=self.img_ext, xlabel="FDE [m]")
        pil_image = Image.fromarray(img)
        output_folder = os.path.expanduser(os.path.expandvars(self.params["runner_output_folder"]))
        os.makedirs(output_folder, exist_ok=True)
        save_filename = os.path.join(output_folder, f"{self.dataloader_type}_histogram_mon_fde.{self.img_ext}")
        pil_image.save(save_filename)

        for key, horizon_fde in self.horizon_fdes.items():
            fdes_concatenated = np.concatenate(horizon_fde)
            if len(fdes_concatenated) == 0:
                continue
            img = visualize_histogram(fdes_concatenated, f"MoN FDE@{key}s", image_format=self.img_ext, xlabel="FDE [m]")
            pil_image = Image.fromarray(img)
            output_folder = os.path.expanduser(os.path.expandvars(self.params["runner_output_folder"]))
            save_filename = os.path.join(
                output_folder,
                f"{self.dataloader_type}_histogram_mon_fde_{key}.{self.img_ext}",
            )
            pil_image.save(save_filename)


class ImageStatsLogHandler(LoggingHandler):
    def __init__(self, params, input_name) -> None:
        super().__init__(params)
        self.input_name = input_name
        self.stats = {}
        self.WIDTH_KEY = "width"
        self.HEIGHT_KEY = "height"

    def initialize_training(self, writer):
        super().initialize_training(writer)

    def epoch_start(self, dataloader_type):
        super().epoch_start(dataloader_type)
        self.stats[self.WIDTH_KEY] = []
        self.stats[self.HEIGHT_KEY] = []

    def iteration_update(self, data_dictionary: dict, stats_dict: dict):
        """Per iteration update -- save statistics of the images into tensorboard

        data_dictionary : dict
            A dictionary of data that the logging handler can use.
            This logger requires the following keys be present:
                additional_inputs, agent_additional_inputs, batch_itm, expected_trajectories, input_tensor,
                is_future_valid, is_valid, predicted_trajectories,

        stats_dict : dict
            A dictionary of statistics to be aggregated and logged.
            This logger requires the following keys be present:
            **No keys required from this dict**
        """
        super().iteration_update(data_dictionary, stats_dict)
        additional_inputs = data_dictionary["additional_inputs"]
        agent_additional_inputs = data_dictionary["agent_additional_inputs"]
        batch_itm = data_dictionary["batch_itm"]

        data = None
        if self.input_name in agent_additional_inputs:
            data = agent_additional_inputs[self.input_name]
        elif self.input_name in additional_inputs:
            data = additional_inputs[self.input_name]

        if data is not None and ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_STATS in batch_itm:
            for itm in batch_itm[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_STATS]:
                stats = json.loads(itm)
                if len(stats["width"]) > 0:
                    self.stats[self.WIDTH_KEY].append(stats["width"][0])
                    self.stats[self.HEIGHT_KEY].append(stats["height"][0])

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool = False) -> None:
        """Logger logic triggered at the end of an epoch.

        Parameters
        ----------
        idx : int
            Current epoch id.

        global_batch_cnt : int
            Number of batches processed so far.
        """
        for key in [self.WIDTH_KEY, self.HEIGHT_KEY]:
            if len(self.stats[key]) > 0:
                if self.writer:
                    self.writer.add_text(
                        self.dataloader_type + "/stats/" + str(key) + "_" + self.input_name,
                        str(self.stats[key]),
                        global_batch_cnt,
                    )


class SaveErrorStatistics(LoggingHandler):
    VALID_ADES = "ades"
    VALID_FUTURE_POINTS = "valid_future_points"
    VALID_HORIZON_POINTS = "valid_horizon_points"
    VALID_FUTURE_TIMES = "valid_future_times"
    VALID_MAX_TIMES = "valid_max_times"
    VALID_FDES = "fdes"
    VALID_MON_FDE = "mon_fde_full"
    VALID_MON_ADE = "mon_ade_full"
    TRAVERSALS_TP = "traversal_TP"
    TRAVERSALS_TN = "traversal_TN"
    TRAVERSALS_FP = "traversal_FP"
    TRAVERSALS_FN = "traversal_FN"
    PREDICTED_DISTANCES_FROM_CROSSING = "predicted_distances_from_crossing"
    OBSERVED_DISTANCES_FROM_CROSSING = "observed_distances_from_crossing"
    TRAVERSALS = [
        TRAVERSALS_TP,
        TRAVERSALS_TN,
        TRAVERSALS_FP,
        TRAVERSALS_FN,
        PREDICTED_DISTANCES_FROM_CROSSING,
        OBSERVED_DISTANCES_FROM_CROSSING,
    ]
    # Saves only a subset of (horizon,threshold) pairs -- basically to address specific threshold per time horizon,
    # similar to Criterion 1.
    VALID_SPECIFIC_MISSES = "specific_misses"
    PEDESTRIANS = "pedestrians"
    ADOVEHICLE_FRAME = "adovehicle_frame"
    LABELS = "labels"

    def __init__(self, params, prefix, output_folder_name=None) -> None:
        """
        Save filenames for the statistics - e.g. ~/intent/saved_examples/prefix_20210122_123434.json
        :param params:
        :param prefix:
        """
        #

        super().__init__(params)

        # Make sure the input thresholds and corresponding time points have the same length.
        assert len(params["err_horizons_timepoints_x"]) == len(params["miss_thresholds_x"])
        assert len(params["err_horizons_timepoints_y"]) == len(params["miss_thresholds_y"])
        assert len(params["err_horizons_timepoints"]) == len(params["miss_thresholds"])
        self.err_timepoint_threshold_pairs = {
            "x": list(zip(params["err_horizons_timepoints_x"], range(len(params["miss_thresholds_x"])))),
            "y": list(zip(params["err_horizons_timepoints_y"], range(len(params["miss_thresholds_y"])))),
            "absolute": list(zip(params["err_horizons_timepoints"], range(len(params["miss_thresholds"])))),
        }
        self.prefix = prefix
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.report_unique_id = timestr

        # Create lists to store aggregate statistics.
        self.aggregate_statistics = {}

        for prefix_ in ["", self.PEDESTRIANS + "_", self.ADOVEHICLE_FRAME + "_"]:
            # Per horizon stats - pedestrians
            for miss_type in MISSES_FULL_LIST + MISSES_PARTIAL_LIST:
                self.aggregate_statistics[prefix_ + miss_type] = {}
            self.aggregate_statistics[prefix_ + self.VALID_ADES] = {}
            self.aggregate_statistics[prefix_ + self.VALID_FDES] = {}
            self.aggregate_statistics[prefix_ + self.VALID_HORIZON_POINTS] = {}
            self.aggregate_statistics[prefix_ + self.VALID_MON_FDE] = {}
            self.aggregate_statistics[prefix_ + self.VALID_MON_ADE] = {}
            for traversal in self.TRAVERSALS:
                self.aggregate_statistics[prefix_ + traversal] = {}
            self.aggregate_statistics[prefix_ + self.VALID_FUTURE_POINTS] = 0
            self.aggregate_statistics[prefix_ + self.VALID_FUTURE_TIMES] = 0.0
            self.aggregate_statistics[prefix_ + self.VALID_MAX_TIMES] = []

            for axis, timepoint_threshold_pairs in self.err_timepoint_threshold_pairs.items():
                self.aggregate_statistics[prefix_ + self.VALID_SPECIFIC_MISSES + "_" + axis] = {}
                self.aggregate_statistics[prefix_ + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"] = {}
                for time_point, _ in timepoint_threshold_pairs:
                    self.aggregate_statistics[prefix_ + self.VALID_SPECIFIC_MISSES + "_" + axis][time_point] = []
                    self.aggregate_statistics[prefix_ + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][
                        time_point
                    ] = []

        self.aggregate_statistics[self.LABELS] = {}
        self.aggregate_statistics["protobuf_filename"] = []

        self.aggregate_statistics["protobuf_filename"] = []

        self.vis_logger_stats = {}

        self.image_counter = 0

        self.img_ext = params["visualization_image_format"]

        self.output_folder_name = output_folder_name

        try:
            with open(self.params["latent_factors_file"]) as fp:
                self.label_definitions = json.load(fp)
            replace_ids = True
            if replace_ids:
                for i, itm in enumerate(self.label_definitions):
                    itm["id"] = i
        except IOError:
            import IPython

            IPython.embed(header="failed to read latent_factors_file")

    def initialize_training(self, writer):
        super().initialize_training(writer)

    def epoch_start(self, dataloader_type):
        super().epoch_start(dataloader_type)
        if self.output_folder_name is not None:
            self.output_folder_name = self.output_folder_name
        else:
            self.output_folder_name = os.path.expanduser(os.path.expandvars(self.params["runner_output_folder"]))
        os.makedirs(self.output_folder_name, exist_ok=True)

    def aggregate_metrics(
        self,
        m: int,
        prefix: str,
        displ_errors: dict,
        err_horizons_timepoints: list,
        err_timepoint_threshold_pairs,
        traversal_metrics: Optional[dict] = None,
    ):
        """Aggregate metrics from examples' displacement errors.

        Parameters
        ----------
        m : int
            The sample index (0,..)
        prefix : str
            The prefix of the metrics/scenario being collected (e.g. "pedestrians_")
        displ_errors : dict
            The displacement errors dictionary computed by displacement_errors.
        err_horizons_timepoints : list
            The list of timepoints at which to compute metrics.
        err_timepoint_threshold_pairs : Optional[dict], optional
            A dictionary mapping from "x" and "y" to the timepoints and threshold indices to be used for modified miss
            rates, by default {}
        """

        for key in err_horizons_timepoints:
            if key not in self.aggregate_statistics[prefix + self.VALID_FDES]:
                self.aggregate_statistics[prefix + self.VALID_FDES][key] = []
                self.aggregate_statistics[prefix + self.VALID_ADES][key] = []
                self.aggregate_statistics[prefix + self.VALID_MON_ADE][key] = []
                self.aggregate_statistics[prefix + self.VALID_MON_FDE][key] = []
                self.aggregate_statistics[prefix + self.VALID_HORIZON_POINTS][key] = 0.0
                for traversal in self.TRAVERSALS:
                    self.aggregate_statistics[prefix + traversal][key] = []
                for miss_type in MISSES_FULL_LIST + MISSES_PARTIAL_LIST:
                    self.aggregate_statistics[prefix + miss_type][key] = OrderedDict()

                for suffix in ["", "_x", "_y"]:
                    for threshold_idx in range(len(self.params["miss_thresholds" + suffix])):
                        for miss_type in [MISSES_STAT_NAME, MISSES_PARTIAL]:
                            self.aggregate_statistics[prefix + miss_type + suffix][key][threshold_idx] = OrderedDict()
                            self.aggregate_statistics[prefix + miss_type + suffix][key][threshold_idx]["sum"] = 0.0
                            self.aggregate_statistics[prefix + miss_type + suffix][key][threshold_idx]["count"] = 0.0

            valid_fde_list = displ_errors["agent_fdes_partial"][key].view(-1).tolist()
            valid_ade_list = displ_errors["agent_ades_partial"][key].view(-1).tolist()
            self.aggregate_statistics[prefix + self.VALID_HORIZON_POINTS][key] += (
                displ_errors["agent_fdes"][key].isnan().logical_not().sum().cpu().numpy()
            )

            for suffix in ["", "_x", "_y"]:
                for threshold_idx in range(len(self.params["miss_thresholds" + suffix])):
                    for miss_type in [MISSES_STAT_NAME, MISSES_PARTIAL]:
                        valid_misses = displ_errors[miss_type + suffix][key][..., threshold_idx][
                            displ_errors[miss_type + suffix][key][..., threshold_idx].isnan().logical_not()
                        ]
                        self.aggregate_statistics[prefix + miss_type + suffix][key][threshold_idx]["sum"] += (
                            valid_misses.sum().cpu().numpy()
                        )
                        self.aggregate_statistics[prefix + miss_type + suffix][key][threshold_idx][
                            "count"
                        ] += valid_misses.numel()

            try:
                valid_fde_list = [x for x in valid_fde_list if not np.isnan(x)]
                valid_ade_list = [x for x in valid_ade_list if not np.isnan(x)]

            except:
                import IPython

                IPython.embed(header="fdes")
            self.aggregate_statistics[prefix + self.VALID_FDES][key].extend(valid_fde_list)
            self.aggregate_statistics[prefix + self.VALID_ADES][key].extend(valid_ade_list)

            if ~displ_errors["ades"][key].isnan():
                if m == 0:
                    self.aggregate_statistics[prefix + self.VALID_MON_ADE][key].extend(
                        [displ_errors["ades"][key].detach().cpu().item()]
                    )
                    self.aggregate_statistics[prefix + self.VALID_MON_FDE][key].extend(
                        [displ_errors["fdes"][key].detach().cpu().item()]
                    )
                else:
                    self.aggregate_statistics[prefix + self.VALID_MON_ADE][key][-1] = min(
                        self.aggregate_statistics[prefix + self.VALID_MON_ADE][key][-1],
                        displ_errors["ades"][key].cpu().detach().item(),
                    )
                    self.aggregate_statistics[prefix + self.VALID_MON_FDE][key][-1] = min(
                        self.aggregate_statistics[prefix + self.VALID_MON_FDE][key][-1],
                        displ_errors["fdes"][key].cpu().detach().item(),
                    )

            if traversal_metrics:
                invalid_agents = traversal_metrics["invalid_agents_full"][key]
                for traversal in self.TRAVERSALS:
                    # Get and store metrics for the relevant agent.
                    metrics = traversal_metrics[traversal][key].detach().cpu()
                    # Only include metrics for samples where we see enough ground truth.
                    valid_metrics = metrics[invalid_agents.logical_not()].tolist()
                    num_valid_metrics = len(valid_metrics)
                    if num_valid_metrics > 0:
                        if m == 0 or not self.params["should_mon_traversals_stats"]:
                            # Add new valid_metrics to the array -- first sample
                            self.aggregate_statistics[prefix + traversal][key].extend(valid_metrics)
                        else:
                            current_sample_list = self.aggregate_statistics[prefix + traversal][key][
                                -num_valid_metrics:
                            ]
                            MIN_UPDATE_TRAVERSALS = ["traversal_FP", "traversal_FN"]
                            MAX_UPDATE_TRAVERSALS = ["traversal_TP", "traversal_TN"]

                            if traversal in MIN_UPDATE_TRAVERSALS:
                                updated_sample_list = np.logical_and(
                                    np.array(valid_metrics), np.array(current_sample_list)
                                ).tolist()
                            elif traversal in MAX_UPDATE_TRAVERSALS:
                                updated_sample_list = np.logical_or(
                                    np.array(valid_metrics), np.array(current_sample_list)
                                ).tolist()
                            else:
                                raise ValueError(f"traversal type unknown: {traversal}")
                            self.aggregate_statistics[prefix + traversal][key][
                                -num_valid_metrics:
                            ] = updated_sample_list

        for axis in err_timepoint_threshold_pairs:
            timepoint_threshold_pairs = err_timepoint_threshold_pairs[axis]
            for time_point_index, threshold_index in timepoint_threshold_pairs:
                if axis != "absolute":
                    relevant_samples = displ_errors["misses_partial_" + axis][time_point_index][:, 1, threshold_index]
                else:
                    relevant_samples = displ_errors["misses_partial"][time_point_index][:, 1, threshold_index]
                valid_misses = relevant_samples[relevant_samples.isnan().logical_not()]
                self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis][time_point_index].extend(
                    valid_misses.cpu().detach().tolist()
                )
                min_per_batch = relevant_samples.cpu().detach().tolist()
                for idx, batch_min in enumerate(min_per_batch):
                    if np.isnan(batch_min):
                        val = 1.0
                    else:
                        val = batch_min
                    idx_from_end = len(min_per_batch) - idx
                    if m == 0:
                        self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][
                            time_point_index
                        ].append(val)
                    else:
                        self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][
                            time_point_index
                        ][-idx_from_end] = min(
                            self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][
                                time_point_index
                            ][-idx_from_end],
                            val,
                        )

    def make_json_compatible(self, aggregate_statistics):
        """
        Make the dictionary compatible w/ jsons by replacing numpy arrays with lists.
        :param aggregate_statistics: The statistics dictionary to be converted.
        :return:
        """
        result = copy.deepcopy(aggregate_statistics)
        return result

    def aggregate_semantic_labels(self, semantic_labels, predicted_semantics):

        if semantic_labels is not None:
            batch_size, num_semantic_labels, _ = semantic_labels.shape
            for b in range(batch_size):
                for i in range(num_semantic_labels):
                    if not semantic_labels[b, i, SEMANTIC_HANDLER_VALID_IDX]:
                        continue
                    label_type = semantic_labels[b, i, SEMANTIC_HANDLER_TYPE_IDX].cpu().item()

                    for label_definition in self.label_definitions:
                        if semantic_labels[b, i, SEMANTIC_HANDLER_TYPE_IDX] == label_definition["id"]:
                            is_id_matched = True
                            break
                    if not is_id_matched:
                        import IPython

                        IPython.embed(header="failed to match label ID")
                    if semantic_labels[b, i, SEMANTIC_HANDLER_VALID_IDX] > 1e-6:
                        gt_label = bool(semantic_labels[b, i, SEMANTIC_HANDLER_VALUE_IDX].detach().cpu().item() > 0)
                    else:
                        gt_label = "Invalid"
                    predicted_label = bool(
                        predicted_semantics[b, i, SEMANTIC_HANDLER_VALUE_IDX].detach().cpu().item() > 0.5
                    )
                    if gt_label != "Invalid":
                        if label_type not in self.aggregate_statistics[self.LABELS]:
                            self.aggregate_statistics[self.LABELS][label_type] = {"predictions": [], "targets": []}

                        try:
                            self.aggregate_statistics[self.LABELS][label_type]["predictions"].append(predicted_label)
                            self.aggregate_statistics[self.LABELS][label_type]["targets"].append(gt_label)
                        except:
                            import IPython

                            IPython.embed(header="check")

    @staticmethod
    def create_rotations_ego_scene(ego_mask, rotations_scene_local):
        """Creates rotation transforms that rotate a scene-frame trajectory into an ego-frame trajectory"""
        assert (ego_mask.sum(1) == 1).all()
        rotations_scene_ego = torch.masked_select(
            rotations_scene_local, ego_mask.bool().unsqueeze(-1).unsqueeze(-1)
        ).view(-1, 2, 2)
        rotations_ego_scene = rotations_scene_ego.transpose(-2, -1)
        return rotations_ego_scene

    def iteration_update(self, data_dictionary: dict, stats_dict: dict):
        """Per iteration update -- save statistics of the images into tensorboard

        data_dictionary : dict
            A dictionary of data that the logging handler can use.
            This logger requires the following keys be present:
                additional_inputs, agent_additional_inputs, batch_agent_type, batch_itm, batch_positions_tensor,
                expected_trajectories, input_tensor, is_future_valid, is_valid, param, predicted_semantics,
                prediction_timestamp, predicted_trajectories, relevant_agents, semantic_labels
        stats_dict : dict
            A dictionary of statistics to be aggregated and logged.
            This logger requires the following keys be present:
            **No keys required from this dict**
        """
        super().iteration_update(data_dictionary, stats_dict)
        predicted_trajectories_scene = data_dictionary["predicted_trajectories"]
        expected_trajectories_scene = data_dictionary["expected_trajectories"]
        is_future_valid = data_dictionary["is_future_valid"]
        batch_itm = data_dictionary["batch_itm"]
        batch_agent_type = data_dictionary["batch_agent_type"]
        relevant_agents = data_dictionary["relevant_agents"]
        prediction_timestamp = batch_itm["prediction_timestamp"]
        semantic_labels = data_dictionary["semantic_labels"]
        predicted_semantics = data_dictionary["predicted_semantics"]

        ego_mask = batch_itm["is_ego_vehicle"]
        param = data_dictionary["param"]
        err_horizons_timepoints = param["err_horizons_timepoints"]
        # Aggregate from every batch statistics about the horizon
        num_future_points = is_future_valid.shape[2]

        # Transforms are generated by PredictionModel.compute_normalizing_transforms
        transforms_local_scene = data_dictionary["agent_transforms"]
        rotations_scene_local = transforms_local_scene[:, :, :2, :].transpose(2, 3)

        relevant_agents_types = [AGENT_TYPE_PEDESTRIAN]

        future_timepoints = batch_itm[ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS][:, -num_future_points:]
        future_horizons = future_timepoints - batch_itm[
            ProtobufPredictionDataset.DATASET_KEY_PREDICTION_TIMESTAMP
        ].unsqueeze(1)

        self.aggregate_statistics[self.VALID_FUTURE_POINTS] += is_future_valid.sum().cpu().detach().item()
        self.aggregate_statistics[self.VALID_FUTURE_TIMES] += (
            (future_horizons.unsqueeze(1) * is_future_valid.float()).sum().cpu().detach().item()
        )
        valid_times = (future_horizons.unsqueeze(1) * is_future_valid.float()).cpu().detach()
        valid_max, _ = valid_times.max(2)
        self.aggregate_statistics[self.VALID_MAX_TIMES].extend(valid_max.tolist())
        assert predicted_trajectories_scene.shape[-1] > 0

        if param["runner_test_vehicle_filter"]:

            # TODO(guy.rosman): Di.Sun: save results for 3PJ table, based on predicted_trajectories_scene, expected
            # TODO trajectories, etc..
            # TODO(guy.rosman): verify correctness: filter out the predicted, expected trajectory, visibility, etc..
            valid_instances = compute_valid_test_instances(batch_itm)
            try:
                predicted_trajectories_scene = predicted_trajectories_scene[valid_instances, ...]
                expected_trajectories_scene = expected_trajectories_scene[valid_instances, ...]
                future_timepoints = future_timepoints[valid_instances, ...]
                relevant_agents = relevant_agents[valid_instances, ...]
                prediction_timestamp = prediction_timestamp[valid_instances, ...]
                is_future_valid = is_future_valid[valid_instances, ...]
                batch_agent_type = batch_agent_type[valid_instances, ...]
                rotations_scene_local = rotations_scene_local[valid_instances, ...]
                ego_mask = ego_mask[valid_instances, ...]
            except:
                import IPython

                IPython.embed()

        rotations_ego_scene = self.create_rotations_ego_scene(ego_mask, rotations_scene_local)

        for m in range(predicted_trajectories_scene.shape[-1]):
            # Compute displacement errors in the egovehicle frame -
            # see displacement_errors for the full list.
            displ_errors_egoframe = displacement_errors(
                predicted_trajectories_scene[..., m],
                expected_trajectories_scene,
                future_timepoints,
                prediction_timestamp=prediction_timestamp,
                relevant_agents=relevant_agents,
                visibility=is_future_valid,
                horizons=err_horizons_timepoints,
                miss_thresholds=param["miss_thresholds"],
                param=param,
                agent_types=batch_agent_type,
                rotations_local_scene=rotations_ego_scene,
            )
            # Compute displacement errors in scene frame.
            displ_errors = displacement_errors(
                predicted_trajectories_scene[..., m],
                expected_trajectories_scene,
                future_timepoints,
                prediction_timestamp=prediction_timestamp,
                relevant_agents=relevant_agents,
                visibility=is_future_valid,
                horizons=err_horizons_timepoints,
                miss_thresholds=param["miss_thresholds"],
                param=param,
                agent_types=batch_agent_type,
                rotations_local_scene=None,
                miss_thresholds_x=param["miss_thresholds_x"],
                miss_thresholds_y=param["miss_thresholds_y"],
            )
            # Compute displacement errors for pedestrians only in egovehicle frame.
            displ_errors_pedestrians_egoframe = displacement_errors(
                predicted_trajectories_scene[..., m],
                expected_trajectories_scene,
                future_timepoints,
                prediction_timestamp=prediction_timestamp,
                relevant_agents=relevant_agents,
                visibility=is_future_valid,
                horizons=err_horizons_timepoints,
                miss_thresholds=param["miss_thresholds"],
                param=param,
                agent_types=batch_agent_type,
                relevant_agent_types=relevant_agents_types,
                rotations_local_scene=rotations_ego_scene,
                miss_thresholds_x=param["miss_thresholds_x"],
                miss_thresholds_y=param["miss_thresholds_y"],
            )

            traversal_errors = calculate_traversal_error(
                predicted_trajectories_scene[..., m],
                expected_trajectories_scene,
                ego_mask,
                rotations_scene_ego=torch.linalg.inv(rotations_ego_scene),
                timestamps=future_timepoints,
                prediction_timestamp=prediction_timestamp,
                relevant_agents=relevant_agents,
                visibility=is_future_valid,
                horizons=err_horizons_timepoints,
                param=param,
            )

            pedestrian_prefix = self.PEDESTRIANS + "_"
            ado_frame_prefix = self.ADOVEHICLE_FRAME + "_"
            self.aggregate_metrics(
                m,
                "",
                displ_errors_egoframe,
                err_horizons_timepoints,
                self.err_timepoint_threshold_pairs,
                traversal_metrics=traversal_errors,
            )
            self.aggregate_metrics(
                m,
                pedestrian_prefix,
                displ_errors_pedestrians_egoframe,
                err_horizons_timepoints,
                self.err_timepoint_threshold_pairs,
            )
            # Store results in adovehicle's frame, e.g. for miss rates (for Waymo-style metrics)
            self.aggregate_metrics(
                m, ado_frame_prefix, displ_errors, err_horizons_timepoints, self.err_timepoint_threshold_pairs
            )
            self.aggregate_semantic_labels(semantic_labels, predicted_semantics)
            self.aggregate_statistics["protobuf_filename"].extend(batch_itm["protobuf_file"])

            self.aggregate_statistics["protobuf_filename"].extend(batch_itm["protobuf_file"])

        # Save specific miss rates to dictionary for external plotting.
        for prefix in ["", self.PEDESTRIANS + "_"]:
            # Specific miss rate/threshold pairs of interest to test vehicle only visualized here.
            for axis in self.err_timepoint_threshold_pairs:
                for key in self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis]:
                    if axis == "absolute":
                        suffix = ""
                    else:
                        suffix = "_" + axis
                    self.vis_logger_stats[
                        prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_{}".format(str(key))
                    ] = np.mean(self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis][key])
                    self.vis_logger_stats[
                        prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN_{}".format(str(key))
                    ] = np.mean(
                        self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][key]
                    )

        # TODO(guy.rosman): Add more statistics.

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool = False) -> None:
        """Logger logic triggered at the end of an epoch.

        Parameters
        ----------
        idx : int
            Current epoch id.

        global_batch_cnt : int
            Number of batches processed so far.
        """
        # Create a statistics dictionary
        stats = {}
        stats["max_horizon"] = np.max(self.aggregate_statistics[self.VALID_MAX_TIMES])
        stats["avg_horizon"] = np.mean(self.aggregate_statistics[self.VALID_MAX_TIMES])
        stats["err_horizons_timepoints"] = self.params["err_horizons_timepoints"]
        stats["err_horizons_timepoints_x"] = self.params["err_horizons_timepoints_x"]
        stats["err_horizons_timepoints_y"] = self.params["err_horizons_timepoints_y"]
        stats["miss_thresholds"] = self.params["miss_thresholds"]
        stats["miss_thresholds_x"] = self.params["miss_thresholds_x"]
        stats["miss_thresholds_y"] = self.params["miss_thresholds_y"]

        fde_histogram_stats = {}  # Stats that should be rendered as histograms

        plot_stats = {
            self.PEDESTRIANS + "_" + "fde": OrderedDict(),
            self.PEDESTRIANS + "_" + "ade": OrderedDict(),
            "fde": OrderedDict(),
            "ade": OrderedDict(),
            self.PEDESTRIANS + "_mon_" + "fde": OrderedDict(),
            self.PEDESTRIANS + "_mon_" + "ade": OrderedDict(),
            "mon_fde": OrderedDict(),
            "mon_ade": OrderedDict(),
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_absolute": OrderedDict(),
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_x": OrderedDict(),
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_y": OrderedDict(),
            self.VALID_SPECIFIC_MISSES + "_absolute": OrderedDict(),
            self.VALID_SPECIFIC_MISSES + "_x": OrderedDict(),
            self.VALID_SPECIFIC_MISSES + "_y": OrderedDict(),
            self.VALID_SPECIFIC_MISSES + "_absolute_MoN": OrderedDict(),
            self.VALID_SPECIFIC_MISSES + "_x_MoN": OrderedDict(),
            self.VALID_SPECIFIC_MISSES + "_y_MoN": OrderedDict(),
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_absolute_MoN": OrderedDict(),
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_x_MoN": OrderedDict(),
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_y_MoN": OrderedDict(),
        }

        plot_labels = {
            self.PEDESTRIANS + "_" + "fde": "Pedestrians FDE",
            self.PEDESTRIANS + "_" + "ade": "Pedestrians ADE",
            "fde": "FDE",
            "ade": "ADE",
            self.PEDESTRIANS + "_mon_" + "fde": "Pedestrians MoN FDE",
            self.PEDESTRIANS + "_mon_" + "ade": "Pedestrians MoN ADE",
            "mon_fde": "MoN FDE",
            "mon_ade": "MoN ADE",
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_absolute": "Pedestrians Absolute Specific Misses",
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_x": "Pedestrians Specific Misses in X",
            self.PEDESTRIANS + "_" + self.VALID_SPECIFIC_MISSES + "_y": "Pedestrians Specific Misses in Y",
            self.VALID_SPECIFIC_MISSES + "_absolute": "Absolute Specific Misses",
            self.VALID_SPECIFIC_MISSES + "_x": "Specific Misses in X",
            self.VALID_SPECIFIC_MISSES + "_y": "Specific Misses in Y",
            self.VALID_SPECIFIC_MISSES + "_absolute_MoN": "Best M of N Absolute Specific Misses",
            self.VALID_SPECIFIC_MISSES + "_x_MoN": "Best M of N Specific Misses in X",
            self.VALID_SPECIFIC_MISSES + "_y_MoN": "Best M of N Specific Misses in Y",
            self.PEDESTRIANS
            + "_"
            + self.VALID_SPECIFIC_MISSES
            + "_absolute_MoN": "Best M of N Pedestrian Absolute Specific Misses",
            self.PEDESTRIANS
            + "_"
            + self.VALID_SPECIFIC_MISSES
            + "_x_MoN": "Best M of N Pedestrian Specific Misses in X",
            self.PEDESTRIANS
            + "_"
            + self.VALID_SPECIFIC_MISSES
            + "_y_MoN": "Best M of N Pedestrian Specific Misses in Y",
        }

        for key in self.aggregate_statistics[self.VALID_FDES]:
            stats["fde_{}".format(key)] = np.mean(self.aggregate_statistics[self.VALID_FDES][key])
            fde_histogram_stats["fde_{}".format(key)] = np.array(self.aggregate_statistics[self.VALID_FDES][key])
            plot_stats["fde"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.VALID_FDES][key]),
                "std": np.std(self.aggregate_statistics[self.VALID_FDES][key]),
            }
            stats["ade_{}".format(key)] = np.mean(self.aggregate_statistics[self.VALID_ADES][key])
            plot_stats["ade"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.VALID_ADES][key]),
                "std": np.std(self.aggregate_statistics[self.VALID_ADES][key]),
            }

            stats[f"{self.PEDESTRIANS}_fde_{key}"] = np.mean(
                self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_FDES][key]
            )
            plot_stats[self.PEDESTRIANS + "_" + "fde"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_FDES][key]),
                "std": np.std(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_FDES][key]),
            }
            stats[f"{self.PEDESTRIANS}_ade_{key}"] = np.mean(
                self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_ADES][key]
            )
            plot_stats[self.PEDESTRIANS + "_" + "ade"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_ADES][key]),
                "std": np.std(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_ADES][key]),
            }

            # MoN Stats
            stats["mon_fde_{}".format(key)] = np.mean(self.aggregate_statistics[self.VALID_MON_FDE][key])
            plot_stats["mon_fde"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.VALID_MON_FDE][key]),
                "std": np.std(self.aggregate_statistics[self.VALID_MON_FDE][key]),
            }
            stats["mon_ade_{}".format(key)] = np.mean(self.aggregate_statistics[self.VALID_MON_ADE][key])
            plot_stats["mon_ade"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.VALID_MON_ADE][key]),
                "std": np.std(self.aggregate_statistics[self.VALID_MON_ADE][key]),
            }

            fde_histogram_stats[f"{self.PEDESTRIANS}_fde_{key}"] = np.array(
                self.aggregate_statistics[f"{self.PEDESTRIANS}_{self.VALID_FDES}"][key]
            )
            stats[f"{self.PEDESTRIANS}_mon_fde_{key}"] = np.mean(
                self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_MON_FDE][key]
            )
            plot_stats[self.PEDESTRIANS + "_mon_fde"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_MON_FDE][key]),
                "std": np.std(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_MON_FDE][key]),
            }
            stats[f"{self.PEDESTRIANS}_mon_ade_{key}"] = np.mean(
                self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_MON_ADE][key]
            )
            plot_stats[self.PEDESTRIANS + "_mon_ade"][key] = {
                "mean": np.mean(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_MON_ADE][key]),
                "std": np.std(self.aggregate_statistics[self.PEDESTRIANS + "_" + self.VALID_MON_ADE][key]),
            }

        for prefix in ["", self.PEDESTRIANS + "_"]:
            # Specific miss rate/threshold pairs of interest to test vehicle only visualized here.
            for axis in self.err_timepoint_threshold_pairs:
                for key in self.aggregate_statistics[self.VALID_SPECIFIC_MISSES + "_" + axis]:
                    if axis == "absolute":
                        suffix = ""
                    else:
                        suffix = "_" + axis
                    stats[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_{}".format(str(key))] = np.mean(
                        self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis][key]
                    )
                    stats[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN_{}".format(str(key))] = np.mean(
                        self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][key]
                    )
                    plot_stats[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis][key] = {
                        "mean": np.mean(
                            self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis][key]
                        ),
                        "std": np.std(self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis][key]),
                    }
                    plot_stats[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][key] = {
                        "mean": np.mean(
                            self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][key]
                        ),
                        "std": np.std(
                            self.aggregate_statistics[prefix + self.VALID_SPECIFIC_MISSES + "_" + axis + "_MoN"][key]
                        ),
                    }
            # All partial and regular miss rates added here.
            for miss_type in MISSES_FULL_LIST + MISSES_PARTIAL_LIST:
                for key in self.aggregate_statistics[miss_type]:
                    for threshold in self.aggregate_statistics[miss_type][key]:
                        for suffix in ["sum", "count"]:
                            stats[f"{prefix}{miss_type}_{key}_{threshold}_{suffix}"] = np.mean(
                                self.aggregate_statistics[prefix + miss_type][key][threshold][suffix]
                            )
        for horizon in self.aggregate_statistics[self.TRAVERSALS_FP]:
            fp = np.count_nonzero(self.aggregate_statistics[self.TRAVERSALS_FP][horizon])
            tp = np.count_nonzero(self.aggregate_statistics[self.TRAVERSALS_TP][horizon])
            fn = np.count_nonzero(self.aggregate_statistics[self.TRAVERSALS_FN][horizon])
            tn = np.count_nonzero(self.aggregate_statistics[self.TRAVERSALS_TN][horizon])
            total_examples = len(self.aggregate_statistics[self.TRAVERSALS_TP][horizon])
            assert fp + tp + fn + tn == total_examples
            if total_examples == 0:
                print(f"####### Horizon {horizon} has 0 total crossing examples!!")
                continue
            stats["crossing_precision_" + str(horizon)] = np.nan if (tp + fn) == 0 else tp / (tp + fn)
            stats["crossing_tnr_" + str(horizon)] = np.nan if (tn + fp) == 0 else tn / (tn + fp)
            stats["crossing_fnr_" + str(horizon)] = np.nan if (fn + tp) == 0 else fn / (fn + tp)
            stats["crossing_accuracy_" + str(horizon)] = (tp + tn) / total_examples

        stats["plot_labels"] = plot_labels
        stats["plot_stats"] = plot_stats
        stats["prefix"] = self.prefix
        stats["report_unique_id"] = self.report_unique_id

        stats["aggregate_statistics"] = self.make_json_compatible(self.aggregate_statistics)

        valid_point_list = [
            self.aggregate_statistics[self.VALID_HORIZON_POINTS][k] for k in self.params["err_horizons_timepoints"]
        ]

        stats["valid_point_list"] = valid_point_list

        plot_statistics_plots(stats, self.output_folder_name, self.img_ext)
        for key, values in fde_histogram_stats.items():
            if not any(values):
                print(f"Skipping {key} visualization due to zero data")
                continue
            img = visualize_histogram(values, key, image_format=self.img_ext, xlabel="FDE [m]")
            pil_image = Image.fromarray(img)
            save_filename = os.path.join(
                self.output_folder_name,
                f"{self.prefix}_{self.report_unique_id}_histogram_{key}.{self.img_ext}",
            )
            print(f"Saving {key} histogram to {save_filename}")
            pil_image.save(save_filename)

        # Save the statistics dictionary
        json_filename = os.path.join(self.output_folder_name, self.prefix + "_" + self.report_unique_id + ".json")
        print("Saving statistics to {}.".format(json_filename))

        with open(json_filename, "w") as fp:
            json.dump(stats, fp, indent=2)

        for horizon in self.aggregate_statistics[self.PREDICTED_DISTANCES_FROM_CROSSING]:
            predicted_distances_from_crossing = self.aggregate_statistics[self.PREDICTED_DISTANCES_FROM_CROSSING][
                horizon
            ]
            observed_distances_from_crossing = self.aggregate_statistics[self.OBSERVED_DISTANCES_FROM_CROSSING][horizon]

            # print ROC curve
            y_true = (torch.tensor([d for d in observed_distances_from_crossing if not np.isnan(d)]) < 1e-5).numpy()
            # Score is the inverse of the distance, as closer is better
            y_score = -1 * torch.tensor([d for d in predicted_distances_from_crossing if not np.isnan(d)]).numpy()
            if len(y_score) > 0:
                img = plot_roc_curve(y_true, y_score, self.img_ext)
                pil_image = Image.fromarray(img)
                save_filename = os.path.join(
                    self.output_folder_name,
                    f"{self.prefix}_{self.report_unique_id}_roc_curve_{horizon}.{self.img_ext}",
                )
                pil_image.save(save_filename)
            else:
                print("Insufficient data to render roc curve")


class SaveExamplesLogHandler(LoggingHandler):
    def __init__(self, params, prefix, output_folder_name=None) -> None:
        super().__init__(params)
        # Save filenames for the statistics - e.g. ~/intent/saved_examples/prefix_20210122_123434.json
        self.prefix = prefix
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.report_unique_id = timestr
        if output_folder_name is not None:
            self.output_folder_name = output_folder_name
        else:
            self.output_folder_name = None
        self.img_ext = params["visualization_image_format"]

        self.image_counter = 0

    def initialize_training(self, writer):
        super().initialize_training(writer)

    def epoch_start(self, dataloader_type):
        super().epoch_start(dataloader_type)
        # The runner_output_folder param is not correctly set at class init, wait until here to read it.
        if self.output_folder_name is None:
            self.output_folder_name = os.path.expanduser(os.path.expandvars(self.params["runner_output_folder"]))
        os.makedirs(self.output_folder_name, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder_name, "all"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder_name, "nonstraight"), exist_ok=True)

    def iteration_update(self, data_dictionary: dict, stats_dict: dict):
        """Per iteration update -- save visualization images.

        data_dictionary : dict
            A dictionary of data that the logging handler can use.
            This logger requires the following keys be present:
                batch_itm, is_future_valid, predicted_trajectories, map_coordinates, map_others, map_validity, offset_x,
                 offset_y, param
        stats_dict : dict
            A dictionary of statistics to be aggregated and logged.
            This logger requires the following keys be present:
            **No keys required from this dict**
        """
        super().iteration_update(data_dictionary, stats_dict)
        predicted_trajectories_scene = data_dictionary["predicted_trajectories"]
        is_future_valid = data_dictionary["is_future_valid"]
        batch_itm = data_dictionary["batch_itm"]
        offset_x = data_dictionary["offset_x"]
        offset_y = data_dictionary["offset_y"]
        map_coordinates = data_dictionary["map_coordinates"]
        map_validity = data_dictionary["map_validity"]
        map_others = data_dictionary["map_others"]
        param = data_dictionary["param"]
        # Aggregate from every batch statistics about the horizon
        batch_size = is_future_valid.shape[0]

        scale = self.params["predictor_normalization_scale"]
        rotations_local_scene = data_dictionary["agent_transforms"][:, :, :2, :].transpose(2, 3)
        if param["runner_test_vehicle_filter"]:
            valid_instances = compute_valid_test_instances(batch_itm)
        else:
            valid_instances = None
        for b in range(batch_size):
            solution = {
                "predicted_trajectories": predicted_trajectories_scene[b, :, :, :, :],
                "is_future_valid": is_future_valid[b, :, :],
                "ego_rotation": torch.masked_select(
                    rotations_local_scene, batch_itm["is_ego_vehicle"].bool().unsqueeze(-1).unsqueeze(-1)
                ).view(-1, 2, 2)[b, ...],
            }
            if (valid_instances is not None) and not valid_instances[b]:
                # this example is not a valid instance according to the test vehicle filter, do not save.
                continue

            # Plot the visualization of the predictions, save each visualization
            img, additional_info_dictionary = visualize_prediction(
                batch_itm,
                b,
                solution,
                scale=scale,
                means=(offset_x[b, :], offset_y[b, :]),
                cost=None,
                param=param,
                map_coordinates=map_coordinates,
                map_validity=map_validity,
                map_others=map_others,
            )
            im = Image.fromarray(img)

            save_filename = os.path.join(
                self.output_folder_name,
                "all",
                f"{self.prefix}_{self.dataloader_type}_{self.image_counter}.{self.img_ext}",
            )
            im.save(save_filename)

            if self.params["save_cases_jsons"]:
                save_json_filename = os.path.join(
                    self.output_folder_name,
                    "all",
                    f"{self.prefix}_{self.dataloader_type}_{self.image_counter}.json",
                )
                tracks = additional_info_dictionary["tracks"]
                with open(save_json_filename, "w") as fp:
                    json.dump(tracks, fp, indent=2)

            # Plot a visualization of only the pedestrian, save
            pedestrian_idx = get_relevant_agent_idx(batch_itm, b)
            if pedestrian_idx is not None:
                param2 = copy.copy(param)
                param2.update({"map_view_margin_around_trajectories": 10})
                img_pedestrian, additional_info_dictionary = visualize_prediction(
                    batch_itm,
                    b,
                    solution,
                    scale=scale,
                    means=(offset_x[b, :], offset_y[b, :]),
                    cost=None,
                    param=param2,
                    map_coordinates=map_coordinates,
                    map_validity=map_validity,
                    map_others=map_others,
                    agent_set=[pedestrian_idx],
                )
                im_pedestrian = Image.fromarray(img_pedestrian)
                save_filename = os.path.join(
                    self.output_folder_name,
                    "all",
                    f"{self.prefix}_{self.dataloader_type}_{self.image_counter}_pedestrian.{self.img_ext}",
                )
                im_pedestrian.save(save_filename)

                if self.params["save_cases_for_table"]:
                    proto_file_name = batch_itm[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE][b]
                    proto_dir, proto_base_name = os.path.split(proto_file_name)
                    proto_number_str = os.path.splitext(proto_base_name)[0]
                    proto_tlog_name = os.path.basename(os.path.normpath(proto_dir))
                    example_output_folder = os.path.expanduser("~/intent/artifacts/table_json/")  # default value
                    if self.params["table_output_folder_name"]:
                        example_output_folder = os.path.expanduser(self.params["table_output_folder_name"])
                    os.makedirs(os.path.join(example_output_folder, proto_tlog_name), exist_ok=True)
                    proto_name = proto_number_str
                    if self.params["use_linear_model"]:
                        proto_name += "_lin"
                    print(f"proto_tlog_name {proto_tlog_name}, proto_name {proto_name}.json")
                    save_json_filename = os.path.join(
                        example_output_folder,
                        proto_tlog_name,
                        proto_name + ".json",
                    )
                    tracks = additional_info_dictionary["tracks"]
                    # "File name number == 2 * time_to_event" is a special requirement for the tabled data.
                    tracks["time_to_pivot_event"] = int(proto_number_str) * 0.5
                    tracks["file_name"] = proto_file_name
                    try:
                        with open(save_json_filename, "w") as fp:
                            json.dump(tracks, fp, indent=2)
                    except:
                        print(f"json file {save_json_filename} save failed")

                elif self.params["save_cases_jsons"]:
                    save_json_filename = os.path.join(
                        self.output_folder_name,
                        "all",
                        self.prefix
                        + "_"
                        + self.dataloader_type
                        + "_{}".format(self.image_counter)
                        + "_pedestrian.json",
                    )
                    tracks = additional_info_dictionary["tracks"]
                    try:
                        with open(save_json_filename, "w") as fp:
                            json.dump(tracks, fp, indent=2)
                    except:
                        print("json save failed")
                        # TODO(guy.rosman): find the cases in the runner where json saving failed.

            try:
                # Plot and save prediction visualization, but only if the trajectory is not straight.
                is_nonstraight = filter_nonstraight_trajectories(batch_itm, b, param)
                if self.params["save_cases_for_table"]:
                    # Save for table will not consider if it is a straight trajectory.
                    is_nonstraight = True

                    example_output_folder = os.path.expanduser("~/intent/artifacts/table_json/")  # default value
                    if self.params["table_output_folder_name"]:
                        example_output_folder = os.path.expanduser(self.params["table_output_folder_name"])

                    proto_file_name = batch_itm[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE][b]
                    proto_dir, proto_base_name = os.path.split(proto_file_name)
                    proto_number_str = os.path.splitext(proto_base_name)[0]
                    proto_tlog_name = os.path.basename(os.path.normpath(proto_dir))
                    image_name = proto_number_str
                    save_examples_filename = os.path.join(
                        example_output_folder,
                        proto_tlog_name,
                        image_name + "." + self.img_ext,
                    )
                    im.save(save_examples_filename)
                    if pedestrian_idx is not None:
                        if self.params["use_linear_model"]:
                            image_name += "_lin"
                        print(f"proto_tlog_name {proto_tlog_name}, image_name {image_name}.{self.img_ext}")
                        save_examples_filename_pedestrian = os.path.join(
                            example_output_folder,
                            proto_tlog_name,
                            image_name + "_ped" + "." + self.img_ext,
                        )
                        im_pedestrian.save(save_examples_filename_pedestrian)
                elif is_nonstraight:
                    save_nonstraight_examples_filename = os.path.join(
                        self.output_folder_name,
                        "nonstraight",
                        f"{self.prefix}_{self.dataloader_type}_{self.image_counter}.{self.img_ext}",
                    )
                    im.save(save_nonstraight_examples_filename)
                    if pedestrian_idx is not None:
                        save_nonstraight_examples_filename_pedestrian = os.path.join(
                            self.output_folder_name,
                            "nonstraight",
                            f"{self.prefix}_{self.dataloader_type}_{self.image_counter}_pedestrian.{self.img_ext}",
                        )
                        im_pedestrian.save(save_nonstraight_examples_filename_pedestrian)
            except:
                import IPython

                IPython.embed(header="check nonstraight filter")

            im.close()
            if pedestrian_idx is not None:
                im_pedestrian.close()

            self.image_counter += 1

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool = False) -> None:
        """Logger logic triggered at the end of an epoch.

        Parameters
        ----------
        idx : int
            Current epoch id.

        global_batch_cnt : int
            Number of batches processed so far.
        """


class SaveTrajectoriesLogHandler(LoggingHandler):
    """
    Saves predicted trajectories to disk

    It iterates through the dataset and writes out the data to a single pickle
    file at the end.  It assumes only data corresponding to instances specified
    by the '--serialize-trajectories' parameter will be in the dataset. It
    serializes everything needed to call
    intent.multiagents.trainer_visualization.visualize_prediction.
    """

    def __init__(self, params) -> None:
        super().__init__(params)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        linear = "-linear" if self.params["use_linear_model"] else ""
        self.filename = f"trajectory-{timestr}{linear}.pkl"
        self.output_dict = {"batch_itm": {}}
        param_keys = [
            "visualization_image_format",
            "use_linear_model",
            "view_min_span_size",
            "map_view_margin_around_trajectories",
        ]
        self.output_keys = [
            "predicted_trajectories",
            "map_validity",
            "map_others",
            "offset_x",
            "offset_y",
            "map_coordinates",
            "is_future_valid",
        ]
        self.output_dict["param"] = {k: self.params[k] for k in param_keys}
        scale = self.params["predictor_normalization_scale"]
        self.output_dict["scale"] = scale

    def iteration_update(self, data_dictionary: dict, stats_dict: dict) -> None:
        """Accumulate necessary data from each batch sufficient to generate trajectory visualizations

        data_dictionary : dict
            A dictionary of data that the logging handler can use.
            This logger requires the following keys be present:
                batch_itm, is_future_valid, predicted_trajectories, map_coordinates, map_others, map_validity, offset_x,
                 offset_y, param
        stats_dict : dict
            A dictionary of statistics to be aggregated and logged.
            This logger requires the following keys be present:
            **No keys required from this dict**
        """
        super().iteration_update(data_dictionary, stats_dict)
        batch_itm = data_dictionary["batch_itm"]

        for k, v in batch_itm.items():
            self.output_dict["batch_itm"] = self.accumulate(self.output_dict["batch_itm"], k, v)
        for k in self.output_keys:
            v = data_dictionary[k]
            self.output_dict = self.accumulate(self.output_dict, k, v)
        rotations_scene_local = data_dictionary["agent_transforms"][:, :, :2, :].transpose(2, 3)
        rotations_scene_ego = torch.masked_select(
            rotations_scene_local, batch_itm["is_ego_vehicle"].bool().unsqueeze(-1).unsqueeze(-1)
        ).view(-1, 2, 2)
        self.output_dict = self.accumulate(self.output_dict, "ego_rotation", rotations_scene_ego)

    @staticmethod
    def accumulate(dictionary: dict, key: Hashable, val: Union[list, torch.Tensor]) -> dict:
        """Accumulate lists and tensors in a dictionary"""
        if isinstance(val, list):
            try:
                dictionary[key].extend(val)
            except KeyError:
                dictionary[key] = val
        if isinstance(val, torch.Tensor):
            try:
                dictionary[key] = torch.cat([dictionary[key], val], dim=0)
            except KeyError:
                dictionary[key] = val
        return dictionary

    def epoch_start(self, dataloader_type):
        super().epoch_start(dataloader_type)

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool = False) -> None:
        """Write out accumulated data to a pickle file"""
        output_folder = os.path.expanduser(os.path.expandvars(self.params["runner_output_folder"]))
        os.makedirs(output_folder, exist_ok=True)
        filepath = os.path.join(output_folder, self.filename)
        pkldata = pickle.dumps(self.output_dict)
        print(f"writing trajectory to {filepath}")
        with open(filepath, "wb") as fp:
            fp.write(pkldata)
        print(f"finished writing trajectory to {filepath}")


class LogWorstCases(LoggingHandler):
    """Allow saving a list of worst (highest) cost examples for visualization.


    Parameters
    ----------
    logger_key : str
        The key to be logged.
    params : dict
        The command line arguments dictionary.
    output_folder_name : str, optional
        The name of the output folder.
    """

    # Keys from the batch that should be logged in the worst case statistics.
    _LOGGED_KEYS = (
        ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS,
        ProtobufPredictionDataset.DATASET_KEY_POSITIONS,
        ProtobufPredictionDataset.DATASET_KEY_DOT_KEYS,
        ProtobufPredictionDataset.DATASET_KEY_IS_EGO_VEHICLE,
        ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT,
        ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS,
    )
    _LOGGED_KEYS_OPTIONAL = (
        ProtobufPredictionDataset.DATASET_KEY_IMAGES,
        ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES,
        ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_MAPPING,
        ProtobufPredictionDataset.DATASET_KEY_MAP,
    )

    def __init__(self, logger_key: str, params: dict, output_folder_name: Optional[str] = None) -> None:
        # TODO(guy.rosman): See if it makes sense to inherit from LoggingHandler as it's a slightly different usecase.
        super().__init__(params)
        self.logger_key = logger_key
        self.params = params

        self.num_future_timepoints = None
        self.num_agents = None
        self.worst_mean_x = None
        self.worst_mean_y = None
        self.worst_is_future_valid = None
        self.worst_semantic_labels = None
        self.worst_predicted_semantics = None
        self.worst_map_coordinates = None
        self.worst_map_validity = None
        self.worst_predicted_traj = None
        self.worst_cost = None
        self.worst_itm = {}
        self.logged_keys_selected = list(self._LOGGED_KEYS)
        self.img_ext = params["visualization_image_format"]
        self.visual_size = None

        if output_folder_name is not None:
            self.output_folder_name = os.path.join(
                os.path.expanduser(os.path.expandvars(output_folder_name + "/worst_cases/")),
                self.logger_key.replace("/", "_"),
            )
        else:
            self.output_folder_name = None

    def epoch_start(self, dataloader_type, visual_size=None):
        """
        Signal epoch start
        :param dataloader_type: The dataloader type name - training/val/vis.
        :param visual_size: The number of examples to save.
        :return:
        """
        self.dataloader_type = dataloader_type
        self.num_future_timepoints = None
        self.worst_mean_x = None
        self.worst_mean_y = None
        self.worst_is_future_valid = None
        self.worst_predicted_traj = None
        self.worst_cost = None
        self.worst_itm = {}
        if visual_size:
            self.visual_size = visual_size
        else:
            self.visual_size = min(
                self.params["vis_batch_size"],
                self.params["num_visualization_worst_cases"],
            )

    def iteration_update(self, data_dict: dict, stats_dict: dict) -> None:
        """Update worst case logger iteration

        Save new, better (higher g_cost1) examples.

        Parameters
        ----------
        data_dict: dict
            A dictionary of data about the batch data.
            This dictionary should contain the following keys (key contents defined in abstract class):
                batch_itm, batch_cost, predicted_trajectories, offset_x, offset_y, is_future_valid,
                semantic_labels, predicted_semantics, map_coordinates, map_validity, map_others
        stats_dict: dict
            A dictionary of data about statistics.
            This dictionary should contain the following keys (key contents defined in abstract class):
                num_future_timepoints
        """
        batch_itm = data_dict["batch_itm"]
        batch_cost = data_dict["batch_cost"]
        predicted_trajectories_scene = data_dict["predicted_trajectories"]
        num_future_timepoints = stats_dict["num_future_timepoints"]
        offset_x = data_dict["offset_x"]
        offset_y = data_dict["offset_y"]
        is_future_valid = data_dict["is_future_valid"]
        semantic_labels = data_dict["semantic_labels"]
        predicted_semantics = data_dict["predicted_semantics"]
        map_coordinates = data_dict["map_coordinates"]
        map_validity = data_dict["map_validity"]
        map_others = data_dict["map_others"]

        batch_itm_cpu = self._to_cpu(batch_itm)
        if self.worst_cost is None:
            self.worst_cost = batch_cost[: self.visual_size].detach().cpu()
            # worst_cost must be 1-dimensional, along the batch dimension.
            assert len(self.worst_cost.shape) == 1
            self.num_future_timepoints = num_future_timepoints
            self.worst_predicted_traj = predicted_trajectories_scene[: self.visual_size, ...].detach().cpu()
            self.worst_mean_x = offset_x[: self.visual_size, :].detach().cpu()
            self.worst_mean_y = offset_y[: self.visual_size, :].detach().cpu()
            self.worst_is_future_valid = is_future_valid[: self.visual_size, :, :].detach().cpu()
            self.worst_itm = {}
            for key in batch_itm_cpu:
                self.worst_itm[key] = batch_itm_cpu[key][: self.visual_size]
            if semantic_labels is not None:
                self.worst_semantic_labels = semantic_labels.detach()
            if predicted_semantics is not None:
                self.worst_predicted_semantics = predicted_semantics.detach()
            if ProtobufPredictionDataset.DATASET_KEY_MAP in batch_itm_cpu and self.params["map_input_type"] == "point":
                self.worst_map_coordinates = map_coordinates.detach().cpu()
                self.worst_map_validity = map_validity.detach().cpu()
                self.worst_map_others = map_others.detach().cpu()
            else:
                self.worst_map_coordinates = None
                self.worst_map_validity = None
                self.worst_map_others = None

            for key in self._LOGGED_KEYS_OPTIONAL:
                if key in batch_itm_cpu and key not in self.logged_keys_selected:
                    self.logged_keys_selected.append(key)
        else:
            # Concatenate the worst and current batch
            # This code assumes self.worst_cost, and batch_cost, are 1D tensors
            assert len(self.worst_cost.shape) == 1
            concat_cost = torch.cat([self.worst_cost, batch_cost.detach().cpu()], dim=0)
            idx_worst = torch.argsort(concat_cost, descending=True, dim=0)[: self.visual_size]
            self.worst_cost = concat_cost[idx_worst]
            self.worst_predicted_traj = self._concat_and_select(
                idx_worst, [self.worst_predicted_traj, predicted_trajectories_scene.detach().cpu()]
            )
            self.worst_mean_x = self._concat_and_select(idx_worst, [self.worst_mean_x, offset_x.detach().cpu()])
            self.worst_mean_y = self._concat_and_select(idx_worst, [self.worst_mean_y, offset_y.detach().cpu()])
            self.worst_is_future_valid = self._concat_and_select(
                idx_worst, [self.worst_is_future_valid, is_future_valid.detach().cpu()]
            )
            for key in self._LOGGED_KEYS_OPTIONAL:
                if key in batch_itm_cpu:
                    self.worst_itm[key] = self._concat_and_select(
                        idx_worst, [self.worst_itm[key], batch_itm_cpu[key].detach()]
                    )

            if semantic_labels is not None:
                self.worst_semantic_labels = self._concat_and_select(
                    idx_worst, [self.worst_semantic_labels, semantic_labels.detach()]
                )
            if predicted_semantics is not None:
                self.worst_predicted_semantics = self._concat_and_select(
                    idx_worst, [self.worst_predicted_semantics, predicted_semantics.detach()]
                )
            if ProtobufPredictionDataset.DATASET_KEY_MAP in batch_itm_cpu and self.params["map_input_type"] == "point":
                map_coord = map_coordinates.detach().cpu()
                self.worst_map_coordinates = self._concat_and_select(idx_worst, [self.worst_map_coordinates, map_coord])
                self.worst_map_validity = self._concat_and_select(
                    idx_worst, [self.worst_map_validity, map_validity.detach().cpu()]
                )
                self.worst_map_others = self._concat_and_select(
                    idx_worst, [self.worst_map_others, map_others.detach().cpu()]
                )

            cur_instance_info = self.worst_itm[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO]
            cur_protobuf_file = self.worst_itm[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE]
            self.worst_itm = self._concat_and_select_dict(
                idx_worst, [self.worst_itm, batch_itm_cpu], self.logged_keys_selected
            )

            concat_instance_info = (
                cur_instance_info + batch_itm_cpu[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO]
            )
            self.worst_itm[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO] = []
            for i in range(min(len(idx_worst), self.visual_size)):
                self.worst_itm[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO].append(
                    concat_instance_info[idx_worst[i]]
                )

            concat_protobuf_file = (
                cur_protobuf_file + batch_itm_cpu[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE]
            )
            self.worst_itm[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE] = []
            for i in range(min(len(idx_worst), self.visual_size)):
                self.worst_itm[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE].append(
                    concat_protobuf_file[idx_worst[i]]
                )

    @staticmethod
    def _concat_and_select(idx_worst: torch.Tensor, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Concatenates list of tensors (along dim=0) and selects elements according to given index."""
        concatenated_tensors = torch.cat(tensors, dim=0)
        return concatenated_tensors[idx_worst]

    @staticmethod
    def _to_cpu(batch_itm: dict) -> dict:
        """Concatenates list of tensors (along dim=0) and selects elements according to given index."""
        batch_itm_result = copy.copy(batch_itm)
        for key in batch_itm:
            if isinstance(batch_itm[key], torch.Tensor):
                batch_itm_result[key] = batch_itm_result[key].cpu()

        return batch_itm_result

    @staticmethod
    def _concat_and_select_dict(idx_worst: torch.Tensor, dicts: List[dict], logged_keys: List[str]) -> dict:
        """Same as _concat_and_select applied to a list of dictionaries with same keys."""
        result = {}
        for key in logged_keys:
            tensorlist = [cur_dict[key].detach() for cur_dict in dicts]
            result[key] = LogWorstCases._concat_and_select(idx_worst, tensorlist)
        return result

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool = False) -> None:
        """Do any needed processing at the end of the epoch."""
        if self.worst_cost is None:
            print(f"No worst cost logs for {self.logger_key}")
            return
        elif skip_visualization or self.output_folder_name is None:
            return

        os.makedirs(self.output_folder_name, exist_ok=True)
        logger_key_path_safe = self.logger_key.replace("/", "_")

        num_worst_cases = len(self.worst_cost)
        print(f"Visualizing {num_worst_cases} datapoints with the worst '{self.logger_key}' value.")
        for cur_datapoint in range(num_worst_cases):
            solution = {
                "predicted_trajectories": self.worst_predicted_traj[cur_datapoint],
                "is_future_valid": self.worst_is_future_valid[cur_datapoint],
            }
            map_coordinates = (
                (self.worst_map_coordinates / self.params["predictor_normalization_scale"])
                if (self.worst_map_coordinates is not None)
                else None
            )
            img, additional_info_dictionary = visualize_prediction(
                self.worst_itm,
                cur_datapoint,
                solution,
                scale=self.params["predictor_normalization_scale"],
                means=(self.worst_mean_x[cur_datapoint], self.worst_mean_y[cur_datapoint]),
                cost=None,
                param=self.params,
                map_coordinates=map_coordinates,
                map_validity=self.worst_map_validity,
                map_others=self.worst_map_others,
            )
            im = Image.fromarray(img)

            save_filename = os.path.join(
                self.output_folder_name,
                f"worst_{self.dataloader_type}_{logger_key_path_safe}_{cur_datapoint}.{self.img_ext}",
            )
            im.save(save_filename)

            if self.params["worst_case_per_agent_vis"]:
                map_coordinates = (
                    (self.worst_map_coordinates / self.params["predictor_normalization_scale"])
                    if (self.worst_map_coordinates is not None)
                    else None
                )
                for cur_agent in range(self.worst_itm["positions"].shape[1]):
                    img, _ = visualize_prediction(
                        self.worst_itm,
                        cur_datapoint,
                        solution,
                        scale=self.params["predictor_normalization_scale"],
                        means=(self.worst_mean_x[cur_datapoint], self.worst_mean_y[cur_datapoint]),
                        cost=None,
                        param=self.params,
                        map_coordinates=map_coordinates,
                        map_validity=self.worst_map_validity,
                        map_others=self.worst_map_others,
                        agent_set=[cur_agent],
                    )
                    im = Image.fromarray(img)

                    save_filename = os.path.join(
                        self.output_folder_name,
                        f"worst_{self.dataloader_type}_{logger_key_path_safe}_{cur_datapoint}_{cur_agent}.{self.img_ext}",
                    )
                    im.save(save_filename)
            if self.params["save_cases_jsons"]:
                save_json_filename = os.path.join(
                    self.output_folder_name,
                    "worst_{}_{}_{}.json".format(self.dataloader_type, logger_key_path_safe, cur_datapoint),
                )
                tracks = additional_info_dictionary["tracks"]
                with open(save_json_filename, "w") as fp:
                    json.dump(tracks, fp, indent=2)

            if self.params["save_cases_camera_videos"]:
                json_instance_info = json.loads(self.worst_itm[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO][0])
                source_tlog = json_instance_info["source_tlog"]
                source_tlog_pathlib = Path(source_tlog)
                tlog_recent_path = os.path.join(source_tlog_pathlib.parts[-2:])
                search_result = search_tlog_in_s3(tlog_recent_path, self.params)
                timestamp = json_instance_info["timestamp"]
                if search_result is not None:
                    folder_name = search_result["full_folder_path"]
                    save_tlog_foldername = os.path.join(
                        self.output_folder_name,
                        "worst_{}_{}_{}_folder".format(self.dataloader_type, logger_key_path_safe, cur_datapoint),
                    )
                    # TODO(guy.rosman) either aws s3 sync the folder, or save a pointer to it.
                    save_json_cmd_filename = os.path.join(
                        self.output_folder_name,
                        "worst_{}_{}_{}_cmd.json".format(self.dataloader_type, logger_key_path_safe, cur_datapoint),
                    )
                    copy_command = ["aws", "s3", "sync", folder_name, save_tlog_foldername]
                    # TODO(guy.rosman) - here we can use the tlog,timestamp, and fetch the video with the command above.
                    # TODO  Should add an option to do this and not just save the command,
                    # TODO  but it's too expensive to run for every example.
                    with open(save_json_cmd_filename, "w") as fp:
                        json.dump(
                            {
                                "copy_command": " ".join(copy_command),
                                "timestamp": timestamp,
                                "source_tlog": source_tlog,
                            },
                            fp,
                            indent=2,
                        )


class WaymoLogHandler(LoggingHandler):
    """
    A log handler for Waymo metrics.
    """

    def __init__(self, params) -> None:
        super().__init__(params)

        from data_sources.waymo.util.metrics import MotionMetrics, default_metrics_config

        self.waymo_motion_metrics = MotionMetrics(default_metrics_config())

    def initialize_training(self, writer):
        super().initialize_training(writer)

    def epoch_start(self, dataloader_type):
        super().epoch_start(dataloader_type)

    def iteration_update(self, data_dictionary: dict, stats_dict: dict):
        """Per iteration update -- save statistics of the images into tensorboard

        data_dictionary : dict
            A dictionary of data that the logging handler can use.
            This logger requires the following keys be present:
                additional_inputs, agent_additional_inputs, batch_itm, expected_trajectories, input_tensor,
                is_future_valid, is_valid, predicted_trajectories,

        stats_dict : dict
            A dictionary of statistics to be aggregated and logged.
            This logger requires the following keys be present:
            **No keys required from this dict**
        """
        super().iteration_update(data_dictionary, stats_dict)

        # Skip update if not in training or validation.
        if self.dataloader_type not in ("train", "validation"):
            return

        predicted_trajectories_scene = data_dictionary["predicted_trajectories"]

        # Obtain tensors.
        # [batch_size, num_preds, top_k, num_agents_per_joint_prediction, pred_steps, 2].
        pred_trajectory = (
            predicted_trajectories_scene.permute(0, 1, 4, 2, 3) / self.params["predictor_normalization_scale"]
        )

        # [batch_size, num_agents_gt, total_steps, 7].
        gt_trajectory_additional_state = data_dictionary["additional_inputs"][
            ProtobufPredictionDataset.DATASET_KEY_AUXILIARY_STATE
        ]
        gt_trajectory_full = (
            data_dictionary["expected_trajectories_full"] / self.params["predictor_normalization_scale"]
        )
        gt_trajectory_full = torch.cat([gt_trajectory_full, gt_trajectory_additional_state], -1)
        # [batch_size, num_agents_gt, total_steps].
        is_valid_full = data_dictionary["is_valid_full"]
        # [batch_size, num_agents].
        # Map type indices (VEHICLE = 0, PEDESTRIAN = 1, CYCLIST = 2)
        # to Waymo object type: VEHICLE = 1, PEDESTRIAN = 2, CYCLIST = 3
        object_type = torch.argmax(data_dictionary["batch_agent_type"], -1) + 1

        # [batch_size, num_agents].
        batch_size = pred_trajectory.shape[0]
        num_agents = pred_trajectory.shape[1]

        pred_gt_indices = torch.arange(0, num_agents, dtype=torch.int64)
        # [batch_size, num_agents, 1].
        pred_gt_indices = torch.tile(pred_gt_indices.unsqueeze(0).unsqueeze(2), (batch_size, 1, 1))
        # [batch_size, num_agents, 1].
        pred_gt_indices_mask = data_dictionary["relevant_agents"].unsqueeze(-1)

        # Adjust dimensions of predicted trajectories for Waymo metrics computation.
        # Dimension info can be found at data_sources/waymo/util/metrics.py, MotionMetrics.update_state.
        if self.params["use_marginal_error"]:
            pred_trajectory_final = pred_trajectory.unsqueeze(3).cpu().detach()
        else:
            pred_trajectory_final = pred_trajectory.transpose(1, 2).unsqueeze(1).cpu().detach()
            pred_gt_indices = pred_gt_indices.transpose(1, 2)
            pred_gt_indices_mask = pred_gt_indices_mask.transpose(1, 2)

        # Fake the score.
        # TODO(cyrushx): Add score once it is available.
        # [batch_size, num_preds, top_k].
        pred_score = torch.ones(pred_trajectory_final.shape[:3])

        self.waymo_motion_metrics.update_state(
            pred_trajectory_final,
            pred_score.cpu(),
            gt_trajectory_full.cpu().to(torch.float32),
            is_valid_full.cpu() > 0,
            pred_gt_indices.cpu(),
            pred_gt_indices_mask.cpu() > 0,
            object_type.cpu(),
        )

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool = False) -> None:
        """Logger logic triggered at the end of an epoch.

        Parameters
        ----------
        idx : int
            Current epoch id.

        global_batch_cnt : int
            Number of batches processed so far.
        """
        # Skip if not in training or validation.
        if self.dataloader_type not in ("train", "validation"):
            return

        from waymo_open_dataset.metrics.python import config_util_py as config_util

        from data_sources.waymo.util.metrics import default_metrics_config

        waymo_metrics_results = self.waymo_motion_metrics.result()
        metric_names = config_util.get_breakdown_names_from_motion_config(default_metrics_config())
        for i, m in enumerate(["min_ade", "min_fde", "miss_rate", "overlap_rate", "map"]):
            for j, n in enumerate(metric_names):
                self.writer.add_scalar(
                    "{}_waymo_metrics/{}/{}".format(self.dataloader_type, m, n),
                    waymo_metrics_results[i, j].numpy(),
                    global_step=global_batch_cnt,
                )
        self.waymo_motion_metrics.reset_state()
