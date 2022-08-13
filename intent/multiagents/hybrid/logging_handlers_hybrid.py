import json
import os
import time

import numpy as np
import scipy
import torch
from PIL import Image

from intent.multiagents.logging_handlers import LoggingHandler
from intent.multiagents.trainer_visualization import visualize_prediction


# Log stats from hybrid model.
class SaveHybridErrorStatistics(LoggingHandler):
    # Define constants.
    VALID_MON_FDE = "mon_fde_full"
    VALID_MON_ADE = "mon_ade_full"
    MON_DISCRETE = "mon_discrete"
    VALID_MON_BFDE = "b_mon_fde_full"
    VALID_MON_BADE = "b_mon_ade_full"

    def __init__(
        self,
        params: dict,
        prefix: str,
        subsamplers: list = [],
        subsamplers_names: list = [],
        visualize: bool = False,
        output_folder_name: str = None,
        save_results: bool = False,
        save_single_results: bool = False,
    ) -> None:
        """
        Log statistics for hybrid model.

        Parameters
        ----------
        params: dict
            Model and training parameters.
        prefix: str
            Prefix of filename to save.
        subsamplers: list
            List of sub-samplers to use.
        subsamplers_names: list
            List of names for each subsampler.
        visualize: bool
            Whether to visualize predictions.
        output_folder_name: str
            Folder name to save results.
        save_results: bool
            Whether to save individual prediction results into json files.
        save_single_results: bool
            Whether to save accumulated prediction results.
        """
        super().__init__(params)
        self.prefix = prefix
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.report_unique_id = timestr
        if output_folder_name is not None:
            self.output_folder_name = output_folder_name
        else:
            self.output_folder_name = os.path.expanduser(os.path.expandvars(params["runner_output_folder"]))
            self.output_folder_name = os.path.join(self.output_folder_name, params["resume_session_name"])
        os.makedirs(self.output_folder_name, exist_ok=True)

        self.save_results = save_results
        # This option is not supported.
        self.save_single_results = save_single_results
        if self.save_results:
            self.results_output_folder_name = os.path.join(self.output_folder_name, "results")
            os.makedirs(self.results_output_folder_name, exist_ok=True)

        # Create lists to store aggregate statistics.
        self.aggregate_statistics = {}
        self.horizon = [29]
        for name in subsamplers_names:
            for h in self.horizon:
                self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_FDE] = []
                self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_ADE] = []
                self.aggregate_statistics[name + "_" + str(h) + "_" + self.MON_DISCRETE] = []

                self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_BFDE] = []
                self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_BADE] = []

        self.discrete_supervised = params["discrete_supervised"]
        self.subsample_size = params["hybrid_runner_subsample_size"]
        self.subsamplers = subsamplers
        self.subsamplers_names = subsamplers_names
        self.params = params
        self.visualize = visualize
        self.image_counter = 0

        if self.save_results:
            self.results = {}
            self.results_counter = 0

    def initialize_training(self, writer):
        super().initialize_training(writer)

    def epoch_start(self, dataloader_type):
        super().epoch_start(dataloader_type)

    def iteration_update(self, data_dictionary: dict, stats_dict: dict) -> None:
        """
        Per iteration update -- log prediction statistics from samples using different sample selection methods.

        Parameters
        ----------
        data_dictionary: dict
            Output from prediction model.
        stats_dict: dict
            Statistics from prediction model.
        """
        scale = self.params["predictor_normalization_scale"]
        offset_x = data_dictionary["offset_x"]
        offset_y = data_dictionary["offset_y"]

        # Get predicted position shifted to origin in the right scale and rotation.
        # [batch_size, num_agents, num_future_steps, 2, num_samples]
        predicted_trajectories = data_dictionary["predicted_trajectories"] / scale
        # [batch_size, num_agents, num_future_steps, 2]
        expected_trajectories = data_dictionary["expected_trajectories"] / scale
        # [batch_size, num_agents, num_future_steps]
        expected_modes = data_dictionary["additional_inputs"]["maneuvers_future"]
        stats_list = data_dictionary["stats_list"]
        # [batch_size, num_agents, num_future_steps, 5, num_samples]
        predicted_modes = torch.stack([stats["discrete_samples"] for stats in stats_list], -1)
        # [batch_size, num_agents, num_samples]
        discrete_weights = torch.stack([stats["discrete_samples_log_weight"] for stats in stats_list], -1)

        # Remove the last sample if it comes from supervised learning, which uses ground truth future mode.
        if self.discrete_supervised:
            predicted_trajectories = predicted_trajectories[..., :-1]
            predicted_modes = predicted_modes[..., :-1]
            discrete_weights = discrete_weights[..., :-1]

        data_to_save = {
            "predicted_trajectories": predicted_trajectories.cpu().detach().numpy(),
            "predicted_modes": predicted_modes.cpu().detach().numpy(),
            "discrete_weights": discrete_weights.cpu().detach().numpy(),
            "expected_trajectories": expected_trajectories.cpu().detach().numpy(),
            "offset_x": data_dictionary["offset_x"].cpu().detach().numpy(),
            "offset_y": data_dictionary["offset_y"].cpu().detach().numpy(),
            "full_trajectories": data_dictionary["batch_itm"]["positions"].cpu().detach().numpy(),
            "map_coordinates": data_dictionary["map_coordinates"].cpu().detach().numpy(),
            "map_validity": data_dictionary["map_validity"].cpu().detach().numpy(),
        }

        # Collect indices selected from subsamplers.
        for i, subsampler in enumerate(self.subsamplers):
            subsampler_name = self.subsamplers_names[i]
            selected_indices = subsampler(predicted_trajectories, predicted_modes, discrete_weights, self.params)

            # Get predictions based on selected indices.
            selected_indices_traj = selected_indices[:, :, None, None].repeat(
                1, 1, predicted_trajectories.shape[2], predicted_trajectories.shape[3], 1
            )
            selected_predicted_trajectories = torch.gather(predicted_trajectories, -1, selected_indices_traj)
            selected_indices_mode = selected_indices[:, :, None, None].repeat(
                1, 1, predicted_modes.shape[2], predicted_modes.shape[3], 1
            )
            selected_predicted_modes = torch.gather(predicted_modes, -1, selected_indices_mode)
            selected_predicted_trajectories_weights = torch.gather(discrete_weights, -1, selected_indices)
            selected_predicted_trajectories_p = selected_predicted_trajectories_weights / torch.sum(
                selected_predicted_trajectories_weights, -1, keepdims=True
            )
            assert selected_predicted_trajectories.shape[-1] == self.subsample_size, "Predicted sample size mismatch."

            # Visualize predictions and save to files.
            if self.visualize:
                batch_itm = data_dictionary["batch_itm"]
                is_future_valid = data_dictionary["is_future_valid"]
                param = data_dictionary["param"]
                map_coordinates = data_dictionary["map_coordinates"]
                map_validity = data_dictionary["map_validity"]
                map_others = data_dictionary["map_others"]

                batch_size = is_future_valid.shape[0]
                for b in range(batch_size):
                    solution = {
                        "predicted_trajectories": selected_predicted_trajectories[b] * scale,
                        "is_future_valid": is_future_valid[b, :, :],
                    }

                    # Plot the visualization of the predictions, save each visualization in a png.
                    img, _ = visualize_prediction(
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
                        self.prefix + "_" + subsampler_name + "_{}".format(self.image_counter) + ".png",
                    )
                    im.save(save_filename)
                    self.image_counter += 1

            # Compute marginal MoN errors.
            pred_l2 = ((selected_predicted_trajectories - expected_trajectories[..., :2].unsqueeze(-1)) ** 2).sum(3)
            pred_dist = torch.sqrt(pred_l2)
            # Compute binary difference in mode predictions.
            pred_mode_diff = selected_predicted_modes.argmax(-2) != expected_modes.unsqueeze(-1).repeat(
                [1, 1, 1, selected_predicted_trajectories.shape[-1]]
            )

            # Collect stats for each horizon (1s and 3s).
            for h in self.horizon:
                pred_MoN_ADE = pred_dist[:, :, : h + 1].mean(2).min(2)[0].mean(1).cpu().detach().numpy()
                pred_MoN_FDE = pred_dist[:, :, h].min(2)[0].mean(1).cpu().detach().numpy()
                pred_MoN_discrete = (
                    (pred_mode_diff[:, :, : h + 1].sum(2) / (h + 1)).min(2)[0].mean(1).cpu().detach().numpy()
                )
                best_sample_index = pred_dist[:, 0, : h + 1].mean(1).argmin(-1)
                best_sample_p = [
                    selected_predicted_trajectories_p[b, 0, best_sample_index[b]]
                    for b in range(best_sample_index.shape[0])
                ]
                best_sample_p = torch.stack(best_sample_p).cpu().detach().numpy()
                brier_constant = (1.0 - best_sample_p) ** 2

                self.aggregate_statistics[subsampler_name + "_" + str(h) + "_" + self.VALID_MON_ADE] += list(
                    pred_MoN_ADE
                )
                self.aggregate_statistics[subsampler_name + "_" + str(h) + "_" + self.VALID_MON_FDE] += list(
                    pred_MoN_FDE
                )
                self.aggregate_statistics[subsampler_name + "_" + str(h) + "_" + self.MON_DISCRETE] += list(
                    pred_MoN_discrete
                )
                self.aggregate_statistics[subsampler_name + "_" + str(h) + "_" + self.VALID_MON_BADE] += list(
                    pred_MoN_ADE + brier_constant
                )
                self.aggregate_statistics[subsampler_name + "_" + str(h) + "_" + self.VALID_MON_BFDE] += list(
                    pred_MoN_FDE + brier_constant
                )

            data_to_save[subsampler_name + "_predicted_trajectories"] = (
                selected_predicted_trajectories.cpu().detach().numpy()
            )
            data_to_save[subsampler_name + "_predicted_modes"] = selected_predicted_modes.cpu().detach().numpy()
            data_to_save[subsampler_name + "_discrete_weights"] = (
                selected_predicted_trajectories_weights.cpu().detach().numpy()
            )
            data_to_save[subsampler_name + "_mon_ade"] = pred_MoN_ADE
            data_to_save[subsampler_name + "_mon_fde"] = pred_MoN_FDE

        keys_raw = data_dictionary["batch_itm"]["instance_info"]

        # Save individual results into jsons.
        if self.save_results:
            keys = [json.loads(key)["json_dir"] + json.loads(key)["source_tlog"] for key in keys_raw]

            for i, k in enumerate(keys):
                results_data = {}
                for key_data_to_save in data_to_save:
                    results_data[key_data_to_save] = data_to_save[key_data_to_save][i].tolist()

                # Save each sample into a json file.
                json_filename = os.path.join(self.results_output_folder_name, k.split("/")[-1] + ".json")
                with open(json_filename, "w") as fp:
                    json.dump(results_data, fp, indent=2)

                self.results_counter += 1

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool) -> None:
        """
        Logger logic triggered at the end of an epoch.

        Parameters
        ----------
        idx: int
            Current epoch id.
        global_batch_cnt : int
            Number of batches processed so far.
        skip_visualization : bool
            Whether to skip visualizations.
        """
        # Create a statistics dictionary
        stats = {}

        # Compute mean for each stats.
        for name in self.subsamplers_names:
            stats["sample_count"] = len(
                self.aggregate_statistics[name + "_" + str(self.horizon[0]) + "_" + self.VALID_MON_ADE]
            )
            for h in self.horizon:
                stats[name + "_" + str(h) + "_" + self.VALID_MON_ADE] = float(
                    np.mean(self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_ADE])
                )
                stats[name + "_" + str(h) + "_" + self.VALID_MON_FDE] = float(
                    np.mean(self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_FDE])
                )
                stats[name + "_" + str(h) + "_" + self.MON_DISCRETE] = float(
                    np.mean(self.aggregate_statistics[name + "_" + str(h) + "_" + self.MON_DISCRETE])
                )
                stats[name + "_" + str(h) + "_" + self.VALID_MON_BADE] = float(
                    np.mean(self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_BADE])
                )
                stats[name + "_" + str(h) + "_" + self.VALID_MON_BFDE] = float(
                    np.mean(self.aggregate_statistics[name + "_" + str(h) + "_" + self.VALID_MON_BFDE])
                )

        # Print stats.
        for s in stats:
            print(s, stats[s])

        # Save the statistics dictionary
        json_filename = os.path.join(self.output_folder_name, self.prefix + "_" + self.report_unique_id + ".json")
        print("Saving statistics to {}.".format(json_filename))
        with open(json_filename, "w") as fp:
            json.dump(stats, fp, indent=2)

        if self.save_results:
            print("{} results saved to to {}.".format(self.results_counter, self.results_output_folder_name))
