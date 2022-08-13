import json
import os

import matplotlib.cm as cm
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from intent.multiagents.additional_callback import AdditionalTrainerCallback
from intent.multiagents.logging_handlers import LoggingHandler
from intent.multiagents.trainer_visualization import get_img_from_fig, visualize_prediction
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset


class LanguageVisualizer(LoggingHandler):
    def __init__(self, params: dict, trainer_callback: AdditionalTrainerCallback) -> None:
        """A logging handler to visualize trajetories with language tokens.

        Parameters
        ----------
        params: dict
            The dictionary of parameters
            "language_vocab": The torchtext Vocab instance that maps tokens to ids.
        trainer_callback: AdditionalTrainerCallback
            The callback to convert solutions and decoding results.
        """
        super().__init__(params)
        self.vocab = params["language_vocab"]
        self.color_map = [cm.rainbow(i / len(self.vocab)) for i in range(len(self.vocab))]
        self.trainer_callback = trainer_callback

    def epoch_start(self, dataloader_type: str) -> None:
        return None

    def viz_token_trajectory(
        self, batch: dict, batch_idx: int, solution: dict, scale: float, means: tuple, params: dict
    ):
        """Add image that shows associated parts in trajectory for predicted tokens.

        Parameters
        ----------
        batch: dict
            A batch from the data loader.
        batch_idx: int
            The index to get item for a batch.
        solution: dict
            The solution ot be visualized.
        scale: float
            Scale for normalizing the coordinates.
        means: tuple
            Means for normalizing the coordinates.

        Returns
        -------
        numpy.ndarray
            An image with the visualization.
        """
        mean_x, mean_y = means
        positions = batch[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][batch_idx, :, :, :2].float()
        is_valid = batch[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][batch_idx, :, :, 2]
        num_past_points = batch[ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS][batch_idx]
        num_agents = positions.shape[0]
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.2, 0.75, 0.75])  # [left, bottom, width, height]
        max_trajectory_position_x = -np.Inf
        max_trajectory_position_y = -np.Inf
        min_trajectory_position_x = np.Inf
        min_trajectory_position_y = np.Inf
        if "predicted_trajectories" not in solution or len(solution["token_attention_weights"]) == 0:
            return
        for agent_idx in range(num_agents):
            # Plot the past trajectory.
            valid_pos = positions[agent_idx, is_valid[agent_idx, :].bool(), :]
            if len(valid_pos) == 0:
                continue
            valid_x = valid_pos[:, 0].detach().cpu().numpy()
            valid_y = valid_pos[:, 1].detach().cpu().numpy()
            past_agent_color = "gray"
            ax.plot(valid_x[:num_past_points], valid_y[:num_past_points], color=past_agent_color, lw=1.0)
            ax.plot(
                valid_x[num_past_points : num_past_points + 1],
                valid_y[num_past_points : num_past_points + 1],
                "x",
                color=past_agent_color,
                lw=2.0,
            )
            ax.plot(valid_x[num_past_points:], valid_y[num_past_points:], ":", color=past_agent_color, lw=2.0)
            max_trajectory_position_x = max(max_trajectory_position_x, max(valid_x))
            max_trajectory_position_y = max(max_trajectory_position_y, max(valid_y))
            min_trajectory_position_x = min(min_trajectory_position_x, min(valid_x))
            min_trajectory_position_y = min(min_trajectory_position_y, min(valid_y))
            # Plot the predicted future trajectories.
            if "predicted_trajectories" in solution:
                is_future_valid = solution["is_future_valid"]
                for sample_i in range(solution["predicted_trajectories"].shape[3]):
                    predicted_sample = solution["predicted_trajectories"][
                        agent_idx, is_future_valid[agent_idx, :].bool(), :, sample_i
                    ]
                    if predicted_sample.numel() == 0:
                        continue
                    predicted_x = (
                        predicted_sample[:, 0].detach().cpu().numpy() + mean_x[agent_idx].cpu().numpy()
                    ) / scale
                    predicted_y = (
                        predicted_sample[:, 1].detach().cpu().numpy() + mean_y[agent_idx].cpu().numpy()
                    ) / scale
                    attention_weights = solution["token_attention_weights"][agent_idx]
                    predicted_tokens = solution["predicted_tokens"][agent_idx]
                    num_time_steps = len(attention_weights)
                    for t in range(num_time_steps - 1):
                        token_id = torch.argmax(attention_weights[t])
                        if token_id >= len(predicted_tokens):  # attend to the paddings
                            color = self.color_map[self.vocab.get_stoi()["<pad>"]]
                        else:
                            token = predicted_tokens[token_id]
                            color = self.color_map[self.vocab.get_stoi()[token]]
                        ax.plot(
                            predicted_x[t : t + 2],
                            predicted_y[t : t + 2],
                            color=color,
                            lw=0.5,
                        )
                    max_trajectory_position_x = max(max_trajectory_position_x, max(predicted_x))
                    max_trajectory_position_y = max(max_trajectory_position_y, max(predicted_y))
                    min_trajectory_position_x = min(min_trajectory_position_x, min(predicted_x))
                    min_trajectory_position_y = min(min_trajectory_position_y, min(predicted_y))
        ax.set_aspect("equal")
        # Make sure the size is the same as the original visualization.
        x_min, x_max, y_min, y_max = ax.axis()
        map_view_min_span_size = params["view_min_span_size"]
        map_view_margin_around_trajectories = params["map_view_margin_around_trajectories"]
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        x_min = min(x_min, x_c - map_view_min_span_size)
        y_min = min(y_min, y_c - map_view_min_span_size)
        x_max = max(x_max, x_c + map_view_min_span_size)
        y_max = max(y_max, y_c + map_view_min_span_size)
        x_min = max(x_min, min_trajectory_position_x - map_view_margin_around_trajectories)
        y_min = max(y_min, min_trajectory_position_y - map_view_margin_around_trajectories)
        x_max = min(x_max, max_trajectory_position_x + map_view_margin_around_trajectories)
        y_max = min(y_max, max_trajectory_position_y + map_view_margin_around_trajectories)
        try:
            ax.axis((x_min, x_max, y_min, y_max))
        except:
            print("Failed to change axis")
        # Add token legends.
        label_handles = []
        for token_id, color in enumerate(self.color_map):
            token = self.vocab.lookup_token(token_id)
            l = Line2D([], [], color=color, label=token)
            label_handles.append(l)
        ax.legend(handles=label_handles, title="Tokens", bbox_to_anchor=(1.0, 0.85), loc="upper left")
        img = get_img_from_fig(fig, image_format=params["visualization_image_format"])
        plt.close(fig)
        return img

    def iteration_update(self, data_dictionary: dict, stats_dict: dict) -> None:
        if data_dictionary["dataloader_type"] == "vis":
            batch_itm = data_dictionary["batch_itm"]
            offset_x = data_dictionary["offset_x"]
            offset_y = data_dictionary["offset_y"]
            scale = data_dictionary["param"]["predictor_normalization_scale"]
            is_future_valid = data_dictionary["is_future_valid"]
            predicted_trajectories = data_dictionary["predicted_trajectories"]
            batch_size = is_future_valid.shape[0]
            for b in range(batch_size):
                solution = {
                    "predicted_trajectories": predicted_trajectories[b],
                    "is_future_valid": is_future_valid[b],
                }
                predicted = {}
                self.trainer_callback.update_decoding(predicted, data_dictionary["stats_list"])
                self.trainer_callback.update_solution(solution, predicted, b)
                visual_idx = data_dictionary["batch_itm_index"] * batch_size + b
                img = self.viz_token_trajectory(
                    batch_itm,
                    b,
                    solution,
                    scale=scale,
                    means=(offset_x[b, :], offset_y[b, :]),
                    params=data_dictionary["param"],
                )
                if img is not None:
                    self.writer.add_image(
                        "vis/{}/token".format(visual_idx),
                        img.transpose(2, 0, 1),
                        global_step=data_dictionary["global_batch_cnt"],
                    )

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool) -> None:
        return None


class SaveLanguageErrorStatistics(LoggingHandler):
    # Define constants.
    MULTI_TOKENS = "multitokens"

    def __init__(
        self, params: dict, trainer_callback: AdditionalTrainerCallback, output_folder_name: str = None
    ) -> None:
        """Save statistics for language model.

        Parameters
        ----------
        params: dict
            The dictionary of parameters
            "language_vocab": The torchtext Vocab instance that maps tokens to ids.
        trainer_callback: AdditionalTrainerCallback
            The callback to convert solutions and decoding results.
        output_folder_name: str
            Folder name to save results.
        """

        super().__init__(params)
        self.trainer_callback = trainer_callback
        self.vocab = params["language_vocab"]
        if output_folder_name is not None:
            self.output_folder_name = output_folder_name
        else:
            self.output_folder_name = os.path.expanduser(os.path.expandvars(params["runner_output_folder"]))
            self.output_folder_name = os.path.join(self.output_folder_name, params["resume_session_name"])
        os.makedirs(self.output_folder_name, exist_ok=True)
        self.image_counter = 0
        self.image_id2tokens = {}

        self.aggregate_statistics = {}
        if params["use_waymo_dataset"]:
            self.horizon = [29, 49, 79]
        else:
            self.horizon = [9, 29]

    def initialize_training(self, writer):
        super().initialize_training(writer)

    def epoch_start(self, dataloader_type):
        super().epoch_start(dataloader_type)

    def iteration_update(self, data_dictionary: dict, stats_dict: dict) -> None:
        # Skip for discriminator updates.
        if "g_stats" not in stats_dict or stats_dict["g_stats"] is None:
            return

        batch_itm = data_dictionary["batch_itm"]
        offset_x = data_dictionary["offset_x"]
        offset_y = data_dictionary["offset_y"]
        scale = data_dictionary["param"]["predictor_normalization_scale"]
        is_future_valid = data_dictionary["is_future_valid"]
        # [batch_size, num_agents, num_future_steps, 2, num_samples]
        predicted_trajectories = data_dictionary["predicted_trajectories"] / scale
        # [batch_size, num_agents, num_future_steps, 2]
        expected_trajectories = data_dictionary["expected_trajectories"] / scale
        param = data_dictionary["param"]
        map_coordinates = data_dictionary["map_coordinates"]
        map_validity = data_dictionary["map_validity"]
        map_others = data_dictionary["map_others"]

        expected = {}
        # language_tokens: [batch_size, num_agents, num_tokens]
        self.trainer_callback.update_expected_results(expected, batch_itm, -1)
        predicted = {}
        self.trainer_callback.update_decoding(predicted, data_dictionary["stats_list"])

        batch_size = is_future_valid.shape[0]
        if self.params["visualize_prediction"]:
            for b in range(batch_size):
                solution = {
                    "predicted_trajectories": predicted_trajectories[b] * scale,
                    "is_future_valid": is_future_valid[b],
                }
                self.trainer_callback.update_solution(solution, predicted, b)

                # One sentence for each agent in the scenario
                instance_info = json.loads(batch_itm[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO][b])
                self.image_id2tokens[self.image_counter] = {
                    "caption_per_agent": [" ".join(tokens) for tokens in solution["predicted_tokens"]],
                    "instance_info": instance_info,
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
                    visualization_callbacks=[self.trainer_callback],
                )
                im = Image.fromarray(img)
                save_filename = os.path.join(
                    self.output_folder_name,
                    "vis_{}".format(self.image_counter) + ".png",
                )
                im.save(save_filename)
                self.image_counter += 1

        # Compute number of tokens
        token_masks = torch.where(expected["language_tokens"] > self.vocab.get_stoi()["<pad>"], 1, 0)
        # [batch_size, num_agents]
        num_tokens = torch.sum(token_masks, dim=-1)
        has_multitokens = torch.where(num_tokens > 1, True, False)

        for h in stats_dict["g_stats"]["agent_mon_fdes_partial"]:
            all_label = str(h) + "_MoN_fde"
            if all_label not in self.aggregate_statistics:
                self.aggregate_statistics[all_label] = []
            token_label = self.MULTI_TOKENS + "_" + str(h) + "_MoN_fde"
            if token_label not in self.aggregate_statistics:
                self.aggregate_statistics[token_label] = []

            fdes = stats_dict["g_stats"]["agent_mon_fdes_partial"][h].detach()
            self.aggregate_statistics[all_label].append(fdes[~fdes.isnan()].cpu().numpy())

            selected_fdes = torch.masked_select(fdes, has_multitokens)
            self.aggregate_statistics[token_label].append(selected_fdes[~selected_fdes.isnan()].cpu().numpy())

        # Collect stats for each horizon
        n_language_samples = predicted_trajectories.shape[-1]
        if self.params["compute_information_gain"]:
            n_language_samples = int(n_language_samples / 2)
        for h in self.horizon:
            entory_all = []
            ig_all = []
            for b in range(predicted_trajectories.shape[0]):
                for agent_i in range(predicted_trajectories.shape[1]):
                    predicted_trajectories_b = predicted_trajectories[b, agent_i].cpu().detach().numpy()
                    entropy_b = []
                    ig_b = []
                    for t in range(h + 1):
                        predicted_points = predicted_trajectories_b[t]
                        predicted_points_x = predicted_points[0] + np.random.normal(0, 1.0, predicted_points[0].shape)
                        predicted_points_y = predicted_points[1] + np.random.normal(0, 1.0, predicted_points[0].shape)
                        # Create Gaussion kernel.
                        values = np.vstack(
                            [predicted_points_x[:n_language_samples], predicted_points_y[:n_language_samples]]
                        )
                        kernel = scipy.stats.gaussian_kde(values)
                        p = kernel.pdf(values)
                        logp = kernel.logpdf(values)
                        entropy = 0
                        for sample_i in range(n_language_samples):
                            entropy += p[sample_i] * logp[sample_i]
                        entropy_b.append(-entropy)
                        if self.params["compute_information_gain"]:
                            values = np.vstack(
                                [predicted_points_x[n_language_samples:], predicted_points_y[n_language_samples:]]
                            )
                            kernel = scipy.stats.gaussian_kde(values)
                            p_no_lang = kernel.pdf(values)
                            logp_no_lang = kernel.logpdf(values)
                            entropy_no_lang = 0
                            for sample_i in range(n_language_samples):
                                entropy_no_lang += p_no_lang[sample_i] * logp_no_lang[sample_i]
                            ig_b.append(-entropy_no_lang - (-entropy))
                    entory_all.append(np.mean(entropy_b))
                    if len(ig_b) > 0:
                        ig_all.append(np.mean(ig_b))
            entropy_label = str(h) + "_mean_entropy"
            if entropy_label not in self.aggregate_statistics:
                self.aggregate_statistics[entropy_label] = []
            self.aggregate_statistics[entropy_label] += entory_all
            if self.params["compute_information_gain"]:
                ig_label = str(h) + "_mean_information_gain"
                if ig_label not in self.aggregate_statistics:
                    self.aggregate_statistics[ig_label] = []
                self.aggregate_statistics[ig_label] += ig_all

    def epoch_end(self, idx: int, global_batch_cnt: int, skip_visualization: bool) -> None:
        # Aggregate stats
        stats = {}
        for key in self.aggregate_statistics.keys():
            if "MoN_fde" in key:
                val = np.concatenate(self.aggregate_statistics[key])
            else:
                val = self.aggregate_statistics[key]
            stats[key] = float(np.mean(val))

        for s in stats:
            print(s, stats[s])

        # Save the statistics dictionary
        json_filename = os.path.join(self.output_folder_name, "stats.json")
        print("Saving statistics to {}".format(json_filename))
        with open(json_filename, "w") as fp:
            json.dump(stats, fp, indent=2)
        # Save image id to predicted sentences
        json_filename = os.path.join(self.output_folder_name, "image2caption.json")
        print("Saving image to caption mapping to {}".format(json_filename))
        with open(json_filename, "w") as fp:
            json.dump(self.image_id2tokens, fp, indent=2)
        return None
