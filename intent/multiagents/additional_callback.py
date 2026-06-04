import abc
from abc import ABC
from typing import Dict, Optional, Tuple, Union

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from intent.multiagents import trainer_visualization

MatplotlibColor = Union[str, Tuple[float, float, float]]


class AdditionalModelCallback(ABC):
    def __init__(self) -> None:
        """A generic interface to handle additional structure in prediction model."""

    def update_costs(
        self,
        additional_stats: dict,
        params: dict,
        predicted: dict,
        expected: dict,
        future_encoding: torch.Tensor,
        extra_context: dict,
    ) -> None:
        """Compute the cross-entropy loss of the generated token sequence.

        Parameters
        ----------
        additional_stats : dict
            A dictionary of the additional costs.
        params : dict
            A parameters dictionary.
        predicted : dict
            A dictionary of predicted values
        expected : dict
            A dictionary of expected values.
        future_encoding : dict
            A dictionary with additional encoding information for the future prediction.
        extra_context : dict
            A dictionary that provides extra context for computing costs.
        """

    def augment_sample_index(self, trainer_param: dict, sample_indices: list):
        """
        Callback function to add additional sample index.

        Parameters
        ----------
        trainer_param : dict
            Dictionary of trainer parameters.
        sample_indices : list
            List of sample indices.
        """

    def update_stats_list(self, trainer_param: dict, stats_list: list):
        """
        Callback function to update stats list.

        Parameters
        ----------
        trainer_param : dict
            Dictionary of trainer parameters.
        stats_list : list
            List of prediction statistics.
        """

    def update_model_stats(
        self,
        param: dict,
        stats: list,
        num_samples: int,
        num_pred_timesteps: int,
        predicted_trajectories_scene: torch.Tensor,
        expected_trajectories_scene: torch.Tensor,
        timestamps: torch.Tensor,
        prediction_timestamp: torch.Tensor,
        relevant_agents: torch.Tensor,
        is_future_valid: torch.Tensor,
        err_horizons_timepoints: list,
        agent_types: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Callback function to update model statistics.

        Parameters
        ----------
        param : dict
            Dictionary of trainer parameters.
        stats : dict
            List of stats per future time step.
        num_samples : int
            Number of samples from prediction.
        num_pred_timesteps : int
            Number of prediction time steps.
        predicted_trajectories_scene : torch.Tensor
            Predicted trajectories.
        expected_trajectories_scene : torch.Tensor
            Expected trajectories.
        timestamps : torch.Tensor
            Timestamps.
        prediction_timestamp : torch.Tensor
            Prediction timestamp.
        relevant_agents : torch.Tensor
            Relevant agents.
        is_future_valid : torch.Tensor
            A mask indicating validity of future steps.
        err_horizons_timepoints : list
            A list of timepoints containing the ADE/FDE computation lookahead horizon.
        agent_types : torch.Tensor
            One-hot agent type information, (batch_size x num_agents x num_types).

        Returns
        -------
        stats : dict
            Updated list of stats per future time step.
        """
        return num_samples

    def update_data_cost(self, param: dict, additional_stats: dict, data_cost: torch.Tensor) -> torch.Tensor:
        """
        Callback function to update generator cost.

        Parameters
        ----------
        param : dict
            Dictionary of trainer parameters.
        additional_stats : dict
            Dictionary of additional statistics.
        data_cost : torch.Tensor
            Data cost.

        Returns
        -------
        data_cost : torch.Tensor
            Updated data cost.
        """
        return data_cost

    def update_agent_additional_inputs(
        self, batch, agent_additional_inputs: dict, transforms: torch.Tensor, normalizer
    ):
        """
        Callback function to update agent additional inputs.

        Parameters
        ----------
        agent_additional_inputs : dict
            Dictionary of agent additional inputs.
        transforms : torch.Tensor
            Agent-centric transforms.
        normalizer
            Normalizer function.
        """
        pass

    def update_prediction_stats(
        self, param: dict, stats: list, is_valid: torch.Tensor, sample_size: int, transforms: torch.Tensor, normalizer
    ):
        """
        Callback function to update prediction stats.

        Parameters
        ----------
        param : dict
            Dictionary of trainer parameters.
        stats : []
            A list of statistics.
        is_valid : torch.Tensor
            Validity of positions.
        sample_size : int
            Number of samples.
        transforms : torch.Tensor
            Agent-centric transforms.
        normalizer
            Normalizer function.
        """
        pass


class AdditionalTrainerCallback(ABC):
    def __init__(self, params: dict) -> None:
        """A generic interface to handle additional structure in training script.

        Parameters
        ----------
        params: dict
            A dictionary of training parameters.
        """
        self.params = params

    def get_additional_dataloader_info(self, ctx, dataloader_info, other_inputs):
        pass

    def update_statistic_keys(self, statistic_keys: list) -> None:
        """
        Callback function to add additional keys to the last of statistic_keys.

        Parameters
        ----------
        statistic_keys : list
            Statistic keys.
        """
        pass

    def set_tqdm_iter_description(
        self, tqdm_iter: tqdm.std.tqdm, iter: int, avg_g_cost: float, statistics: dict, trainer_param: dict
    ) -> None:
        """
        Callback function to overwrite tqdm iter description.

        Parameters
        ----------
        tqdm_iter : tqdm.std.tqdm
            tqdm iterator.
        iter : int
            Training iterator count.
        avg_g_cost : float
            Average generator cost.
        statistics : dict
            Dictionary of statistics.
        trainer_param : dict
            Dictionary of trainer parameters.
        """
        pass

    def visualize_hybrid_callback(
        self,
        batch_itm: dict,
        predicted: dict,
        num_past_timepoints: int,
        stats: dict,
        batch_idx: int,
        visualization_idx: int,
        writer,
    ) -> None:
        """
        Callback function to visualize additional results.

        Parameters
        ----------
        batch_itm : dict
            Dictionary of batch items.
        predicted : dict
            Dictionary of predictions.
        num_past_timepoints : int
            Number of past time steps.
        stats : dict
            Dictionary of prediction stats.
        batch_idx : int
            Batch index.
        visualization_idx : int
            Visualization index.
        writer :
            Tensorboard writer.
        """
        pass

    def update_batch_item_before_training(self, ctx, batch_item, other_inputs):
        pass

    def update_solution(self, solution: dict, predicted: dict, batch_idx: int, batch_itm: dict = None):
        """
        Callback function to add additional states to solution.

        Parameters
        ----------
        solution : dict
            Dictionary of solution.
        predicted : dict
            Dictionary of prediction.
        batch_idx : int
            Batch index.
        batch_itm : dict
            Dictionary containing the batch item.
        """
        pass

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
        pass

    def update_additional_inputs(self, additional_inputs: dict, batch_itm: dict, num_past_timepoints: int):
        """
        Callback function to add additional info to additional_inputs.

        Parameters
        ----------
        additional_inputs : dict
            Dictionary of additional inputs.
        batch_itm : dict
            Dictionary of batch item.
        num_past_timepoints : int
            Number of past time steps.
        """
        pass

    def update_agent_additional_inputs(
        self,
        agent_additional_inputs: dict,
        batch_positions_tensor: torch.Tensor,
        batch_is_valid: torch.Tensor,
        offset_x: torch.Tensor,
        offset_y: torch.Tensor,
        num_past_timepoints: int,
    ):
        """
        Callback function to add discrete labels (i.e. maneuvers) as additional inputs.

        Parameters
        ----------
        agent_additional_inputs : dict
            Dictionary of agent additional inputs.
        batch_positions_tensor : torch.Tensor
            Row positions tensor.
        batch_is_valid : torch.Tensor
            Position validity tensor.
        offset_x : torch.Tensor
            Global offset x vector.
        offset_y : torch.Tensor
            Global offset y vector.
        num_past_timepoints : int
            Number of past time steps.
        """
        pass

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
        pass

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
        return text

    def update_visualization_agent_color(
        self, agent_color_predicted: MatplotlibColor, param: dict, sample_i: int, solution: dict
    ) -> MatplotlibColor:
        """
        Update visualization text.

        Parameters
        ----------
        agent_color_predicted : str or Tuple
            Color for predicted agent trajectory.
        param : dict
            Parameters.
        sample_i : int
            Sample index.
        solution : dict
            Solution dictionary.
        """
        return agent_color_predicted

    def visualize_agent_additional_info(
        self,
        agent_id: int,
        is_future_valid: torch.Tensor,
        sample_i: int,
        solution: dict,
        ax: plt.Axes,
        predicted_x: Union[torch.Tensor, np.ndarray],
        predicted_y: Union[torch.Tensor, np.ndarray],
        agent_color_predicted: MatplotlibColor,
    ):
        """
        Visualize additional info to an agent's trajectory.

        Parameters
        ----------
        agent_id : int
            Agent index.
        is_future_valid : torch.Tensor
            Validity of future time steps.
        sample_i : int
            Sample index.
        solution : dict
            Solution dictionary.
        ax : matplotlib.axes._axes.Axes
            Plot ax.
        predicted_x : torch.Tensor
            Predicted x coordinates.
        predicted_y : torch.Tensor
            Predicted y coordinates.
        agent_color_predicted : str or tuple
            Color for predicted agent trajectory.
        """
        pass

    def visualize_additional_info(self, solution: dict, fig: matplotlib.figure.Figure, ax, batch, batch_idx):
        """Visualize additional info of a solution.

        Parameters
        ----------
        solution: dict
            Solution dictionary.
        fig: matplotlib.figure.Figure
            The figure instance.
        """
        pass

    def epoch_update(self, epoch: int, params: dict):
        """This callback function makes an update per epoch, an update may include
        freezing or unfreezing parameters of layers if needed.

        epoch: int
            The current epoch.
        params: dict
            The dictionary of parameters.
        """
        pass

    def update_statistics(
        self,
        statistic_keys: list,
        statistics: dict,
        logger,
        dataloader_type: str,
        global_batch_cnt: int,
    ):
        """
        Callback function to add additional info to trainer statistics.

        Parameters
        ----------
        statistic_keys : []
            List of statistics keys.
        statistics : dict
            Dictionary of statistics.
        logger
            Training logger.
        dataloader_type : str
            Dataloader type.
        global_batch_cnt : int
            Global batch count.
        """
        pass


class CustomVisualizationCallback(ABC):
    """
    A class to handle generation of custom images

    """

    def __init__(self, params):
        self.params = params

    def visualize_custom_process(
        self,
        batch_itm_idx,
        batch_size,
        batch_itm,
        result,
        g_stats,
        cost,
        param,
        summary_writer,
        iter,
        tag_prefix,
        predicted_semantics,
        visualization_callbacks,
    ):
        """
        Callback function to plot and log custom images
        Parameters
        ----------
        batch_itm_idx
        batch_size
        batch_itm
        result
        g_stats
        cost
        param
        summary_writer
        iter
        tag_prefix
        predicted_semantics
        visualization_callbacks

        Returns
        -------

        """

        fig_handle_dict = self.make_custom_fig(
            batch_itm_idx,
            batch_size,
            batch_itm,
            result,
            g_stats,
            cost,
            param,
            summary_writer,
            iter,
            tag_prefix,
            predicted_semantics,
            visualization_callbacks,
        )

        self.log_fig(summary_writer, fig_handle_dict, param, tag_prefix, iter)

    @abc.abstractmethod
    def make_custom_fig(
        self,
        batch_itm_idx,
        batch_size,
        batch_itm,
        result,
        g_stats,
        cost,
        param,
        summary_writer,
        iter,
        tag_prefix,
        predicted_semantics,
        visualization_callbacks,
    ) -> Dict[str, "fig"]:
        """
        Callback function to generate and return the figure to plot

        Parameters
        ----------
        batch_itm_idx
        batch_size
        batch_itm
        result
        g_stats
        cost
        param
        summary_writer
        iter
        tag_prefix
        predicted_semantics
        visualization_callbacks

        Returns
        -------
        Dictionary mapping logger keys -> matplotlib.figure.Figurec handle
        """
        pass

    def log_fig(self, summary_writer, fig_handle_dict, param, tag_prefix, iter):
        """
        Callback function to log the custom figures as an image.

        Parameters
        ----------
        summary_writer
        fig_handle_dict
        param
        iter

        Returns
        -------

        """
        if fig_handle_dict is None:
            return
        for fig_name, fig in fig_handle_dict.items():
            img = trainer_visualization.get_img_from_fig(fig, image_format=param["visualization_image_format"])
            plt.close(fig)
            summary_writer.add_image(f"{tag_prefix}/{fig_name}", img.transpose(2, 0, 1), global_step=iter)
