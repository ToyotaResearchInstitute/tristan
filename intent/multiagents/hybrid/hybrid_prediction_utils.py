from typing import Optional, Union

import matplotlib.axes
import numpy as np
import torch
import tqdm

from data_sources.argoverse.argoverse_hybrid_utils import MODE_DICT_IDX_TO_MANEUVER, MODE_DICT_IDX_TO_MANEUVER_ONLINE
from intent.multiagents.additional_callback import AdditionalModelCallback, AdditionalTrainerCallback, MatplotlibColor
from intent.multiagents.additional_costs import AdditionalCostCallback
from intent.multiagents.trainer_visualization import visualize_mode_sequences
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from util.prediction_metrics import displacement_errors


def subsample_hybrid_random(
    predicted_trajectories: torch.Tensor, predicted_modes: torch.Tensor, discrete_weights: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    Subsample predictions randomly based on log likelihoods.

    Parameters
    ----------
    predicted_trajectories : torch.Tensor
        predicted trajectories with shape [batch_size, num_agent, num_future_steps, 2, num_sample_size].
    predicted_modes : torch.Tensor
        predicted modes with shape [batch_size, num_agent, num_future_steps, mode_size, num_sample_size].
    discrete_weights : torch.Tensor
        discrete weights with shape [batch_size, num_agent, num_sample_size].
    params : dict
        dictionary of model parameters.

    Returns
    -------
    top_indices_total : torch.Tensor
        Subsampled prediction indices.
    """
    subsample_size = params["hybrid_runner_subsample_size"]
    batch_size, num_agents, num_steps, _, sample_size = list(predicted_trajectories.shape)
    top_indices_total = []
    for b in range(batch_size):
        top_indices_b = []
        # Randomly sample prediction indices given weight logits.
        for i in range(num_agents):
            discrete_weights_b_i = discrete_weights[b, i]
            discrete_prob = torch.exp(discrete_weights_b_i)
            discrete_prob = discrete_prob / torch.sum(discrete_prob)

            discrete_prob_np = discrete_prob.cpu().detach().numpy()
            top_indices = np.random.choice(
                discrete_prob_np.shape[0], size=subsample_size, replace=True, p=discrete_prob_np
            )
            top_indices = torch.tensor(top_indices).to(discrete_prob.device)
            top_indices_b.append(top_indices)

        top_indices_b = torch.stack(top_indices_b, 0)
        top_indices_total.append(top_indices_b)

    # [batch_size, num_agents, num_subsamples].
    top_indices_total = torch.stack(top_indices_total, 0)
    return top_indices_total


def subsample_hybrid_top_k(
    predicted_trajectories: torch.Tensor, predicted_modes: torch.Tensor, discrete_weights: torch.Tensor, params: dict
) -> torch.Tensor:
    """
    Subsample most likely predictions based on log likelihoods.

    Parameters
    ----------
    predicted_trajectories : torch.Tensor
        predicted trajectories with shape [batch_size, num_agent, num_future_steps, 2, num_sample_size].
    predicted_modes : torch.Tensor
        predicted modes with shape [batch_size, num_agent, num_future_steps, mode_size, num_sample_size].
    discrete_weights : torch.Tensor
        discrete weights with shape [batch_size, num_agent, num_sample_size].
    params : dict
        dictionary of model parameters.

    Returns
    -------
    top_indices_total : torch.Tensor
        Subsampled prediction indices.
    """
    subsample_size = params["hybrid_runner_subsample_size"]
    batch_size, num_agents, num_steps, _, sample_size = list(predicted_trajectories.shape)
    top_indices_total = []
    for b in range(batch_size):
        top_indices_b = []
        # Select most likely unique samples from predictions.
        for i in range(num_agents):
            discrete_weights_b_i = discrete_weights[b, i]

            # Get unique samples based on weights (assume different weights -> different samples).
            neg_discrete_weights_b_i_unique, indices_raw = torch.unique(-discrete_weights_b_i, return_inverse=True)
            perm = torch.arange(indices_raw.size(0), dtype=indices_raw.dtype, device=indices_raw.device)
            indices, perm = indices_raw.flip([0]), perm.flip([0])
            unique_indices = indices.new_empty(neg_discrete_weights_b_i_unique.size(0)).scatter_(0, indices, perm)

            # Verify unique indices are correct.
            discrete_weights_b_i_unique = discrete_weights_b_i[unique_indices]
            assert (
                discrete_weights_b_i_unique == -neg_discrete_weights_b_i_unique
            ).all(), "Unique function not working."

            # Get top k indices based on log likelihood.
            top_indices = unique_indices[:subsample_size]

            # If there is not enough samples, add the top sample.
            selected_sample_size = top_indices.shape[0]
            if selected_sample_size < subsample_size:
                num_samples_to_fill = subsample_size - selected_sample_size
                top_indices = torch.cat([top_indices[:1].repeat(num_samples_to_fill), top_indices], 0)
            top_indices_b.append(top_indices)

        top_indices_b = torch.stack(top_indices_b, 0)
        top_indices_total.append(top_indices_b)

    # [batch_size, num_agents, num_subsamples].
    top_indices_total = torch.stack(top_indices_total, 0)
    return top_indices_total


def subsample_hybrid_fps(
    predicted_trajectories: torch.Tensor,
    predicted_modes: torch.Tensor,
    discrete_weights: torch.Tensor,
    params: dict,
    dist_type: str = "final",
) -> torch.Tensor:
    """
    Subsample predictions using fartherst point sampling.

    Parameters
    ----------
    predicted_trajectories : torch.Tensor
        predicted trajectories with shape [batch_size, num_agent, num_future_steps, 2, num_sample_size].
    predicted_modes : torch.Tensor
        predicted modes with shape [batch_size, num_agent, num_future_steps, mode_size, num_sample_size].
    discrete_weights : torch.Tensor
        discrete weights with shape [batch_size, num_agent, num_sample_size].
    params : dict
        dictionary of model parameters.
    dist_type : str
        definition of distance used in FPS.

    Returns
    -------
    top_indices_total : torch.Tensor
        Subsampled prediction indices.
    """
    subsample_size = params["hybrid_runner_subsample_size"]
    batch_size, num_agents, num_steps, _, sample_size = list(predicted_trajectories.shape)
    top_indices_total = []
    for b in range(batch_size):
        top_indices_b = []
        for i in range(num_agents):
            predicted_trajectories_b_i = predicted_trajectories[b, i]
            discrete_weights_b_i = discrete_weights[b, i]

            # Get unique samples based on weights (assume different weights -> different samples).
            neg_discrete_weights_b_i_unique, indices_raw = torch.unique(-discrete_weights_b_i, return_inverse=True)
            perm = torch.arange(indices_raw.size(0), dtype=indices_raw.dtype, device=indices_raw.device)
            indices, perm = indices_raw.flip([0]), perm.flip([0])
            unique_indices = indices.new_empty(neg_discrete_weights_b_i_unique.size(0)).scatter_(0, indices, perm)

            # Verify unique indices are correct.
            discrete_weights_b_i_unique = discrete_weights_b_i[unique_indices]
            assert (
                discrete_weights_b_i_unique == -neg_discrete_weights_b_i_unique
            ).all(), "Unique function not working."
            predicted_trajectories_b_i_unique = predicted_trajectories_b_i[:, :, unique_indices]

            num_unique_samples = unique_indices.shape[0]
            # Get all samples if there are not enough samples.
            if num_unique_samples <= subsample_size:
                num_samples_to_fill = subsample_size - num_unique_samples
                top_indices = torch.cat([unique_indices[:1].repeat(num_samples_to_fill), unique_indices], 0)

            # Perform FPS to find samples.
            else:
                predicted_trajectories_b_i_unique = predicted_trajectories_b_i_unique.permute(2, 0, 1)
                predicted_trajectories_b_i_diff = predicted_trajectories_b_i_unique.unsqueeze(
                    0
                ) - predicted_trajectories_b_i_unique.unsqueeze(1)

                predicted_trajectories_b_i_end_dist = (torch.sqrt((predicted_trajectories_b_i_diff**2).sum(-1)))[
                    ..., -1
                ]
                predicted_trajectories_b_i_avg_dist = (torch.sqrt((predicted_trajectories_b_i_diff**2).sum(-1))).mean(
                    -1
                )

                # Compute FPS distance based on end point.
                if dist_type == "final":
                    dist = predicted_trajectories_b_i_end_dist
                # Compute FPS distance based on average distance.
                else:
                    dist = predicted_trajectories_b_i_avg_dist

                fps_indices = torch.zeros((1), dtype=torch.int64).to(predicted_trajectories_b_i_diff.device)
                for i in range(subsample_size - 1):
                    fps_dist = dist[fps_indices]
                    fps_dist = torch.min(fps_dist, 0, keepdim=True)[0]
                    fps_index = torch.argmax(fps_dist, -1)
                    fps_indices = torch.cat([fps_indices, fps_index], 0)

                top_indices = unique_indices[fps_indices]
            top_indices_b.append(top_indices)

        top_indices_b = torch.stack(top_indices_b, 0)
        top_indices_total.append(top_indices_b)

    # [batch_size, num_agents, num_subsamples].
    top_indices_total = torch.stack(top_indices_total, 0)
    return top_indices_total


def subsample_hybrid_nms(
    predicted_trajectories: torch.Tensor,
    predicted_modes: torch.Tensor,
    discrete_weights: torch.Tensor,
    params: dict,
    dist_threshold: float = 2.0,
    dist_type: str = "final",
) -> torch.Tensor:
    """
    Subsample predictions using non maximum suppression.

    Parameters
    ----------
    predicted_trajectories : torch.Tensor
        predicted trajectories with shape [batch_size, num_agent, num_future_steps, 2, num_sample_size].
    predicted_modes : torch.Tensor
        predicted modes with shape [batch_size, num_agent, num_future_steps, mode_size, num_sample_size].
    discrete_weights : torch.Tensor
        discrete weights with shape [batch_size, num_agent, num_sample_size].
    params : dict
        dictionary of model parameters.
    dist_threshold : float
        distance threshold to ignore nearby samples.
    dist_type : str
        definition of distance used in NMS.

    Returns
    -------
    top_indices_total : torch.Tensor
        Subsampled prediction indices.
    """
    subsample_size = params["hybrid_runner_subsample_size"]
    batch_size, num_agents, num_steps, _, sample_size = list(predicted_trajectories.shape)
    top_indices_total = []
    for b in range(batch_size):
        top_indices_b = []
        for i in range(num_agents):
            predicted_trajectories_b_i = predicted_trajectories[b, i]
            discrete_weights_b_i = discrete_weights[b, i]

            # Get unique samples based on weights (assume different weights -> different samples).
            neg_discrete_weights_b_i_unique, indices_raw = torch.unique(-discrete_weights_b_i, return_inverse=True)
            perm = torch.arange(indices_raw.size(0), dtype=indices_raw.dtype, device=indices_raw.device)
            indices, perm = indices_raw.flip([0]), perm.flip([0])
            unique_indices = indices.new_empty(neg_discrete_weights_b_i_unique.size(0)).scatter_(0, indices, perm)

            # Verify unique indices are correct.
            discrete_weights_b_i_unique = discrete_weights_b_i[unique_indices]
            assert (
                discrete_weights_b_i_unique == -neg_discrete_weights_b_i_unique
            ).all(), "Unique function not working."
            predicted_trajectories_b_i_unique = predicted_trajectories_b_i[:, :, unique_indices]

            num_unique_samples = unique_indices.shape[0]
            # Get all samples if there are not enough samples.
            if num_unique_samples <= subsample_size:
                num_samples_to_fill = subsample_size - num_unique_samples
                top_indices = torch.cat([unique_indices[:1].repeat(num_samples_to_fill), unique_indices], 0)

            # Perform NMS to find samples.
            else:
                predicted_trajectories_b_i_unique = predicted_trajectories_b_i_unique.permute(2, 0, 1)
                predicted_trajectories_b_i_diff = predicted_trajectories_b_i_unique.unsqueeze(
                    0
                ) - predicted_trajectories_b_i_unique.unsqueeze(1)

                predicted_trajectories_b_i_end_dist = (torch.sqrt((predicted_trajectories_b_i_diff**2).sum(-1)))[
                    ..., -1
                ]
                predicted_trajectories_b_i_avg_dist = (torch.sqrt((predicted_trajectories_b_i_diff**2).sum(-1))).mean(
                    -1
                )

                # Compute NMS distance based on end point.
                if dist_type == "final":
                    dist = predicted_trajectories_b_i_end_dist
                # Compute NMS distance based on average distance.
                else:
                    dist = predicted_trajectories_b_i_avg_dist

                nms_indices = torch.zeros((1), dtype=torch.int64).to(predicted_trajectories_b_i_diff.device)
                nms_all_indices = torch.arange(dist.shape[0]).to(predicted_trajectories_b_i_diff.device)
                for i in range(subsample_size - 1):
                    nms_dist = dist[nms_indices]
                    nms_dist = torch.min(nms_dist, 0, keepdim=True)[0]
                    nms_dist_mask = nms_dist > dist_threshold
                    nms_valid = torch.masked_select(nms_all_indices, nms_dist_mask)

                    # Break if we have exhausted all samples.
                    if nms_valid.shape[0] == 0:
                        break
                    nms_indices = torch.cat([nms_indices, nms_valid[:1]], 0)

                if nms_indices.shape[0] < subsample_size:
                    try:
                        # Add top samples if there are no enough samples.
                        nms_indices_mask = torch.ones(dist.shape[0], dtype=torch.bool)
                        nms_indices_mask[nms_indices] = False
                        nms_indices_left = nms_all_indices[nms_indices_mask]

                        # Sample randomly based on sample probs.
                        discrete_weights_p = torch.exp(discrete_weights_b_i)
                        discrete_weights_p_unique = discrete_weights_p[unique_indices]
                        discrete_weights_p_left = discrete_weights_p_unique[nms_indices_mask]
                        discrete_weights_p_left = discrete_weights_p_left / torch.sum(discrete_weights_p_left)
                        discrete_weights_p_np = discrete_weights_p_left.cpu().detach().numpy()

                        top_indices = np.random.choice(
                            nms_indices_left.cpu().detach().numpy(),
                            size=subsample_size - nms_indices.shape[0],
                            replace=True,
                            p=discrete_weights_p_np,
                        )
                        top_indices = torch.tensor(top_indices).to(discrete_weights_p.device)

                        nms_indices = torch.cat([nms_indices, top_indices])
                    except:
                        import IPython

                        IPython.embed()

                top_indices = unique_indices[nms_indices]
            top_indices_b.append(top_indices)

        top_indices_b = torch.stack(top_indices_b, 0)
        top_indices_total.append(top_indices_b)

    # [batch_size, num_agents, num_subsamples].
    top_indices_total = torch.stack(top_indices_total, 0)
    return top_indices_total


def hybrid_label_smooth_filter(input: np.array) -> np.array:
    """
    Smooth a data sequence so that there is no label that is different from both neighbors.

    Parameters
    ----------
    input : np.array
        Input sequence.

    Returns
    -------
    smoothed_input : np.array
        Smoothed input sequence.
    """
    smoothed_input = np.zeros_like(input)
    smoothed_input[0] = input[0]
    for k in range(1, input.shape[0] - 1):
        # If a value is different from both of its neighbors, set to previous value.
        if input[k] != smoothed_input[k - 1] and input[k] != input[k + 1]:
            if k < input.shape[0] - 2:
                if input[k] == input[k + 2]:
                    smoothed_input[k] = input[k]
                else:
                    smoothed_input[k] = smoothed_input[k - 1]
            else:
                smoothed_input[k] = smoothed_input[k - 1]
        else:
            smoothed_input[k] = input[k]
    smoothed_input[-1] = input[-1]
    return smoothed_input


class HybridGenerationCost(AdditionalCostCallback):
    def __init__(self):
        """Module for computing the supervised/semi-supervised loss for the hybrid generator."""
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
        if params["discrete_supervised"]:
            self.update_supervised_costs(additional_stats, extra_context)
        else:
            # This is NOT supported.
            self.update_unsupervised_costs(additional_stats, extra_context, params)

    def update_supervised_costs(self, additional_stats: dict, extra_context: dict) -> None:
        """
        Compute supervised loss given known discrete labels.

        Parameters
        ----------
        additional_stats : dict
            Dictionary of additional stats.
        extra_context : dict
            Extra context from predictor.
        """
        stats = extra_context["stats"]

        # Add continuous prediction loss.
        additional_stats["hybrid_error_supervised_continuous"] = stats[-1]["discrete_info"]["sqr_error_gt"]
        # Add discrete prediction loss.
        additional_stats["hybrid_error_supervised_discrete"] = -1.0 * stats[-1]["gt_sample_log_weight"].mean(-1)

        # Add regularization loss on proposal/transition logits.
        # Skip the last stats as it comes from ground truth.
        proposal_logits = [stat["proposal_logits_total"] for stat in stats[:-1]]
        # [batch_size, num_agents, num_future_steps, discrete_domain_size, sample_size]
        proposal_logits = torch.stack(proposal_logits, -1)
        additional_stats["hybrid_error_proposal_reg"] = (proposal_logits**2).mean(-1).mean(-1).mean(-1).mean(-1)

        # Add regularization on transitions between modes.
        discrete_samples = [stat["discrete_samples"] for stat in stats[:-1]]
        # [batch_size, num_agents, num_future_steps, discrete_domain_size, sample_size]
        discrete_samples = torch.stack(discrete_samples, -1)
        discrete_transitions = (discrete_samples[:, :, 1:] - discrete_samples[:, :, :-1]) ** 2
        additional_stats["hybrid_error_transition_reg"] = discrete_transitions.sum(3).mean(2).mean(-1).mean(-1)

    def update_unsupervised_costs(self, additional_stats: dict, extra_context: dict, params: dict) -> None:
        """
        Compute unsupervised loss given known discrete labels.

        Parameters
        ----------
        additional_stats : dict
            Dictionary of additional stats.
        extra_context : dict
            Extra context from predictor.
        params : dict
            Trainer/model parameters.
        """
        stats = extra_context["stats"]
        use_empirical_dist = params["use_empirical_distribution"]
        sample_weight = stats[-1]["discrete_info"]["sample_weight_T"]

        if use_empirical_dist:
            # Multiple sample probabilities by proposal weight.
            proposal_weight = sample_weight[:, 0]
            log_p_obs = -proposal_weight * extra_context["sqr_errs"]
            log_p_total = log_p_obs

            # Compute EM likelihood per sample per batch.
            em_likelihood = log_p_total
        else:
            # Compute probability of mode sequence for the first agent.
            log_p_prior = sample_weight[:, 0]
            # Assume fixed sigma for observation function.
            sigma = params["hybrid_observation_sigma"]
            epsilon = 1e-40
            p_obs = (
                torch.exp(-0.5 * (extra_context["sqr_errs"] * params["future_timesteps"]) / (sigma**2))
                / (sigma * np.sqrt(2 * np.pi))
                + epsilon
            )
            log_p_obs = torch.log(p_obs)
            log_p_total = log_p_prior + log_p_obs

            posterior = p_obs
            posterior = posterior / posterior.sum(dim=-1, keepdim=True)

            # Compute EM likelihood per sample per batch.
            em_likelihood = posterior.detach() * log_p_total

        # Compute negative EM likelihood loss.
        em_loss = -1.0 * torch.sum(em_likelihood, dim=-1)
        additional_stats["hybrid_error_em"] = em_loss

        # Add discrete bootstrap loss if it exists.
        if "gt_sample_log_weight" in stats[-1]:
            additional_stats["hybrid_error_bootstrap"] = -1.0 * stats[-1]["gt_sample_log_weight"][:, 0]


class HybridModelCallback(AdditionalModelCallback):
    def __init__(self):
        """Module for handling additional hybrid callbacks in the model."""
        super().__init__()

    def augment_sample_index(self, param: dict, sample_indices: list):
        """
        Callback function to add additional sample index.

        Parameters
        ----------
        param : dict
            Dictionary of trainer parameters.
        sample_indices : list
            List of sample indices.
        """
        # In supervised training, add a special sample to provide ground truth future mode to compute loss.
        if param["discrete_supervised"]:
            sample_indices.append(-1)

    def update_stats_list(self, param: dict, stats_list: list):
        """
        Callback function to update stats list.

        Parameters
        ----------
        param : dict
            Dictionary of trainer parameters.
        stats_list : list
            List of prediction statistics.
        """
        if param["discrete_supervised"]:
            # Pass stats.
            stats_gt = stats_list[-1]
            for stats in stats_list:
                stats["gt_sample_log_weight"] = stats_gt["gt_sample_log_weight"]

    def update_model_stats(
        self,
        param: dict,
        stats: list,
        num_samples: int,
        num_pred_timesteps: int,
        predicted_trajectories: torch.Tensor,
        expected_trajectories: torch.Tensor,
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
        predicted_trajectories : torch.Tensor
            Predicted trajectories.
        expected_trajectories : torch.Tensor
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
        num_samples : int
            Updated number of samples from prediction.
        """
        # Obtain sample weights if it exists.
        if "discrete_samples_log_weight" in stats[0]:
            log_probs = [s["discrete_samples_log_weight"] for s in stats]
            log_prob = torch.stack(log_probs, -1)
            sample_weight_T = torch.exp(log_prob) + 1e-80
        else:
            sample_weight_T = None

        # For supervised hybrid prediction, compute error of the last predicted trajectory predicted from ground truth future mode.
        if param["discrete_supervised"]:
            assert (
                param["MoN_number_samples"] == num_samples - 1
            ), "Sample size mismatch during supervised discrete training."

            displ_errors_gt = displacement_errors(
                predicted_trajectories[..., 0:2, -1],
                expected_trajectories[:, :, :, 0:2],
                timestamps[:, -num_pred_timesteps:],
                prediction_timestamp,
                relevant_agents,
                is_future_valid,
                err_horizons_timepoints,
                param["miss_thresholds"],
                param,
                agent_types=agent_types,
            )
            # Compute the squared error of the trajectory predicted using ground truth mode.
            sqr_error_gt = displ_errors_gt["square_error"]
            # Use the correct number of samples.
            num_samples = num_samples - 1
        else:
            sqr_error_gt = None

        # Compute additional stats.
        discrete_info = {
            "sqr_error_gt": sqr_error_gt,
            "sample_weight_T": sample_weight_T,
        }
        stats[-1]["discrete_info"] = discrete_info
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
        if param["discrete_supervised"]:
            # For supervised training, we include a discrete supervised loss and a continuous one.
            hybrid_loss_names = ["hybrid_error_supervised_continuous", "hybrid_error_supervised_discrete"]

            # If proposal function is used, add regularization loss on its magnitude.
            if param["learn_discrete_proposal"]:
                hybrid_loss_names.append("hybrid_error_proposal_reg")

        else:
            # For unsupervised training, use EM loss.
            hybrid_loss_names = ["hybrid_error_em"]

        for hybrid_loss_name in hybrid_loss_names:
            data_cost += additional_stats[hybrid_loss_name] * param["discrete_term_coeff"]

        # In cases where we do not include discrete loss, we add an option to regularize transition functions.
        if param["discrete_term_coeff"] == 0 and "hybrid_error_proposal_reg" in additional_stats:
            data_cost += additional_stats["hybrid_error_proposal_reg"] * param["discrete_reg_term_coeff"]

        # Add regularization on transition.
        if param["discrete_transition_coeff"] > 0 and "hybrid_error_transition_reg" in additional_stats:
            data_cost += additional_stats["hybrid_error_transition_reg"] * param["discrete_transition_coeff"]

        return data_cost


class HybridTrainerCallback(AdditionalTrainerCallback):
    def __init__(self, params):
        """Module for handling additional hybrid callbacks in the trainer script."""
        super().__init__(params)

    def update_statistic_keys(self, statistic_keys: list) -> None:
        """
        Callback function to add hybrid keys to the last of statistic_keys.

        Parameters
        ----------
        statistic_keys : list
            Statistic keys.
        """
        # Stats for hybrid-related errors.
        statistic_keys.append("hybrid_error_em")
        statistic_keys.append("hybrid_error_bootstrap")
        statistic_keys.append("hybrid_error_supervised_continuous")
        statistic_keys.append("hybrid_error_supervised_discrete")
        statistic_keys.append("hybrid_error_proposal_reg")
        statistic_keys.append("hybrid_error_transition_reg")

    def set_tqdm_iter_description(
        self, tqdm_iter: tqdm.std.tqdm, iter: int, avg_g_cost: float, statistics: dict, trainer_param: dict
    ) -> None:
        """
        Callback function to set tqdm iter description with discrete errors, based on supervised vs. unsupervised training.

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
        # Set description to include supervised errors in supervised training.
        if "discrete_supervised" in trainer_param and trainer_param["discrete_supervised"]:
            tqdm_iter.set_description(
                "iter {}, g_cost {}, l2_error {:.2f}, ADE_error {:.2f}, MoN_ADE_error {:.2f}, discrete_error_c {:.2f}, discrete_error_d {:.2f}".format(
                    iter,
                    avg_g_cost,
                    np.mean(np.concatenate(statistics["l2_error"])),
                    np.mean(np.concatenate(statistics["ade_error"])),
                    np.mean(np.concatenate(statistics["MoN_ade_error"])),
                    np.mean(np.concatenate(statistics["hybrid_error_supervised_continuous"])),
                    np.mean(np.concatenate(statistics["hybrid_error_supervised_discrete"])),
                )
            )
        # Set description to include EM losses during unsupervised training.
        else:
            tqdm_iter.set_description(
                "iter {}, g_cost {}, l2_error {:.2f}, ADE_error {:.2f}, MoN_ADE_error {:.2f}, discrete_error (EM) {:.2f}".format(
                    iter,
                    avg_g_cost,
                    np.mean(np.concatenate(statistics["l2_error"])),
                    np.mean(np.concatenate(statistics["ade_error"])),
                    np.mean(np.concatenate(statistics["MoN_ade_error"])),
                    np.mean(np.concatenate(statistics["hybrid_error_em"])),
                )
            )

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
        Callback function to visualize discrete results in hybrid prediction.

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
        if ProtobufPredictionDataset.DATASET_KEY_MANEUVERS in batch_itm:
            # Visualize discrete mode as text.
            batch_maneuver_tensor = batch_itm[ProtobufPredictionDataset.DATASET_KEY_MANEUVERS].float()
            mode_img = visualize_mode_sequences(
                batch_maneuver_tensor, predicted["maneuvers"], num_past_timepoints, stats, batch_idx
            )
            mode_img = mode_img.transpose(2, 0, 1)
            writer.add_image("vis_" + str(visualization_idx) + "/mode", mode_img, global_step=iter)

    def update_solution(self, solution: dict, predicted: dict, batch_idx: int):
        """
        Callback function to add predicted discrete states to solution.

        Parameters
        ----------
        solution : dict
            Dictionary of solution.
        predicted : dict
            Dictionary of prediction.
        batch_idx : int
            Batch index.
        """
        solution["predicted_maneuvers"] = predicted["maneuvers"][batch_idx]

    def update_decoding(self, data: dict, stats_list: list):
        """
        Callback function to add decoding to decoder inputs.

        Parameters
        ----------
        data : dict
            Dictionary of data.
        stats_list : list
            List of stats dictionaries.
        """
        mode_predictions = [stats["discrete_samples"] for stats in stats_list]
        # Shape [batch_size, num_agent, num_future_step, discrete_size, sample_size]
        mode_predictions = torch.stack(mode_predictions, -1)
        data["maneuvers"] = mode_predictions

    def update_additional_inputs(self, additional_inputs: dict, batch_itm: dict, num_past_timepoints: int):
        """
        Callback function to add maneuvers as additional inputs.

        Parameters
        ----------
        additional_inputs : dict
            Dictionary of additional inputs.
        batch_itm : dict
            Dictionary of batch item.
        num_past_timepoints : int
            Number of past time steps.
        """
        discrete_names = [
            (
                ProtobufPredictionDataset.DATASET_KEY_MANEUVERS,
                ProtobufPredictionDataset.DATASET_KEY_MANEUVERS_PAST,
                ProtobufPredictionDataset.DATASET_KEY_MANEUVERS_FUTURE,
            ),
        ]
        for (full, past, future) in discrete_names:
            data = batch_itm[full].float()
            additional_inputs[full] = data.clone().detach()
            data_past = data[:, :, :num_past_timepoints].clone().detach()
            additional_inputs[past] = data_past
            data_future = data[:, :, num_past_timepoints:].clone().detach()
            additional_inputs[future] = data_future

    def update_expected_results(self, expected: dict, batch_itm: dict, num_future_timepoints: int):
        """
        Callback function to add maneuvers to expected states.

        Parameters
        ----------
        expected : dict
            Dictionary of expected values.
        batch_itm : dict
            Dictionary of batch item.
        num_future_timepoints : int
            Number of future time steps.
        """
        batch_maneuver_tensor = batch_itm[ProtobufPredictionDataset.DATASET_KEY_MANEUVERS].float()
        expected_maneuvers = batch_maneuver_tensor[:, -num_future_timepoints:]
        # Cast to desired shape.
        expected_maneuvers = expected_maneuvers[:, None, :, None, None]
        expected["maneuvers"] = expected_maneuvers

    def update_save_validation_criterion_key(self, trainer_param: dict) -> list:
        """
        Callback function to overwrite save_validation_criterion_key in prediction_trainer.

        Parameters
        ----------
        trainer_param : dict
            Dictionary of trainer parameters.

        Returns
        -------
        save_validation_criterion_key : str
            Criterion key for saving in validation.
        folder_postfix : str
            Postfix of folder to save.
        """
        if trainer_param["discrete_supervised"] and trainer_param["MoN_number_samples"] == 1:
            save_validation_criterion_key = "hybrid_error_supervised_continuous"
            folder_postfix = "_best_hybrid_cont"
            return [save_validation_criterion_key, folder_postfix]
        else:
            return []

    def update_visualization_text(
        self, cost_str: str, solution: dict, batch: dict, batch_idx: int, num_past_points: int
    ) -> str:
        """
        Update visualization text to include predicted mode for the first agent.

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
        if "predicted_maneuvers" in solution:
            text += "\nmode seqs ="

            predicted_modes = solution["predicted_maneuvers"].argmax(-2)
            num_mode_samples = solution["predicted_maneuvers"].shape[-1]

            if self.params["compute_maneuvers_online"]:
                mode_dict = MODE_DICT_IDX_TO_MANEUVER_ONLINE
            else:
                mode_dict = MODE_DICT_IDX_TO_MANEUVER

            # Print ground truth label.
            gt_mode = batch[ProtobufPredictionDataset.DATASET_KEY_MANEUVERS][batch_idx, 0, num_past_points:]
            gt_mode = list(gt_mode.detach().cpu().numpy())
            gt_mode_str = [mode_dict[mode] for mode in gt_mode]
            text += "\n GT: " + str(gt_mode_str)
            for s in range(num_mode_samples):
                predicted_mode_s = list(predicted_modes[0, :, s].detach().cpu().numpy())
                predicted_mode_s_str = [mode_dict[mode] for mode in predicted_mode_s]
                text += "\n" + str(predicted_mode_s_str)
        return text

    def update_visualization_agent_color(
        self, agent_color_predicted: MatplotlibColor, param: dict, sample_i: int, solution: dict
    ) -> MatplotlibColor:
        """
        Update visualization text.

        Parameters
        ----------
        agent_color_predicted : str
            Color for predicted agent trajectory.
        param : dict
            Parameters.
        sample_i : int
            Sample index.
        solution : dict
            Solution dictionary.
        """
        if param["discrete_supervised"] and (sample_i == solution["predicted_trajectories"].shape[3] - 1):
            agent_color_predicted = "magenta"
        return agent_color_predicted

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
    ) -> None:
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
        ax: matplotlib.axes._axes.Axes
            Plot ax.
        predicted_x : torch.Tensor
            Predicted x coordinates.
        predicted_y : torch.Tensor
            Predicted y coordinates.
        agent_color_predicted : str or tuple
            Color for predicted agent trajectory.
        """
        predicted_mode = (
            solution["predicted_maneuvers"][agent_id, is_future_valid[agent_id, :].bool(), :, sample_i]
            .argmax(-1)
            .cpu()
            .numpy()
        )
        change_indices = predicted_mode[1:] != predicted_mode[:-1]
        ax.plot(
            predicted_x[1:][change_indices],
            predicted_y[1:][change_indices],
            "o",
            color=agent_color_predicted,
            alpha=1.0,
            ms=5,
            zorder=15,
        )
        # Add sample index to the end of position.
        ax.text(predicted_x[-1], predicted_y[-1], str(sample_i), color="black", size=6)
