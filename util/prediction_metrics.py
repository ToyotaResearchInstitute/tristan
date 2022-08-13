""" Implements different metrics to evaluate trajectory predictions. """
from enum import Enum
from typing import Dict, List, Optional, cast

import shapely.geometry as geom
import torch

from loaders.ado_key_names import AGENT_TYPE_NAME_MAP
from radutils.torch.torch_utils import apply_2d_coordinate_rotation_transform

TRAVERSAL_RECTANGLE_BASE_X_LENGTH = 20.0
TRAVERSAL_RECTANGLE_BASE_Y_LENGTH = 1.9


def get_horizon_masks(
    timestamps: torch.Tensor,
    prediction_timestamp: torch.Tensor,
    horizons: List[float],
    agent_visibility: torch.Tensor,
    tmp_res_eps: float = 1e-5,
) -> Dict[float, Dict[str, torch.Tensor]]:
    result: Dict[float, Dict[str, torch.Tensor]] = {}
    for horizon in horizons:
        relative_timestamps = timestamps - prediction_timestamp.unsqueeze(-1) - horizon
        before_time_point = relative_timestamps <= tmp_res_eps
        after_time_point = relative_timestamps >= -tmp_res_eps
        incomplete_horizons = after_time_point.any(dim=-1).logical_not()

        final_index = (timestamps * before_time_point).argmax(axis=-1)
        final_time_point = torch.zeros_like(before_time_point).scatter_(-1, final_index[:, None], 1).bool()
        final_time_point *= incomplete_horizons.logical_not()[:, None]

        # Masks of all time points visible before horizon and the last time point visible before horizon
        # (shape batch_size x num_agents x num_time_points)
        visible_in_horizon = before_time_point.unsqueeze(-2).expand(agent_visibility.shape) * (agent_visibility > 0)

        last_visible_index = ((timestamps * before_time_point).unsqueeze(1) * visible_in_horizon).argmax(axis=-1)
        last_visible_in_horizon = (
            torch.zeros_like(agent_visibility).scatter_(-1, last_visible_index[..., None], 1).bool()
        )
        last_visible_in_horizon &= agent_visibility > 0

        result[horizon] = {
            "final": final_time_point,
            "visible_in_horizon": visible_in_horizon,
            "last_visible_in_horizon": last_visible_in_horizon,
            "incomplete_horizons": incomplete_horizons,
        }

    return result


class Discard(Enum):
    """Discarding policies for handling invalid data during aggregation.

    General idea for the behavior is as follows:
        * TRAJECTORY indicates that only the invalid trajectory should be discarded in data aggregation.
        * DATAPOINT indicates discaring of entire scene if one of the trajectories therein is invalid.
        * BATCH indicates discarding entire batch if one of the trajectories therein is invalid.
    """

    TRAJECTORY = 1
    DATAPOINT = 2
    BATCH = 3


# This method is used for work not currently committed to this repository
def gaussian_kl_divergence(
    mean_p: torch.Tensor, mean_q: torch.Tensor, cov_p: torch.Tensor, cov_q: torch.Tensor
) -> torch.Tensor:
    """Computes the Kullback-Leibler Divergence between multivariate Gaussians

    This computes the KLD
        KLD(p|q) = \\int p(x) Log(p(x) / q(x)) dx
    assuming that both, distributions are gausssian given by their means and covariances.
    Parameters
    ----------
    mean_p, mean_q : torch.Tensor
        The means of the gaussians of shape (num_dim).

    cov_p, cov_q : torch.Tensor
        The covariance matrices of the gaussians of same shape (num_dim x num_dim).

    Returns
    -------
    torch.Tensor
        The resulting KLD between both distributions.
    """

    det_cov_p = torch.linalg.det(cov_p)
    det_cov_q = torch.linalg.det(cov_q)

    inv_cov_q = torch.linalg.inv(cov_q)

    mean_diff = mean_q - mean_p

    # Using https://stats.stackexchange.com/a/60699/18530
    kld = (
        torch.log(det_cov_q)
        - torch.log(det_cov_p)
        - mean_p.shape[0]
        + torch.trace(torch.matmul(inv_cov_q, cov_p))
        + torch.dot(torch.matmul(mean_diff, inv_cov_q), mean_diff)
    )

    return kld


# This method is used for work not currently committed to this repository
def gaussian_kl_divergence_isotropic(
    mean_p: torch.Tensor, mean_q: torch.Tensor, cov_p: torch.Tensor, cov_q: torch.Tensor
) -> torch.Tensor:
    """Computes the Kullback-Leibler Divergence between isotropic multivariate Gaussians

    This computes the KLD
        KLD(p|q) = \\int p(x) Log(p(x) / q(x)) dx
    assuming that both, distributions are gausssian given by their means and covariances.

    Parameters
    ----------
    mean_p, mean_q : torch.Tensor
        The means of the gaussians of shape (bath_size, num_dim).

    cov_p, cov_q : torch.Tensor
        Diagonal entries of the covariance matrices of the gaussians of shape (batch_size, num_dim).

    Returns
    -------
    torch.Tensor
        The resulting KLD between both distributions.
    """
    mean_diff = mean_q - mean_p
    # Using https://stats.stackexchange.com/a/60699/18530
    kld = (
        torch.sum(torch.log(cov_q), axis=1)  # log(det(cov_q))
        - torch.sum(torch.log(cov_p), axis=1)  # log(det(cov_p))
        - mean_p.shape[1]
        + torch.sum(cov_p / cov_q, axis=1)  # tr(cov_q^-1 * cov_p)
        + torch.einsum("bi,bi->b", mean_diff / cov_q, mean_diff)  # (mu_p - mu_q)^T * cov_q^-1 * (mu_p - mu_q)
    )
    return kld


def mean_traj_stats(
    stats: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    strictness: Discard = Discard.TRAJECTORY,
) -> torch.Tensor:
    """Computes the (weighted) mean over a given trajectory statistic (which itself can be ADE, FDE, ...).

    Parameters
    ----------
    stats : torch.Tensor
        The statistic that has to be aggregated. Invalid values are assumed to be nan (shape batch_size x num_agents).
    weights : torch.Tensor, optional
        An optional weight matrix of same shape as stats. This is assumed to be non-nan and non-inf for every entry corresponding to a non-nan entry in stats.
    strictness : Discard, optional
        Controls aggregation behavior. By default, a trajectory not containing a statistic is simply discarded from the
        computation. Alternative behaviors are discarding the entire datapoint/scene that contains a trajectory without
        valid statistic (DATAPOINT). Or even discarding the entire batch and returning a nan (BATCH).

    Returns
    -------
    torch.Tensor
        The aggregared (weighted) mean. Depending on the strictness parameter, this can be nan in the following cases:
        Discard.TRAJECTORY: the result is only nan if all entries in stats are nans.
        Discard.DATAPOINT: the result is nan if all scenes/datapoints contain a nan trajectory.
        Discard.BATCH: the result is nan if stats contains at least one nan.
    """
    num_agents = stats.shape[1]
    validity = torch.isnan(stats).logical_not()

    if not validity.any() or (strictness == Discard.BATCH and not validity.all()):
        return stats.new_tensor(float("nan"))

    traj_mask = validity
    if strictness == Discard.DATAPOINT:
        valid_datapoint_mask = (validity.sum(1) == num_agents).unsqueeze(1).expand(-1, num_agents)
        traj_mask = valid_datapoint_mask * traj_mask

    if weights is not None:
        # Normalize weights
        normalized_weights = weights * traj_mask
        normalized_weights /= normalized_weights.sum()
        result = (stats[traj_mask] * normalized_weights[traj_mask]).sum()
    else:
        result = stats[traj_mask].sum() / traj_mask.sum()

    return result


def displacement_errors(
    predicted_trajectory_scene: torch.Tensor,
    observed_trajectory_scene: torch.Tensor,
    timestamps: torch.Tensor,
    prediction_timestamp: torch.Tensor,
    relevant_agents: torch.Tensor,
    visibility: torch.Tensor,
    horizons: list,
    miss_thresholds: torch.Tensor,
    param: dict,
    tmp_res_eps: float = 1e-5,
    relevant_agent_types: Optional[list] = None,
    agent_types: Optional[torch.Tensor] = None,
    miss_thresholds_x: Optional[torch.Tensor] = None,
    miss_thresholds_y: Optional[torch.Tensor] = None,
    rotations_local_scene: Optional[torch.Tensor] = None,
) -> dict:
    """This method implements the average & final displacement error computation.

    The displacement errors can be computed at different time horizons, i.e. only considering a predefined number of future timestemps assuming that predicted

    :param predicted_trajectory_scene: Predicted trajectories of shape (batch_size x num_agents x num_timesteps x 2).
    :param observed_trajectory_scene: Observed true trajectories of same shape as pred
    :param timestamps: Timestamps of the trajectory data of shape (batch_size x num_timesteps)
    :param prediction_timestamp: All timestamps larger than the timestamps given here are considered for ADE/FDE
        computation. Shape (batch_size,). This is
        necessary because the first entry of timestamps might not correspond to the timestamp of the first entry in predicted / observed.
    :param relevant_agents: Binary integer tensor of shape (batch_size x num_agents) indicating which agent is
        relevant for error computation.
    :param visibility: Boolean tensor of shape (batch_size x num_agents x num_timesteps) indicating in which
        timesteps an agent is visible. It is assumed that all agents are always visible if this parameter is not set.
    :param horizons: A list of timepoints containing the ADE/FDE computation lookahead horizon.
    :param miss_thresholds: A list of distance thresholds for a prediction to be considered a miss.
    :param param: A dictionary of configuration parameters.
    :param relevant_agent_types: An optional list of relevant agent types.
    :param agent_types: One-hot agent type information (also optional), (batch_size x num_agents x num_types).
    :param tmp_res_eps: Epsilon value to account for temporal resolution. Has to be small enough such that two
        timesteps are never in a 2*tmp_res_eps interval and big enough to account for numerical errors (default 1e-5).
    :param rotations_local_scene: The 2D rotation matrix of agents to transform displacements into a specified frame
    :returns: A dictionary containing the following entries:
        * agent_ades_partial, agent_fdes_partial: Dict of Average Displacement Error (ADE) and Final Displacement Error
          (FDE) for different horizons indexed by horizon each containing per agent data (shape batch_size x num_agents)
          which is nan in case of invalidity (e.g. because agent irrelevant). If an agent is predicted beyond the
          horizon for which we have ground truth, the agent is treated as being predicted at the horizon limit for which
          we have ground truth. Data for earlier horizons are also included to show prediction performance for the given
          trajectory at earlier horizons.
        * agent_ades, agent_fdes: Same as the above except that here all agents are considered invalid that
          have not been observed over the full horizon. Error values are only provided for the final horizon for the
          given trajectory. Data for earlier horizons are removed from these dicts in order to avoid mistaken
          comparisons between different horizons and calculations using values from different horizons.
        * misses_partial, misses: Dict indexed by horizons indicating where a fde prediction has been a miss with
            regard to a given miss threshold (shape num_batches x num_agents x num_miss_thresholds). misses_partial
            accepts partial horizon (i.e too short) trajectories, whereas miss only takes full-horizon ones into
            account. Like above, partial misses will contain data for all horizons up to the final horizon for the given
            trajectory and misses will only contain data corresponding to the final horizon in the provided trajectory.
        * ades_partial, ades, fdes_partial, fdes: Dicts indexed by horizons with the statistics above aggregated over
            all valid trajectories.
        * agent_ade, agent_fde: Tensor of ADEs / FDEs for each valid & visible agent individually with each invalid
            entry being set to nan. In contrast to the entries in "agent_ades_partial", "agent_fdes_partial", this
            tensor is independent of the horizon (shape batch_size x num_agents).
        * ade, fde: mean over all non-nan values of agent_ade / agent_fde
        * square_error: Mean square error weighted with the number of visible locations per agent.
        All other dict entries remain for compatibility and may potentially be updated / discarded in the future.
    """
    # TODO expand result docstring
    num_agents = predicted_trajectory_scene.shape[1]

    results = {}

    # Populate per-location validity mask (shape  batch_size x num_agents x num_timesteps)
    agent_visibility = (visibility > 0).float()

    # Ignore loss and metrics for the first agent if conditional prediction is used.
    # This assume the first agent is the ego agent.
    if "conditional_prediction" in param and param["conditional_prediction"]:
        relevant_agents[:, 0] = relevant_agents[:, 0] * 0

    # Treat irrelevant agents as invisible
    relevant_agents_mask = relevant_agents[..., None]
    agent_visibility = (agent_visibility * relevant_agents_mask).float()

    # Treat agents of wrong type as invisible.
    if relevant_agent_types is not None:
        assert agent_types is not None, "agent_types cannot be None if relevant_agent_types is set"
        assert "agent_types" in param, "full agent types need to be specified if relevant_agent_types is set"

        # Mask containing all agents of relevant type (shape batch_size x num_agents)
        rel_types_mask = agent_types.new_zeros(agent_types.shape[0:-1], dtype=float)
        for cur_agent_type_id in relevant_agent_types:
            cur_agent_type_id_index = param["agent_types"].index(cur_agent_type_id)
            rel_types_mask += agent_types[..., cur_agent_type_id_index]

        rel_types_mask = rel_types_mask > 0
        agent_visibility *= rel_types_mask.unsqueeze(-1).expand_as(agent_visibility)

    # Squared position error. Rescaling necessary to undo previous downscaling for training.
    # (shape batch_size x num_agents x num_timesteps)
    if rotations_local_scene is not None:
        predicted_trajectory_local = apply_2d_coordinate_rotation_transform(
            rotations_local_scene,
            predicted_trajectory_scene,
            result_einsum_prefix="bat",
            rotation_einsum_prefix="b",
        )
        observed_trajectory_local = apply_2d_coordinate_rotation_transform(
            rotations_local_scene,
            observed_trajectory_scene,
            result_einsum_prefix="bat",
            rotation_einsum_prefix="b",
        )
    else:
        predicted_trajectory_local = predicted_trajectory_scene
        observed_trajectory_local = observed_trajectory_scene

    sqr_err = ((predicted_trajectory_scene - observed_trajectory_scene) ** 2).sum(axis=3).float() / (
        param["predictor_normalization_scale"] ** 2
    )
    sqr_err *= agent_visibility
    sqr_err_x = ((predicted_trajectory_local - observed_trajectory_local) ** 2)[..., 0] / (
        param["predictor_normalization_scale"] ** 2
    )
    sqr_err_y = ((predicted_trajectory_local - observed_trajectory_local) ** 2)[..., 1] / (
        param["predictor_normalization_scale"] ** 2
    )
    sqr_err_x *= agent_visibility
    sqr_err_y *= agent_visibility

    truncation_scale = param["predictor_truncation_scale"]
    if agent_types is not None and param["tailored_robust_function"]:
        # TODO(guy.rosman): Feed in mapping from agent type to scale, to generalize pedestrians.
        pedestrian_noise_scale = param["pedestrian_noise_scale"]
        pedestrian_type_id = param["pedestrian_type_id"]
        per_agent_scale = torch.ones_like(agent_types[0:1, 0:1, :])
        per_agent_scale[0, 0, pedestrian_type_id] = pedestrian_noise_scale
        agent_specific_scale = (agent_types * per_agent_scale).sum(2) / agent_types.sum(2)
    else:
        agent_specific_scale = torch.ones_like(sqr_err)

    time_specific = param["tailored_robust_function"]
    if time_specific:
        temporal_slope = param["robust_function_temporal_slope"]
        temporal_intercept = param["robust_function_temporal_intercept"]
        robust_error_scale = agent_specific_scale.unsqueeze(2) * (
            timestamps.unsqueeze(1) * temporal_slope + temporal_intercept
        )
    else:
        robust_error_scale = agent_specific_scale

    if torch.isnan(robust_error_scale).any() or torch.isinf(robust_error_scale).any():
        robust_error_scale[torch.isnan(robust_error_scale)] = robust_error_scale[~torch.isnan(robust_error_scale)].max()
    sqr_err_ = sqr_err / robust_error_scale**2
    robust_individual_err = sqr_err_.clamp(0, truncation_scale) + (sqr_err_ - truncation_scale).clamp(min=1e-5).sqrt()
    robust_individual_err *= agent_visibility
    results["robust_error"] = (
        robust_individual_err.sum(dim=2).sum(dim=1) / agent_visibility.sum(dim=2).sum(dim=1).float()
    )

    results["square_error"] = sqr_err.sum(dim=2).sum(dim=1) / agent_visibility.sum(dim=2).sum(dim=1).float()

    results.update(
        {
            "fdes_partial": {},
            "fdes": {},
            "ades_partial": {},
            "ades": {},
            "agent_ades_partial": {},
            "agent_ades": {},
            "agent_fdes_partial": {},
            "agent_fdes_x_partial": {},
            "agent_fdes_y_partial": {},
            "agent_fdes_x": {},
            "agent_fdes_y": {},
            "agent_fdes": {},
            "misses_partial": {},
            "misses_partial_x": {},
            "misses_partial_y": {},
            "misses": {},
            "misses_x": {},
            "misses_y": {},
        }
    )
    report_agent_type_metrics = param["report_agent_type_metrics"]
    if report_agent_type_metrics:
        assert "agent_types" in param, "Agent types need to be provided if report_agent_type_metrics is set."
        for agent_type_id in param["agent_types"]:
            agent_type = AGENT_TYPE_NAME_MAP[agent_type_id]
            results.update({"agent_fdes_partial_{}".format(agent_type): {}})
            results.update({"agent_fdes_{}".format(agent_type): {}})
            results.update({"agent_ades_partial_{}".format(agent_type): {}})
            results.update({"agent_ades_{}".format(agent_type): {}})

    miss_thresholds_tensor = predicted_trajectory_scene.new_tensor(miss_thresholds)
    # TODO(guy.rosman): Populate miss_thresholds_x_tensor from miss_thresholds_x, same for y.
    if not miss_thresholds_x:
        miss_thresholds_x = miss_thresholds
    if not miss_thresholds_y:
        miss_thresholds_y = miss_thresholds
    miss_thresholds_x_tensor = predicted_trajectory_scene.new_tensor(miss_thresholds_x)
    miss_thresholds_y_tensor = predicted_trajectory_scene.new_tensor(miss_thresholds_y)

    pos_norm_err = torch.sqrt(sqr_err)
    pos_norm_err_x = torch.sqrt(sqr_err_x)
    pos_norm_err_y = torch.sqrt(sqr_err_y)
    horizon_masks = get_horizon_masks(timestamps, prediction_timestamp, horizons, agent_visibility, tmp_res_eps)
    for cur_horizon in horizons:
        # Masks for all timepoints at or before/after desired horizon (shape batch_size x num_timesteps)
        # final_timepoint should have a value as long as the final trajectory timestamp is within one dt
        # of prediction_timestamp.
        final_timepoint = horizon_masks[cur_horizon]["final"]
        visible_in_horizon = horizon_masks[cur_horizon]["visible_in_horizon"]
        last_visible_in_horizon = horizon_masks[cur_horizon]["last_visible_in_horizon"]
        assert (final_timepoint.sum(1) <= 1.0).all(), "Not more than one time point can be used for full FDE"

        # Mask of timepoints that correspond to desired horizon inflated to number of agents
        # (shape batch_size x num_agents x num_timepoints)
        final_timepoint_per_agent = final_timepoint.unsqueeze(-2).expand(-1, num_agents, -1) * agent_visibility

        # Agents for which we don't have (full) trajectories observed (shape batch_size x num_agents)
        invalid_agents = agent_visibility.sum(2) == 0
        invalid_agents_full = (final_timepoint_per_agent.sum(2) == 0).bool()

        # Resulting ADEs (shape batch_size x num_agents)
        results["agent_ades_partial"][cur_horizon] = (visible_in_horizon * pos_norm_err).sum(
            2
        ) / visible_in_horizon.sum(2)
        results["agent_ades_partial"][cur_horizon][invalid_agents] = pos_norm_err.new_tensor(float("nan"))
        results["agent_ades"][cur_horizon] = results["agent_ades_partial"][cur_horizon].clone()
        results["agent_ades"][cur_horizon][invalid_agents_full] = pos_norm_err.new_tensor(float("nan"))

        # Resulting FDEs (shape batch_size x num_agents)
        results["agent_fdes_partial"][cur_horizon] = (last_visible_in_horizon * pos_norm_err).sum(2)
        results["agent_fdes_partial"][cur_horizon][invalid_agents] = pos_norm_err.new_tensor(float("nan"))
        results["agent_fdes"][cur_horizon] = results["agent_fdes_partial"][cur_horizon].clone()
        results["agent_fdes"][cur_horizon][invalid_agents_full] = pos_norm_err.new_tensor(float("nan"))

        results["agent_fdes_x_partial"][cur_horizon] = (last_visible_in_horizon * pos_norm_err_x).sum(2)
        results["agent_fdes_y_partial"][cur_horizon] = (last_visible_in_horizon * pos_norm_err_y).sum(2)
        results["agent_fdes_x"][cur_horizon] = results["agent_fdes_x_partial"][cur_horizon].clone()
        results["agent_fdes_x"][cur_horizon][invalid_agents_full] = pos_norm_err_x.new_tensor(float("nan"))
        results["agent_fdes_y"][cur_horizon] = results["agent_fdes_y_partial"][cur_horizon].clone()
        results["agent_fdes_y"][cur_horizon][invalid_agents_full] = pos_norm_err_y.new_tensor(float("nan"))

        # TODO(guy.rosman): change to match threshold to cur_horizon.
        results["misses_partial_x"][cur_horizon] = miss_mask(
            results["agent_fdes_x_partial"][cur_horizon], miss_thresholds_x_tensor
        )
        results["misses_partial_y"][cur_horizon] = miss_mask(
            results["agent_fdes_y_partial"][cur_horizon], miss_thresholds_y_tensor
        )

        bool_y_misses_partial = results["misses_partial_y"][cur_horizon].bool()
        bool_x_misses_partial = results["misses_partial_x"][cur_horizon].bool()
        bool_y_misses_partial[results["misses_partial_y"][cur_horizon].isnan()] = False
        bool_x_misses_partial[results["misses_partial_x"][cur_horizon].isnan()] = False
        partial_nan_mask = torch.bitwise_or(
            results["misses_partial_y"][cur_horizon].isnan(), results["misses_partial_x"][cur_horizon].isnan()
        )
        # results["misses_partial"][cur_horizon] = miss_mask(
        #     results["agent_fdes_partial"][cur_horizon], miss_thresholds_tensor
        # )
        results["misses_partial"][cur_horizon] = torch.bitwise_or(bool_x_misses_partial, bool_y_misses_partial).float()
        results["misses_partial"][cur_horizon][partial_nan_mask] = float("nan")

        # results["misses"][cur_horizon] = miss_mask(results["agent_fdes"][cur_horizon], miss_thresholds_tensor)
        results["misses_x"][cur_horizon] = miss_mask(results["agent_fdes_x"][cur_horizon], miss_thresholds_x_tensor)
        results["misses_y"][cur_horizon] = miss_mask(results["agent_fdes_y"][cur_horizon], miss_thresholds_y_tensor)

        bool_y_misses = results["misses_y"][cur_horizon].bool()
        bool_x_misses = results["misses_x"][cur_horizon].bool()
        bool_y_misses[results["misses_y"][cur_horizon].isnan()] = False
        bool_x_misses[results["misses_x"][cur_horizon].isnan()] = False
        nan_mask = torch.bitwise_or(results["misses_y"][cur_horizon].isnan(), results["misses_x"][cur_horizon].isnan())

        results["misses"][cur_horizon] = torch.bitwise_or(bool_x_misses, bool_y_misses).float()
        results["misses"][cur_horizon][nan_mask] = float("nan")

        results["ades_partial"][cur_horizon] = mean_traj_stats(results["agent_ades_partial"][cur_horizon])
        results["ades"][cur_horizon] = mean_traj_stats(results["agent_ades"][cur_horizon])
        results["fdes_partial"][cur_horizon] = mean_traj_stats(results["agent_fdes_partial"][cur_horizon])
        results["fdes"][cur_horizon] = mean_traj_stats(results["agent_fdes"][cur_horizon])

        # Add metrics for each agent type.
        if report_agent_type_metrics and agent_types is not None:
            for agent_type_id in param["agent_types"]:
                agent_type = AGENT_TYPE_NAME_MAP[agent_type_id]
                agent_type_index = param["agent_types"].index(agent_type_id)
                invalid_agent_type_mask = agent_types.argmax(-1) != agent_type_index
                invalid_agents_type = torch.logical_or(invalid_agents, invalid_agent_type_mask)
                invalid_agents_full_type = torch.logical_or(invalid_agents_full, invalid_agent_type_mask)

                # Resulting ADEs (shape batch_size x num_agents)
                results["agent_ades_partial_{}".format(agent_type)][cur_horizon] = (
                    visible_in_horizon * pos_norm_err
                ).sum(2) / visible_in_horizon.sum(2)
                results["agent_ades_partial_{}".format(agent_type)][cur_horizon][
                    invalid_agents_type
                ] = pos_norm_err.new_tensor(float("nan"))
                results["agent_ades_{}".format(agent_type)][cur_horizon] = results["agent_ades_partial"][
                    cur_horizon
                ].clone()
                results["agent_ades_{}".format(agent_type)][cur_horizon][
                    invalid_agents_full_type
                ] = pos_norm_err.new_tensor(float("nan"))

                # Resulting FDEs (shape batch_size x num_agents)
                results["agent_fdes_partial_{}".format(agent_type)][cur_horizon] = (
                    last_visible_in_horizon * pos_norm_err
                ).sum(2)
                results["agent_fdes_partial_{}".format(agent_type)][cur_horizon][
                    invalid_agents_type
                ] = pos_norm_err.new_tensor(float("nan"))
                results["agent_fdes_{}".format(agent_type)][cur_horizon] = results["agent_fdes_partial"][
                    cur_horizon
                ].clone()
                results["agent_fdes_{}".format(agent_type)][cur_horizon][
                    invalid_agents_full_type
                ] = pos_norm_err.new_tensor(float("nan"))

    # Per agent ADE for all available timesteps.
    results["agent_ade"] = pos_norm_err.sum(2) / agent_visibility.sum(2)
    results["agent_ade"][agent_visibility.sum(2) == 0] = pos_norm_err.new_tensor(float("nan"))

    # Mask of the last visible timestep in valid trajectories (shape batch_size x num_agents x num_timesteps)
    last_visible_timestep = (
        agent_visibility.cumsum(2) == agent_visibility.cumsum(2).max(2, keepdim=True)[0]
    ) * agent_visibility

    results["agent_fde"] = (pos_norm_err * last_visible_timestep).sum(2)
    results["agent_fde"][agent_visibility.sum(2) == 0] = pos_norm_err.new_tensor(float("nan"))

    results["agent_square_error"] = sqr_err.sum(2) / agent_visibility.sum(2)
    results["agent_square_error"][agent_visibility.sum(2) == 0] = pos_norm_err.new_tensor(float("nan"))

    # Save raw square error without any operations (i.e. averaging, summing).
    results["agent_square_error_raw"] = sqr_err
    results["agent_visibility"] = agent_visibility

    results["ade"] = results["agent_ade"][~results["agent_ade"].isnan()].mean()
    results["fde"] = results["agent_fde"][~results["agent_fde"].isnan()].mean()

    # Compute square_error weighting each agent with the number of its observations.
    # TODO(igor.gilitschenski): Discuss with Guy if we should average this in the same way we average other stats.
    # results["square_error"] = mean_traj_stats(results["agent_square_error"], agent_visibility.sum(2))

    # Take only timepoints where all agent positions are valid.
    weights_ade = agent_visibility.sum(1) == relevant_agents.sum(1, keepdims=True)
    # Get a rising number for nonzero entries
    weights = weights_ade.cumsum(1)
    w1, _ = weights.max(1, keepdim=True)
    # Get a mask of last nonzero + following zero entries on rows.
    weights_fde = (weights == w1) * weights_ade
    # Weight the last nonzero entry (the rest should be zero).
    # TODO(cyrushx): Change to a more explicit name?
    results["individual_fde_err"] = (
        weights_fde * (pos_norm_err.sum(1) / (relevant_agents.sum(1, keepdims=True) + 1e-20))
    ).sum(1)

    results["individual_ade_err"] = (
        weights_ade * (pos_norm_err.sum(1) / (relevant_agents.sum(1, keepdims=True) + 1e-20))
    ).sum(1) / weights_ade.sum(1)

    if param["raw_l2_for_mon"]:
        results["mon_individual_error"] = sqr_err.sum(dim=2).sum(dim=1) / agent_visibility.sum(dim=2).sum(dim=1).float()
    else:
        results["mon_individual_error"] = (
            robust_individual_err.sum(dim=2).sum(dim=1) / agent_visibility.sum(dim=2).sum(dim=1).float()
        )

    return results


def miss_mask(fde: torch.Tensor, miss_thresholds: torch.Tensor):
    """Computes a binary mask of misses based on different thresholds.

    :param fde: The fde points at which the misses will be computed. Invalid points are assumed to be given as nan
        (shape can be arbitrary).
    :param miss_thresholds: The thresholds used for computing the misses (shape can be arbitrary).
    :return Resulting miss mask (shape FDE_SHAPE x MISS_THRESHOLDS_SHAPE)
    """
    # Inflate fde and miss_thresholds to output dimension
    fde_inflated = fde.view(*fde.shape, *(1,) * len(miss_thresholds.shape)).expand(*fde.shape, *miss_thresholds.shape)
    miss_thresholds_inflated = miss_thresholds.view(*(1,) * len(fde.shape), *miss_thresholds.shape).expand(
        *fde.shape, *miss_thresholds.shape
    )

    result = (fde_inflated >= miss_thresholds_inflated).float()
    result[fde_inflated.isnan()] = fde.new_tensor(float("nan"))
    return result


def check_trajectory_traversal(
    trajectory_scene: torch.Tensor, crossing_rectangles_scene: list, visibility: torch.Tensor
):
    """Check if a provided trajectory enters a provided polygon.

    :param trajectory_scene: Shape (batch_size x num_agents x num_time_steps x 2) Tensor corresponding to trajectories
        for all agents
    :param crossing_rectangles_scene: list of Shapely polygons of size [batch_size] defining zones to check for trajectory traversal
    :param visibility: Shape (batch_size x num_agents x num_timesteps) Boolean tensor indicating in which timesteps an
        agent is visible.
    :return Shape (batch_size x num_agents) Boolean tensor whose value corresponds to whether the trajectory enters
        the zone in question.
    """
    traversal_distances = torch.zeros(trajectory_scene.shape[0:2], dtype=torch.float, device=trajectory_scene.device)
    for i in range(trajectory_scene.shape[0]):
        for j in range(trajectory_scene.shape[1]):
            trajectory_points_scene = trajectory_scene[i, j, :, :][visibility[i, j, :].bool()]

            if len(trajectory_points_scene) > 1:
                trajectory_geom_scene = geom.LineString(trajectory_points_scene.tolist())
                distance = trajectory_geom_scene.distance(crossing_rectangles_scene[i])
                if distance:
                    # Trajectory does not intersect
                    traversal_distances[i, j] = distance
                else:
                    # Trajectory intersects, return the negative of the distance between the deepest point of the
                    # trajectory and the crossing boundary
                    # TODO profile use of shapely
                    internal_point_list = [
                        p
                        for p in map(geom.Point, trajectory_geom_scene.coords)
                        if p.within(crossing_rectangles_scene[i])
                    ]
                    if len(internal_point_list) > 0:
                        traversal_distances[i, j] = -max(
                            p.distance(crossing_rectangles_scene[i].exterior) for p in internal_point_list
                        )
                    else:
                        # Trajectory passes through crossing boundary but no individual point is within the crossing
                        # zone.  Set distance to zero
                        traversal_distances[i, j] = 0
            else:
                # If we have less than 2 points, we don't have a trajectory so it doesn't intersect.
                traversal_distances[i, j] = float("nan")

    return traversal_distances


def create_expected_traversal_rectangles(
    rotations_scene_ego: torch.Tensor, ego_poses: torch.Tensor, scale: float
) -> List[geom.Polygon]:
    """Create a rectangle in front of each ego position in order to check if a pedestrian crosses in front of the vehicle.

    :param rotations_scene_ego: A tensor [batch, 2, 2] of rotation matrices specifying ego orientations
    :param ego_poses: A tensor [batch, 2] of [x,y] points specifying ego poses
    :param scale: An overall scene scale
    :return list: A list of shapely Polygons
    """
    traversal_rectangle_ego = torch.tensor(
        list(
            geom.box(
                0,
                -TRAVERSAL_RECTANGLE_BASE_Y_LENGTH / 2,
                TRAVERSAL_RECTANGLE_BASE_X_LENGTH,
                TRAVERSAL_RECTANGLE_BASE_Y_LENGTH / 2,
            ).exterior.coords
        ),
        device=rotations_scene_ego.device,
    )

    # Apply scale
    traversal_rectangle_ego = traversal_rectangle_ego * scale
    # Apply rotation using einsum, preserving batch index and polygon grouping
    traversal_rectangles_scene = apply_2d_coordinate_rotation_transform(
        rotations_scene_ego.unsqueeze(1),
        traversal_rectangle_ego.unsqueeze(0),
        result_einsum_prefix="ab",
    )
    # Translate each polygon group to ego position
    traversal_rectangles_scene = traversal_rectangles_scene + ego_poses.unsqueeze(1)

    return [geom.Polygon(polygon_points) for polygon_points in traversal_rectangles_scene.tolist()]


def calculate_traversal_error(
    predicted_trajectory_scene: torch.Tensor,
    observed_trajectory_scene: torch.Tensor,
    ego_mask: torch.Tensor,
    rotations_scene_ego: torch.Tensor,
    timestamps: torch.Tensor,
    prediction_timestamp: torch.Tensor,
    relevant_agents: torch.Tensor,
    visibility: torch.Tensor,
    horizons: list,
    param: dict,
    tmp_res_eps: float = 1e-5,
    relevant_agent_types: Optional[list] = None,
    agent_types: Optional[torch.Tensor] = None,
) -> dict:
    """This method implements the trajectory traversal error computation.

    The traversal error can be computed at different time horizons, i.e. only considering a predefined number of future
    timestemps assuming that the predicted trajectory reaches that horizon.

    :param predicted_trajectory_scene: Shape (batch_size x num_agents x num_timesteps x 2) Predicted trajectories
    :param observed_trajectory_scene: Shape (batch_size x num_agents x num_timesteps x 2) Observed true trajectories
    :param ego_mask: Shape (batch_size x num_agents) Mask of the ego vehicle of each batch
    :param timestamps: Shape (batch_size x num_timesteps) Timestamps of the trajectory data
    :param prediction_timestamp: Shape (batch_size,) All timestamps larger than the timestamps given here are considered
        for traversal computation. This is necessary because the first entry of timestamps might not correspond
        to the timestamp of the first entry in predicted / ground truth.
    :param relevant_agents: Shape (batch_size x num_agents) Binary integer tensor indicating which agent is
        relevant for error computation.
    :param visibility: Shape (batch_size x num_agents x num_timesteps) Boolean tensor indicating in which timesteps an
        agent is visible. It is assumed that all agents are always visible if this parameter is not set.
    :param horizons: A list of timepoints containing the traversal computation lookahead horizon.
    :param param: A dictionary of configuration parameters.
    :param tmp_res_eps: Epsilon value to account for temporal resolution. Has to be small enough such that two
        timesteps are never in a 2*tmp_res_eps interval and big enough to account for numerical errors (default 1e-5).
    :param relevant_agent_types: An optional list of relevant agent types.
    :param agent_types: Shape (batch_size x num_agents x num_types) Optional one-hot agent type information
    :param rotations_scene_ego: The 2D rotation matrix of agents to transform displacements into agents' frames.
    :returns: A dictionary containing the following entries:
        * traversal_miss_rate_partial: Shape (batch_size x num_agents x horizons x 1) Dict of zone traversal miss rates
          for different horizons indexed by horizon each containing per agent data which is nan in case of invalidity
          (e.g. because agent irrelevant). If an agent is predicted beyond the horizon for which we have ground truth,
          the agent is treated as being predicted at the horizon limit for which we have ground truth. Data for earlier
          horizons are also included to show prediction performance for the given trajectory at earlier horizons.
        * traversal_miss_rate_full: Shape (batch_size x num_agents x horizons x 1) Same as the above except that here
          all agents are considered invalid that have not been observed over the full horizon. Error values are only
          provided for the final horizon for the given trajectory. Data for earlier horizons are removed from these
          dicts in order to avoid mistaken comparisons between different horizons and calculations using values from
          different horizons.
        * traversal_miss_rate_partial_{agent_type}: Shape () Dict of traversal miss rates indexed by horizon for a
          specific agent type.
    """
    num_agents = predicted_trajectory_scene.shape[1]
    batch_size = predicted_trajectory_scene.shape[0]

    results = {}

    ego_poses = observed_trajectory_scene[ego_mask.bool(), :, :][..., 0, :]
    if param["fixed_ego_orientation"]:
        crossing_rectangles_scene = create_expected_traversal_rectangles(
            torch.eye(2, device=predicted_trajectory_scene.device).unsqueeze(0).expand(batch_size, -1, -1),
            ego_poses,
            param["predictor_normalization_scale"],
        )
    else:
        crossing_rectangles_scene = create_expected_traversal_rectangles(
            rotations_scene_ego, ego_poses, param["predictor_normalization_scale"]
        )

    # Populate per-location validity mask (shape  batch_size x num_agents x num_timesteps)
    agent_visibility = (visibility > 0).float()

    # Treat irrelevant agents as invisible
    relevant_agents_mask = relevant_agents[..., None]
    agent_visibility = (agent_visibility * relevant_agents_mask * ego_mask.logical_not().unsqueeze(-1)).float()

    # Treat agents of wrong type as invisible.
    if relevant_agent_types is not None:
        assert agent_types is not None, "agent_types cannot be None if relevant_agent_types is set"
        assert "agent_types" in param, "full agent types need to be specified if relevant_agent_types is set"

        # Mask containing all agents of relevant type (shape batch_size x num_agents)
        rel_types_mask = agent_types.new_zeros(agent_types.shape[0:-1], dtype=float)
        for cur_agent_type_id in relevant_agent_types:
            cur_agent_type_id_index = param["agent_types"].index(cur_agent_type_id)
            rel_types_mask += agent_types[..., cur_agent_type_id_index]

        rel_types_mask = rel_types_mask > 0
        agent_visibility *= rel_types_mask.unsqueeze(-1).expand_as(agent_visibility)

    for traversal in [
        "traversal_TP",
        "traversal_TN",
        "traversal_FP",
        "traversal_FN",
        "predicted_distances_from_crossing",
        "observed_distances_from_crossing",
        "invalid_agents_full",
        "total_positive",
    ]:
        results[traversal] = {}

    horizon_masks = get_horizon_masks(timestamps, prediction_timestamp, horizons, agent_visibility, tmp_res_eps)
    for cur_horizon in horizons:
        # Masks for all timepoints at or before/after desired horizon (shape batch_size x num_timesteps)
        # final_timepoint should have a value as long as the final trajectory timestamp is within one dt
        # of prediction_timestamp.
        final_timepoint = horizon_masks[cur_horizon]["final"]
        visible_in_horizon = horizon_masks[cur_horizon]["visible_in_horizon"]
        last_visible_in_horizon = horizon_masks[cur_horizon]["last_visible_in_horizon"]

        # Agents for which we don't have (full) trajectories observed (shape batch_size x num_agents)
        invalid_agents_full = (final_timepoint.unsqueeze(1) * last_visible_in_horizon).any(-1).logical_not()
        visible_in_horizon *= invalid_agents_full.logical_not().unsqueeze(-1)

        # Rotation and translation is not necessary since this metric is invariant to whole scene translation/rotation.
        predicted_distances_from_crossing = check_trajectory_traversal(
            predicted_trajectory_scene, crossing_rectangles_scene, visible_in_horizon
        )
        observed_distances_from_crossing = check_trajectory_traversal(
            observed_trajectory_scene, crossing_rectangles_scene, visible_in_horizon
        )

        # The minimum distance between the trajectory and the crossing zone when intersecting is always "0"
        predicted_intersects = cast(torch.Tensor, predicted_distances_from_crossing <= tmp_res_eps)
        observed_intersects = cast(torch.Tensor, observed_distances_from_crossing <= tmp_res_eps)

        results["traversal_TP"][cur_horizon] = torch.bitwise_and(predicted_intersects, observed_intersects)  # [4, 4, 1]
        results["traversal_TN"][cur_horizon] = (
            torch.bitwise_not(torch.bitwise_or(predicted_intersects, observed_intersects))
            * invalid_agents_full.logical_not()
        )
        traversal_xor = torch.bitwise_xor(predicted_intersects, observed_intersects)
        results["traversal_FN"][cur_horizon] = torch.bitwise_and(traversal_xor, observed_intersects)
        results["traversal_FP"][cur_horizon] = torch.bitwise_and(traversal_xor, torch.bitwise_not(observed_intersects))
        results["predicted_distances_from_crossing"][cur_horizon] = predicted_distances_from_crossing
        results["observed_distances_from_crossing"][cur_horizon] = observed_distances_from_crossing
        results["total_positive"][cur_horizon] = torch.count_nonzero(observed_intersects)
        results["invalid_agents_full"][cur_horizon] = invalid_agents_full

    return results
