from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, OrderedDict, Tuple, Union

import torch
from torch import nn

from model_zoo.intent.create_networks import create_mlp


class LatentFactors(ABC, nn.Module):
    """
    Interface for predictions that use temporal latent factors representation. The latent factors capture aspects of the
    state of the roll-out, and feed the downstream decoding as an inductive bias.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        return None

    @abstractmethod
    def generate_initial_factors(self, initial_state_tensor: torch.Tensor, stats: dict) -> object:
        """Generate the initial latent factor representation. This method should be called before running prediction
        for the first time step, to generate a state.
        This is useful for representations that have a state that's updated for every time point, or are generated once,
        and conditioned on, such as language tokens that underlie a prediction.

        Parameters
        ----------
        initial_state_tensor : torch.Tensor
            The intermediate state of the target agents. The dimension is (batch_size, intermediate_dim).
        stats : dict
            The statistics dictionary for additional latent factor statistics.

        Returns
        -------
        initial factors value : object
            The initial factors state of the latent factors, can be any object.
            This object is stored and set by the decoder (using set_initial_factors),
            in order to update every agent at every timepoint.
        """

    @abstractmethod
    def set_initial_factors(self, initial_state_tensor: object) -> None:
        """Set initial factors. This is used to set a saved initial factors state.

        Parameters
        ----------
        initial_state_tensor : object
            The intermediate state of the latent factors object.
        """

    @abstractmethod
    def time_sample(
        self,
        current_agent_state: Tuple[torch.Tensor, torch.Tensor],
        time_idx: int,
        t: torch.Tensor,
        stats: dict,
        agent_i: int,
        all_agent_states: Union[OrderedDict, list],
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        """Sample a state contribution tensor at time t.
        This sampled value can be based on explicit duration factors, hybrid automata, language tokens, etc.

        Parameters
        ----------
        current_agent_state : Tuple[torch.Tensor, torch.Tensor]
            The last state of the target agents. Two tensors of dimension (batch_size, lstm_dim).
        time_idx : int
            The index of the current time.
            NOTE: This and the next param (t) are consistent. You are feel to chose which to use.
            Just document which one you are using, time_idx or t.
        t : torch.Tensor
            The time point to generate the factor. The dimension is (batch_size, 1). This is time_idx*time_step_size
        agent_i: int
            The index of the agent.
        all_agent_states: OrderedDict or list
            The last lstm state of all agents.
            If it is a dict, each value is a tuple of two tensors of dimension (batch_size, lstm_dim).
            If it is a list, it has two tensors of dimension (batch_size, num_agents, lstm_dim).
        additional_info : Optional[object]
            Additional information for the sampling

        Returns
        -------
        sample: torch.Tensor
            The state contribution tensor at time t.
        """

    def get_augmented_state_dim(self):
        """
        Get additional state dim.
        """
        return 0

    def reset_generated_samples(self):
        """
        Reset generated samples.
        """

    def update_generated_samples(self, res_traj: torch.Tensor):
        """
        Update the set generated samples, for every sample instantiation. Allows trajectory samples to be generated
        consecutively, e.g. for neural adaptive sampling, as in the HYPER algorithm.

        Parameters
        ----------
        res_traj : torch.Tensor
            Predicted trajectory sample.
        """

    def overwrite_dropouts(self):
        """
        Overwrite dropouts with specific values

        Parameters
        ----------
        state_dropout_p : float
            Dropout value for state vector.
        additional_factors_dropout_p : float
            Dropout value for additional factors
        """
        return None

    def initialize_latent_data(
        self,
        batch_size: int,
        device: torch.device,
        inputs: dict,
        num_agents: int,
        num_timepoints: int,
        additional_params: dict,
    ) -> None:
        """
        Initialize latent data before prediction starts.

        Parameters
        ----------
        batch_size : int
            Batch size.
        device : str
            Model device.
        inputs : dict
            Dictionary of inputs.
        num_agents : int
            Number of agents.
        num_timepoints : int
            Number of time points.
        """

    def augment_state(self, current_position: torch.Tensor, t: int, agent_i: int) -> torch.Tensor:
        """
        Augment agent state with additional info.

        Parameters
        ----------
        current_position : torch.Tensor
            Current agent position.
        t : int
            Timestep.
        agent_i : int
            Agent index.

        Returns
        -------
        current_state : torch.Tensor
            Updated agent state (e.g. positions + modes for the HYPER algorithm).
        """
        return None

    def overwrite_initial_lstm_state(self, intermediate_output: torch.Tensor, i: int, t: int) -> torch.Tensor:
        """
        Overwrite initial lstm state at decoder when t=0.

        Parameters
        ----------
        intermediate_output : torch.Tensor
            Intermediate output from LSTM.
        i : int
            Agent index.
        t : int
            Time index.

        Returns
        -------
        state : torch.Tensor
            Updated state.
        """
        return None

    def predict_agent_state(
        self,
        inputs: dict,
        out: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        agent_i: int,
        t: int,
        time_point: float,
        batch_size: int,
        coordinate_decoder: torch.nn.modules.container.Sequential,
    ) -> torch.Tensor:
        """
        Predict agent state for agent i at time step t. If the function is populated and returns a non-None value, it is
        assumed to be the predicted agent coordinate, and any further coordinate decoding from the state vector will be
        skipped.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs.
        out : torch.Tensor
            LSTM output.
        state: Tuple[torch.Tensor, torch.Tensor]
            LSTM state.
        agent_i : int
            Agent index.
        t : int
            Time index.
        time_point: float
            The timepoint
        batch_size : int
            Batch size.
        coordinate_decoder : torch.nn.modules.container.Sequential
            Coordinate decoder.

        Returns
        -------
        generated_i_t : torch.Tensor
            Generated coordinate/state at time t for agent i.
        """
        return None

    def update_stats(self, stats: list, sample_i: int) -> None:
        """
        Update stats from latent predictions. These can be used to define additional internal information during the
        latent factor decoding.

        Parameters
        ----------
        stats : dict
            Dictionary of statistics.
        sample_i: int
            The sample index to update stats.

        Returns
        -------
        stats : dict
            Updated dictionary of statistics.
        """


class ConstantTemporalFunction(nn.Module):
    """
    Temporal latent factors that are a constant value. Mostly as a test.
    """

    def __init__(self):
        super().__init__()
        self.value = nn.Linear(1, 1, bias=False)
        self.out_features = self.value.out_features

    def forward(self):
        return None

    def time_sample(
        self,
        current_agent_state: Tuple[torch.Tensor, torch.Tensor],
        t: torch.Tensor,
        agent_i: int,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        assert isinstance(t, torch.Tensor)
        batch_t = t.unsqueeze(0).repeat(current_agent_state[0][0, :, :1].shape)
        return self.value(batch_t.unsqueeze(-1)).squeeze(-1)


class ExplicitDurationTruncatedFunction(nn.Module):
    """
    Explicit duration temporal latent factors, truncated sequence of upto scalar-valued N segments.
    """

    def __init__(self, intermediate_dim, internal_dims, num_segments, transition_scale):
        super().__init__()
        self.duration_param_dim = 3  # duration log mean, duration log standard deviation, log probability
        self.num_segments = num_segments
        self.internal_dims = internal_dims
        self.intermediate_dim = intermediate_dim
        # Scale - how fast (in seconds) is the transition of each latent factor.
        self.transition_scale = transition_scale

        # log-normal
        self.segment_param_module = create_mlp(
            input_dim=self.intermediate_dim, layers_dim=self.internal_dims + [self.duration_param_dim], dropout_ratio=0
        )

        self.value = nn.Linear(1, 1, bias=False)
        self.out_features = self.value.out_features
        self.initial_state_tensor = None

    def forward(self):
        return None

    def generate_initial_factors(self, initial_state_tensor, stats) -> object:
        self.initial_state_tensor = initial_state_tensor
        return self.initial_state_tensor

    def set_initial_factors(self, initial_state_tensor: object) -> None:
        self.initial_state_tensor = initial_state_tensor

    def time_sample(
        self,
        current_agent_state: Tuple[torch.Tensor, torch.Tensor],
        time_idx: int,
        t: torch.Tensor,
        stats: dict,
        agent_i: int,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        assert isinstance(t, torch.Tensor)
        cumulative_durations, _, _, values = self.get_segment_stats(self.initial_state_tensor)
        # assume last segment started way back. (5 seconds before current time)
        cumulative_durations = torch.nn.functional.pad(cumulative_durations, (1, 0, 0, 0), value=-5)

        batch_t = t.repeat(cumulative_durations.shape)

        dt = cumulative_durations - batch_t
        weights = (dt / self.transition_scale).sigmoid()
        weights = weights[..., 1:] - weights[..., :-1]
        weights = weights / weights.sum(dim=-1, keepdim=True)
        sample = (weights * values).sum(dim=-1).unsqueeze(-1)
        return sample

    @lru_cache(maxsize=512)
    def get_segment_stats(self, intermediate_state):
        def n_copies(tensor, num_copies):
            return tensor.repeat(num_copies, *(1 for _ in range(tensor.dim())))

        def permute_first_to_last(tensor):
            return tensor.permute(*(range(1, tensor.dim())), 0)

        segment_params = self.segment_param_module(intermediate_state)

        base_randomness = torch.normal(0, 1, [self.num_segments], device=intermediate_state.device)
        durations = n_copies(segment_params[..., 1], self.num_segments)
        while base_randomness.dim() < durations.dim():
            base_randomness.unsqueeze_(1)
        durations = (durations * base_randomness + segment_params[..., 0]).clamp(-5, 5).exp()
        durations = permute_first_to_last(durations)
        cumulative_durations = torch.cumsum(durations, -1)

        value_prob = segment_params[..., 2].exp() / (1.0 + segment_params[..., 2].exp())
        probs = n_copies(value_prob, self.num_segments)
        probs = permute_first_to_last(probs)
        # sample values - binary for now
        prob2 = torch.rand(probs.shape, device=probs.device)
        values = (prob2 < probs).float()
        # [batch, ..., num_segments] for all the return values
        return cumulative_durations, durations, probs, values


class ExplicitDurationTruncatedFactors(LatentFactors):
    def __init__(self, factor_names, intermediate_dim, internal_dims, num_segments, transition_scale):
        super().__init__()
        self.factor_names = factor_names
        self.intermediate_dim = intermediate_dim
        self.factors = nn.ModuleDict()
        self.internal_dims = internal_dims
        for key in self.factor_names:
            self.factors[key] = ExplicitDurationTruncatedFunction(
                intermediate_dim, internal_dims, num_segments=num_segments, transition_scale=transition_scale
            )

    def forward(self):
        return None

    def get_dimensionality(self, key):
        return self.factors[key].out_features

    def generate_initial_factors(self, initial_state_tensor, stats):
        for key in self.factor_names:
            self.factors[key].generate_initial_factors(initial_state_tensor, stats)

    def set_initial_factors(self, initial_state_tensor: object):
        pass

    def time_sample(
        self,
        current_agent_state: Tuple[torch.Tensor, torch.Tensor],
        time_idx: int,
        t: torch.Tensor,
        stats: dict,
        agent_i: int,
        all_agent_states: OrderedDict,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        res = 0.0
        for key in self.factor_names:
            factor_res = self.factors[key].time_sample(current_agent_state, time_idx, t, agent_i, key)
            res += factor_res
        return res


class ConstantLatentFactors(LatentFactors):
    def __init__(self, factor_names, intermediate_dim):
        super().__init__()
        self.factor_names = factor_names
        self.factors = nn.ModuleDict()
        for key in self.factor_names:
            self.factors[key] = ConstantTemporalFunction()

    def forward(self):
        return None

    def generate_initial_factors(self, initial_state_tensor: torch.Tensor, stats: dict) -> object:
        return None

    def set_initial_factors(self, initial_state_tensor: object) -> None:
        pass

    def time_sample(
        self,
        current_agent_state: Tuple[torch.Tensor, torch.Tensor],
        time_idx: int,
        t: torch.Tensor,
        stats: dict,
        agent_i: int,
        all_agent_states: OrderedDict,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        key = additional_info
        return self.factors[key].time_sample(current_agent_state, t, agent_i)


def constant_latent_factors_generator(factor_names, intermediate_dim, params):
    return ConstantLatentFactors(factor_names, intermediate_dim)


def explicit_duration_factors_generator(factor_names, intermediate_dim, params):
    num_segments = params["explicit_duration_factors_num_segments"]
    transition_scale = params["explicit_duration_factors_transition_scale"]
    internal_dims = params["latent_factors_internal_dim"]
    return ExplicitDurationTruncatedFactors(
        factor_names, intermediate_dim, internal_dims, num_segments, transition_scale
    )
