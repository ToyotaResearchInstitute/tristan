from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from model_zoo.intent.create_networks import create_mlp
from model_zoo.intent.prediction_latent_factors import LatentFactors
from triceps.protobuf.protobuf_training_parameter_names import PARAM_FUTURE_TIMESTEPS, PARAM_MAX_AGENTS


def attention_factors_generator(factor_names, intermediate_dim, params):
    if params["latent_factors_use_linear_layers"]:
        internal_dims = []
    else:
        internal_dims = params["latent_factors_internal_dim"]
    output_dim = params["latent_factors_output_dim"]
    attention_dim = params["latent_factors_attention_dim"]
    full_attention = params["full_spatiotemporal_attention"]
    max_agents = params[PARAM_MAX_AGENTS]
    future_timesteps = params[PARAM_FUTURE_TIMESTEPS]
    num_attention_heads = params["latent_factors_num_attention_heads"]
    if params["use_multiagent_accelerated_decoder"]:
        cls = AttentionFactorsBatch
        # Move number of heads into output dimension so we only get one loop in the decoder
        params["latent_factors_output_dim"] = params["latent_factors_output_dim"] * num_attention_heads
        params["latent_factors_keys"] = ["0"]
    else:
        cls = AttentionFactors
    return cls(
        factor_names,
        intermediate_dim + output_dim * num_attention_heads,
        internal_dims,
        output_dim,
        attention_dim,
        future_timesteps,
        max_agents,
        num_attention_heads,
        full_attention,
    )


class AttentionFactors(LatentFactors):
    def __init__(
        self,
        factor_names: list,
        intermediate_dim: int,
        internal_dims: list,
        output_dim: int,
        attention_dim: int,
        future_timesteps: int,
        max_agents: int,
        num_attention_heads: int,
        full_attention: bool,
        attention_dropout_rate: float = 0.3,
    ):
        """Implement temporal and inter-agent attention.

        Args:
            factor_names: list
                The names of the factors. If the list size is larger than one, using multiple attention heads.
            intermediate_dim: int
                The intermediate dimension
            internal_dims: list
                The dimensions for the attention network value and attention mlp's.
            output_dim: int
                The output dimension of values.
            attention_dim: int
                The output dimension used for the attention inner products.
            future_timesteps: int
                The number of future time steps
            max_agents: int
                The number of agents
            num_attention_heads: int
                The number of attention heads
            full_attention: bool
                If set to True, attention is collected for every agent, at every past timestep.
                If False, only collect from last step's agents, and from self-attention.
            attention_dropout_rate: float
                The dropout rate.
        """
        super().__init__()
        self.factor_names = factor_names
        self.factors = nn.ModuleDict()
        for key in factor_names:
            self.factors[key] = nn.Linear(1, 1)
        self.intermediate_dim = intermediate_dim
        self.intermediate_outputs = OrderedDict()
        self.internal_dims = internal_dims
        self.future_timesteps = future_timesteps
        self.max_agents = max_agents
        self.full_attention = full_attention
        self.attention_dropout_rate = attention_dropout_rate
        self.states = []
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.num_attention_heads = num_attention_heads
        self.multi_agents = True
        self.create_attention_heads()

    def create_attention_heads(self):
        # Match the notation from "Attention is all you need" paper,
        # https://arxiv.org/abs/1706.03762
        if len(self.internal_dims) == 0:
            self.state_to_factors_k = nn.ModuleList(
                [nn.Linear(self.intermediate_dim, self.attention_dim) for i in range(self.num_attention_heads)]
            )
            self.state_to_factors_q = nn.ModuleList(
                [nn.Linear(self.intermediate_dim, self.attention_dim) for i in range(self.num_attention_heads)]
            )
            self.state_to_factors_v = nn.ModuleList(
                [nn.Linear(self.intermediate_dim, self.output_dim) for i in range(self.num_attention_heads)]
            )
        else:
            self.state_to_factors_k = nn.ModuleList(
                [
                    create_mlp(
                        self.intermediate_dim,
                        self.internal_dims + [self.attention_dim * self.num_attention_heads],
                        dropout_ratio=self.attention_dropout_rate,
                    )
                    for _ in range(self.num_attention_heads)
                ]
            )
            self.state_to_factors_q = nn.ModuleList(
                [
                    create_mlp(
                        self.intermediate_dim,
                        self.internal_dims + [self.attention_dim],
                        dropout_ratio=self.attention_dropout_rate,
                    )
                    for _ in range(self.num_attention_heads)
                ]
            )
            self.state_to_factors_v = nn.ModuleList(
                [
                    create_mlp(
                        self.intermediate_dim,
                        self.internal_dims + [self.output_dim],
                        dropout_ratio=self.attention_dropout_rate,
                    )
                    for _ in range(self.num_attention_heads)
                ]
            )

    def forward(self):
        return None

    def reset_generated_samples(self):
        self.intermediate_outputs = OrderedDict()

    def time_sample(
        self,
        current_agent_state: torch.Tensor,  # the current state, for agent i
        time_idx: int,
        t: torch.Tensor,
        stats: dict,
        agent_i: int,
        all_agent_states: OrderedDict,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        assert type(additional_info) is str and additional_info.isnumeric()
        head_i = int(additional_info)

        batch_size = current_agent_state[0].shape[1]
        qs = []
        vs = []
        k = self.state_to_factors_k[head_i](current_agent_state[0][0, ...])
        # The last time point that was already populated in predict_agent_state
        last_t = np.max(list(set(self.intermediate_outputs.keys()) - set([round(t.item(), 5)])) + list([-np.inf]))

        for t1 in self.intermediate_outputs:
            if np.isclose(t.item(), t1, 1e-2):
                continue
            if self.multi_agents:
                agents_keys = self.intermediate_outputs[t1].keys()
            else:
                agents_keys = [agent_i]
            for agent_j in agents_keys:
                if agent_j in self.intermediate_outputs[t1]:
                    if self.full_attention:
                        # Attend to all valid agents and time points
                        qs.append(self.state_to_factors_q[head_i](self.intermediate_outputs[t1][agent_j]))
                        vs.append(self.state_to_factors_v[head_i](self.intermediate_outputs[t1][agent_j]))
                    elif agent_i == agent_j or t1 == last_t:
                        # Attend to self-agent and all time points, or to all agents in the last timepoint.
                        qs.append(self.state_to_factors_q[head_i](self.intermediate_outputs[t1][agent_j]))
                        vs.append(self.state_to_factors_v[head_i](self.intermediate_outputs[t1][agent_j]))

        if len(qs) > 1:
            qs = torch.stack(qs, -1)
            vs = torch.stack(vs, -1)
            # Equation 1 from "Attention is all you need"
            weights = (qs * (k.unsqueeze(-1))).sum(-2).softmax(-1)
            result = (weights.unsqueeze(-2) * vs).sum(-1)
        else:
            result = current_agent_state[0].new_zeros([batch_size, self.output_dim])
        return result

    def predict_agent_state(
        self,
        inputs: dict,
        out: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        agent_i: int,
        t: int,  # indexed time
        time_point: float,
        batch_size: int,
        coordinate_decoder: torch.nn.modules.container.Sequential,
    ) -> None:
        # We already have an agent state for the current time point and *will not* modify it.
        # We will store (in self.intermediate_outputs) the state, to be reused later, in attention updates.
        if time_point not in self.intermediate_outputs:
            self.intermediate_outputs[time_point] = {}
        self.intermediate_outputs[time_point][agent_i] = state[0][0, ...]
        return None

    def generate_initial_factors(self, initial_state_tensor, stats):
        self.initial_state_tensor = initial_state_tensor
        return initial_state_tensor

    def set_initial_factors(self, initial_state_tensor: object):
        self.initial_state_tensor = initial_state_tensor


class AttentionFactorsBatch(AttentionFactors):
    """This function implement 'Attention is all you need' as latent factor."""

    def __init__(self, *args, **kwargs):
        super(AttentionFactorsBatch, self).__init__(*args, **kwargs)
        # The sequence is num_agents * future_timesteps
        self.sequence_length = self.max_agents * self.future_timesteps

        # d_k is the scaling factor in the paper, formula 1.
        self.d_k = torch.ones([1]) * np.sqrt(self.attention_dim)

    def create_attention_heads(self):
        self.state_to_factors_k = create_mlp(
            self.intermediate_dim,
            self.internal_dims + [self.attention_dim * self.num_attention_heads],
            dropout_ratio=self.attention_dropout_rate,
        )
        self.state_to_factors_q = create_mlp(
            self.intermediate_dim,
            self.internal_dims + [self.attention_dim * self.num_attention_heads],
            dropout_ratio=self.attention_dropout_rate,
        )
        self.state_to_factors_v = create_mlp(
            self.intermediate_dim,
            self.internal_dims + [self.output_dim * self.num_attention_heads],
            dropout_ratio=self.attention_dropout_rate,
        )
        self.state_to_residual = nn.Linear(self.intermediate_dim, self.output_dim * self.num_attention_heads, bias=True)
        self.layer_norm = nn.LayerNorm(self.num_attention_heads * self.output_dim, eps=1e-6)

    def reset_generated_samples(self):
        self.intermediate_outputs = OrderedDict()

    def generate_initial_factors(self, initial_state_tensor: torch.Tensor, stats):
        self.initial_state_tensor = initial_state_tensor

        # shape = [batch, agents, samples]
        shape = list(initial_state_tensor.shape[:-1])

        # Initialize tensor Query, Key, Value
        # After this block of code, the shape is:
        # Query [batch, samples, agents, future_timesteps, attention_dim]
        # Key [batch, samples, agents, future_timesteps, attention_dim]
        # Value [batch, samples, agents, future_timesteps, output_dim]

        # Swap (agents, samples) because we will later squash agents with the next dim.
        # Storing the seq_tensor_q this way, I think could minimize transpose cost
        shape[-1], shape[-2] = shape[-2], shape[-1]
        shape.append(self.future_timesteps)
        shape.append(self.attention_dim * self.num_attention_heads)

        # Initializing the query, key, and value tensors
        self.seq_tensor_q = initial_state_tensor.new_zeros(*shape)
        self.seq_tensor_k = torch.zeros_like(self.seq_tensor_q)

        shape[-1] = self.output_dim * self.num_attention_heads
        self.seq_tensor_v = initial_state_tensor.new_zeros(*shape)

        self.d_k = self.d_k.to(initial_state_tensor.device)
        return initial_state_tensor

    def set_initial_factors(self, initial_state_tensor: object):
        self.initial_state_tensor = initial_state_tensor

    @staticmethod
    def _merge_dims(tensor: torch.Tensor, dims_to_merge: Tuple[int, int]) -> torch.Tensor:
        """Merge two adjacent dims into one.

        tensor:
            The tensor to operate on.
        dims_to_merge: Tuple[int, int]
            The index of the two dimension to merge.

        return:
            New tensor with one dim less
        """
        dim1, dim2 = dims_to_merge[0], dims_to_merge[1]
        assert abs(dim1 - dim2) == 1, "dims_to_merge must be adjacent dims"
        shape = list(tensor.shape)
        shape[dim1] = shape[dim1] * shape[dim2]
        shape.pop(dim2)
        # [batch, samples, agents*future_timesteps, attention_dim]
        tensor = tensor.reshape(*shape)
        return tensor

    @staticmethod
    def _split_dims(tensor: torch.Tensor, dims_to_split: int, size_of_dim: int):
        """Split one dimension into two.

        tensor: torch.Tensor
            The tensor to split.
        dims_to_split: int
            The index of the dimension to split.
            The new dimension will be at dims_to_split+1
        size_of_dim:
            The length of the new dimension at index dims_to_split.

        Return:
            tensor with an additional dimension
        """
        shape = list(tensor.shape)
        shape.insert(dims_to_split, 0)
        if dims_to_split < 0:  # shift negative index after inserting behind
            dims_to_split -= 1
        shape[dims_to_split] = size_of_dim
        shape[dims_to_split + 1] = -1
        # [batch, samples, agents, future_timesteps, attention_dim]
        tensor = tensor.reshape(*shape)
        return tensor

    def time_sample(
        self,
        current_agent_state: torch.Tensor,
        time_idx: int,
        t: Optional[torch.Tensor] = None,
        stats: Optional[dict] = None,
        agent_i: Optional[int] = None,
        all_agent_states: Optional[OrderedDict] = None,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        # [batch, agents, samples, features]
        """Batch computation for multiple heads into one call of `time_sample`"""
        agent_state = current_agent_state[0].squeeze(0)
        residual = self.state_to_residual(agent_state)
        # transpose dim (agents, samples)
        agent_state = agent_state.transpose(1, 2)

        # [batch, samples, agents, attn_dim * heads]
        q = self.state_to_factors_q(agent_state)
        k = self.state_to_factors_k(agent_state)
        v = self.state_to_factors_v(agent_state)

        # Fill it in the sequence tensor at time t
        # [batch, samples, agents, future_timesteps, attn_dim * heads]
        self.seq_tensor_q[..., time_idx, :] = q
        self.seq_tensor_k[..., time_idx, :] = k
        self.seq_tensor_v[..., time_idx, :] = v

        # [batch, samples, agents, future_timesteps, heads, attn_dim]
        seq_tensor_q = self._split_dims(self.seq_tensor_q, -1, self.num_attention_heads)
        seq_tensor_k = self._split_dims(self.seq_tensor_k, -1, self.num_attention_heads)
        seq_tensor_v = self._split_dims(self.seq_tensor_v, -1, self.num_attention_heads)

        # squash (agents, future_timesteps) -> one dim (agents * future_timesteps), to form the sequence
        # [batch, samples, agents * future_timesteps, heads, attention_dim]
        seq_tensor_q = self._merge_dims(seq_tensor_q, (2, 3))
        seq_tensor_k = self._merge_dims(seq_tensor_k, (2, 3))
        seq_tensor_v = self._merge_dims(seq_tensor_v, (2, 3))

        # [batch, samples, heads, agents * future_timesteps, attention_dim]
        seq_tensor_q = seq_tensor_q.transpose(2, 3)
        seq_tensor_k = seq_tensor_k.transpose(2, 3)
        seq_tensor_v = seq_tensor_v.transpose(2, 3)

        # [batch, samples, heads, agents * future_timesteps, agents * future_timesteps]
        attention_matrix = torch.matmul(seq_tensor_q / self.d_k, seq_tensor_k.transpose(-1, -2))

        # Attention formula
        p_attn = F.softmax(attention_matrix, dim=-1)

        if self.attention_dropout_rate != 0:
            p_attn = F.dropout(p_attn, self.attention_dropout_rate)

        # [batch, samples, heads, agents * future_timesteps, output_dim]
        result = torch.matmul(p_attn, seq_tensor_v)

        # split the 2nd dim from (agents * future_timesteps) -> (agents, future_timesteps)
        # [batch, samples, heads, agents, future_timesteps, output_dim]
        result = self._split_dims(result, 3, self.max_agents)
        self.seq_tensor_q = self._split_dims(seq_tensor_q, 3, self.max_agents).clone()
        self.seq_tensor_k = self._split_dims(seq_tensor_k, 3, self.max_agents).clone()
        self.seq_tensor_v = self._split_dims(seq_tensor_v, 3, self.max_agents).clone()

        # [batch, samples, agents, future_timesteps, heads, output_dim]
        result = result.movedim(2, 4)
        self.seq_tensor_q = self._merge_dims(self.seq_tensor_q.movedim(2, 4), (-2, -1))
        self.seq_tensor_k = self._merge_dims(self.seq_tensor_k.movedim(2, 4), (-2, -1))
        self.seq_tensor_v = self._merge_dims(self.seq_tensor_v.movedim(2, 4), (-2, -1))

        # Extract the value for this time step
        # [batch, samples, agents, heads, output_dim]
        result = result[..., time_idx, :, :]
        # [batch, agents, samples, heads, output_dim]
        result = result.transpose(1, 2)

        # [batch, agents, samples, heads * output_dim]
        result = self._merge_dims(result, (-2, -1))

        # [batch, agents, samples, heads * output_dim]
        result += residual

        # [batch, agents, samples, heads * output_dim]
        result = self.layer_norm(result)
        assert not torch.any(torch.isnan(result))

        # [batch, agents, samples, heads * output_dim]
        return result

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
    ) -> None:
        return None
