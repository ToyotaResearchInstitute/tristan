import torch
import torch.nn as nn


class AgentTemporalModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_agent_types: int, use_layer_norm: bool = False):
        """
        Creates a temporal model to capture temporal sequences, can be agent type-specific.
        This implementation is type-agnostic -- basically an LSTM for encoders/decoders.

        Parameters
        ----------
        input_size : int,
            the dimensionality of the input.
        hidden_size : int
            the dimensionality of the hidden state.
        num_agent_types : int
            the number of agent types.
        use_layer_norm : bool
            Uses layer norm on the hidden inputs when set to true, (default: False)
        """
        super().__init__()
        self.temporal_model = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        if use_layer_norm:
            self.normalize = nn.LayerNorm(hidden_size, elementwise_affine=False)
        else:
            self.normalize = lambda x: x

    def forward(self, inputs, hidden_state, agent_types):
        """Carries out forward pass through agent temporal model

        Parameters
        ----------
        inputs : torch.Tensor
            rnn input of shape (time_step(1) x batch_size x input_size)
        hidden_state : torch.Tensor
            tuple containing hidden state and cell state each of shape (1 x batch_size x hidden_size)
        """
        cell_state_normalized = self.normalize(hidden_state[1])
        hidden_state_normalized = (hidden_state[0], cell_state_normalized)
        return self.temporal_model(inputs, hidden_state_normalized)


class TypeConditionedTemporalModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_agent_types: int):
        """
        Creates a temporal model to capture temporal sequences, can be agent type-specific.
        This implementation is type-specific -- a different LSTM is used for each agent type.

        Parameters
        ----------
        input_size : int
            the dimensionality of the input.
        hidden_size : int
            the dimensionality of the hidden state.
        num_agent_types : int
            the number of agent types.
        """
        super().__init__()
        self.temporal_models = nn.ModuleDict()
        for i in range(num_agent_types):
            self.temporal_models[str(i)] = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor, agent_types: torch.Tensor):
        """Runs temporal model with a different LSTM for different agent types

        Parameters
        ----------
        inputs: torch.Tensor of size 1 x batch_size x input_size
        hidden_state a 2-tuple of torch.Tensor of size 1 x batch_size x hidden_dim
        agent_types torch.Tensor of size 1 x batch_size x num_agent_types

        Returns
        -------

        """
        for key_i, key in enumerate(self.temporal_models):
            result_ = self.temporal_models[key](inputs, hidden_state)
            agents = agent_types[:, int(key)] > 0
            if key_i == 0:
                result = list(result_)
                result[1] = list(result[1])
                result[0] = result[0].clone() * 0
                result[1][0] = result[1][0].clone() * 0
                result[1][1] = result[1][1].clone() * 0

            result[0][agents.unsqueeze(1).unsqueeze(0).repeat(1, 1, result[0].shape[2])] = result_[0][
                agents.unsqueeze(1).unsqueeze(0).repeat(1, 1, result[0].shape[2])
            ].clone()

            for i in range(2):
                if torch.any(agents):
                    result[1][i][agents.unsqueeze(1).unsqueeze(0).repeat(1, 1, result[1][i].shape[2])] = result_[1][i][
                        agents.unsqueeze(1).unsqueeze(0).repeat(1, 1, result[1][i].shape[2])
                    ].clone()
        assert result is not None
        return result
