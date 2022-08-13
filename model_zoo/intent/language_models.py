from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe, Vocab

from model_zoo.intent.create_networks import create_mlp
from model_zoo.intent.prediction_latent_factors import LatentFactors


class TokenGenerator(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        input_size: int,
        hidden_size: int,
        noise_dim: int,
        max_token_length: int,
        use_layer_norm: bool = False,
        use_mlp: bool = False,
        dropout_ratio: float = 0.0,
    ):
        """Create a token generator that generate a sequence of tokens given a trajectory embedding.

        Parameters
        ----------
        vocab: Vocab
            The vocabulary for generation.
        input_size: int
            The dimensionality of the input trajectory embedding for one agent.
        hidden_size: int
            The dimensionality of the LSTM hidden state.
        noise_dim: int
            The dimension of the noise vector.
        max_token_length: int
            The maximum length of token sequence to be generated.
        use_layer_norm: bool
            Uses layer norm on the hidden inputs when set to true (default: False).
        use_mlp: bool
            Uses MLP instead of linear layer for generation. (default: False)
        dropout_ratio: float
            Dropout ratio for the output.
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.use_layer_norm = use_layer_norm
        self.max_token_length = max_token_length
        self.hidden_size = hidden_size

        self.vocab = vocab
        self.token_size = len(self.vocab)

        self.generator = nn.LSTM(input_size=hidden_size + noise_dim, hidden_size=hidden_size)
        if use_mlp:
            self.embed_input = create_mlp(input_size, [hidden_size], dropout_ratio=dropout_ratio)
            self.token_linear = create_mlp(hidden_size, [hidden_size, self.token_size], dropout_ratio=dropout_ratio)
        else:
            self.embed_input = nn.Linear(input_size, hidden_size)
            self.token_linear = nn.Linear(hidden_size, self.token_size)

        if use_layer_norm:
            self.normalize = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, traj_embedding: torch.Tensor):
        """Rollout the generator to produce distribution of tokens till max token length reached.

        Parameters
        ----------
        traj_embedding: torch.Tensor
            The input trajectory embedding of one agent of shape (batch_size, embed_size)

        Returns
        -------
        out_tokens: torch.Tensor
            The sequence of generated token distributions.
        """
        noise_shape = list(traj_embedding.shape)
        noise_shape[-1] = self.noise_dim
        noise = torch.randn(noise_shape, device=traj_embedding.device)
        embeded_input = self.embed_input(traj_embedding)
        input = torch.cat([embeded_input, noise], -1)
        out_tokens_shape = [self.max_token_length] + noise_shape[:-1] + [self.token_size]
        out_tokens = torch.zeros(out_tokens_shape).to(traj_embedding.device)
        hidden_shape = [1] + out_tokens_shape[1:-1] + [self.hidden_size]
        states = (
            torch.zeros(hidden_shape).view(1, -1, self.hidden_size).to(traj_embedding.device),
            torch.zeros(hidden_shape).view(1, -1, self.hidden_size).to(traj_embedding.device),
        )
        for t in range(self.max_token_length):
            if self.use_layer_norm:
                cell_state_normalized = self.normalize(states[1])
                states_normalized = (states[0], cell_state_normalized)
            else:
                states_normalized = states
            out, states = self.generator(input.unsqueeze(0).view(1, -1, input.shape[-1]), states_normalized)
            out_shape = list(input.shape[:-1]) + [out.shape[-1]]
            out_tokens[t, :] = self.token_linear(out.squeeze(0).view(out_shape))
        return out_tokens


class LanguageFactors(LatentFactors):
    def __init__(self, params: dict, vocab: Vocab):
        """LanguageFactors class that encapsulate all language-based operations in the decoder.

        Parameters
        ----------
        params: dict
            The parameters for language-based operations.
        vocab: Vocab
            The vocabulary object that maps trajectory descriptions to indices.
        """

        super().__init__()
        self.vocab = vocab
        self.params = params
        self.max_token_length = params["max_token_length"]
        self.use_layer_norm = params["use_layer_norm"]

        self.token_generator = TokenGenerator(
            vocab=vocab,
            input_size=params["generator_input_size"],
            hidden_size=params["generator_hidden_size"],
            noise_dim=params["generator_noise_dim"],
            max_token_length=params["max_token_length"],
            use_layer_norm=params["use_layer_norm"],
            use_mlp=params["use_mlp"],
            dropout_ratio=params["dropout_ratio"],
        )
        self.init_token_embedding()
        self.token_encoder = nn.LSTM(input_size=params["encoder_input_size"], hidden_size=params["encoder_hidden_size"])
        if self.use_layer_norm:
            self.token_normalize = nn.LayerNorm(params["encoder_hidden_size"], elementwise_affine=False)
        self.attention_scoring = nn.Linear(params["agent_state_size"], params["encoder_hidden_size"])
        if params["use_mlp"]:
            self.agent_attention = create_mlp(
                params["encoder_input_size"],
                [params["encoder_input_size"], params["max_agents"]],
                dropout_ratio=params["dropout_ratio"],
            )
            self.agent_token_encoder = create_mlp(
                params["agent_state_size"],
                [params["agent_state_size"], params["encoder_input_size"]],
                dropout_ratio=params["dropout_ratio"],
            )
        else:
            self.agent_attention = nn.Linear(params["encoder_input_size"], params["max_agents"])
            self.agent_token_encoder = nn.Linear(params["agent_state_size"], params["encoder_input_size"])

        # Caches the token embeddings, dimension is (max_token_length, batch_size, encoder_hidden_size).
        self.embeded_tokens = None

        # Caches the values for upating the output/stats later.
        self.attention_weights = OrderedDict()
        self.agent_attention_weights = OrderedDict()
        self.generated_tokens_dist = []
        self.decoded_tokens = []

    def init_token_embedding(self):
        """Initialize token embedding"""
        if self.params["use_pretrained_word_embedding"]:
            # Initialize with pretrained GloVe word embeddings
            embedding_dim = 50
            pretrained_vectors = GloVe(name="6B", dim=str(embedding_dim))
            pretrained_embedding = pretrained_vectors.get_vecs_by_tokens(self.vocab.get_itos())
            self.token_embedding = nn.Sequential(
                OrderedDict(
                    [
                        ("embed", nn.Embedding(len(self.vocab), embedding_dim, padding_idx=self.vocab["<pad>"])),
                        ("fc", nn.Linear(embedding_dim, self.params["encoder_input_size"])),
                    ]
                )
            )

            self.token_embedding.embed.weight.data.copy_(pretrained_embedding)
            for param in self.token_embedding.embed.parameters():
                param.requires_grad = False
        else:
            # Initialize embedding layer for synthetic language.
            embedding_dim = self.params["encoder_input_size"]
            self.token_embedding = nn.Sequential(
                OrderedDict(
                    [
                        ("embed", nn.Embedding(len(self.vocab), embedding_dim, padding_idx=self.vocab["<pad>"])),
                        ("fc", create_mlp(embedding_dim, [embedding_dim], dropout_ratio=self.params["dropout_ratio"])),
                    ]
                )
            )

    def forward(self):
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
        """Reset the values."""
        self.agent_attention_weights = OrderedDict()
        self.attention_weights = OrderedDict()
        self.generated_tokens_dist = []
        self.decoded_tokens = []

    def generate_initial_factors(self, initial_state_tensor: torch.Tensor, stats: dict) -> torch.Tensor:
        """Generate tokens and their embeddings for predicting future trajectories.

        Parameters
        ----------
        initial_state_tensor: torch.Tensor
            The intermediate state of the target agents. The dimension is (batch_size, intermediate_dim).
        stats: dict
            The statistics dictionary for additional latent factor statistics. It includes
            language_tokens - token distribution for computing the loss.
            decoded_tokens - the decoded token texts from sampled onehot vectors.
        """
        super().generate_initial_factors(initial_state_tensor, stats)
        # Generates a sequence of token distributions.
        token_dist = self.token_generator(initial_state_tensor)
        # Samples tokens from Gumbel-Softmax.
        tokens_oneohot = nn.functional.gumbel_softmax(token_dist, hard=True)
        if self.params["compute_information_gain"]:
            # Pad the second half of samples so that we can compute entropy w/ and w/o language.
            n_samples = token_dist.shape[3]
            tokens_oneohot[..., int(n_samples / 2) :, :] = 0
            tokens_oneohot[..., int(n_samples / 2) :, self.vocab.get_stoi()["<pad>"]] = 1
        # Re-encodes tokens into embeddings.
        self.embeded_tokens = self.token_embedding(torch.argmax(tokens_oneohot, dim=-1))
        self.decoded_tokens = torch.argmax(tokens_oneohot.cpu().detach().permute(1, 2, 3, 0, 4), dim=-1)
        self.generated_tokens_dist = token_dist
        return self.embeded_tokens

    def set_initial_factors(self, initial_state_tensor: torch.Tensor) -> None:
        self.embeded_tokens = initial_state_tensor

    def encode_tokens(self, embeded_tokens):
        device = embeded_tokens.device
        state_shape = [1] + list(embeded_tokens.shape[1:-1]) + [self.params["encoder_hidden_size"]]
        states = (
            torch.zeros(state_shape).view(1, -1, self.params["encoder_hidden_size"]).to(device),
            torch.zeros(state_shape).view(1, -1, self.params["encoder_hidden_size"]).to(device),
        )
        encoded_tokens_shape = list(embeded_tokens.shape[:-1]) + [self.params["encoder_hidden_size"]]
        encoded_tokens = torch.zeros(encoded_tokens_shape).to(device)
        for t in range(self.max_token_length):
            if self.use_layer_norm:
                cell_state_normalized = self.token_normalize(states[1])
                states_normalized = (states[0], cell_state_normalized)
            else:
                states_normalized = states
            input = embeded_tokens[t].unsqueeze(0).view(1, -1, self.params["encoder_input_size"])
            encoded_token, states = self.token_encoder(input, states_normalized)
            encoded_tokens[t] = encoded_token.squeeze(0).view(encoded_tokens_shape[1:])
        return encoded_tokens

    def generate_agent_token_embedding(self, all_agent_states: list, agent_i: int):
        """If a generated token is referring to an agent, compute the token embdding using attention over
        the related agent states.

        Parameters
        ----------
        all_agent_states: list
            The last lstm state for all agents.
        agent_id: int
            The index of the current agent.

        Returns
        -------
        token_emb: torch.Tensor
            The updated token embeddings.
        """
        device = self.embeded_tokens.device
        n_tokens = self.embeded_tokens.shape[0]
        embeded_tokens = self.embeded_tokens.clone()
        agent_state_shape = [1] + list(embeded_tokens.shape[1:-1]) + [all_agent_states[0].shape[-1]]
        agent_states = all_agent_states[0].clone().view(agent_state_shape)
        n_agents = agent_states.shape[2]
        agent_states = agent_states.repeat(n_agents, 1, 1, 1, 1).permute(1, 0, 3, 2, 4)
        for token_idx in range(n_tokens):
            # Creates mask for tokens that refer to other agents.
            num_agents = self.embeded_tokens.shape[2]
            # The last few tokens are agents.
            mask = self.decoded_tokens[..., token_idx].ge(len(self.vocab) - num_agents).int().unsqueeze(-1).to(device)
            if torch.sum(mask) == 0:
                # Skip if no token refers to other agents.
                continue
            # Computes the updated token embedding using attention over agent states.
            weights = self.agent_attention(self.embeded_tokens[token_idx]).softmax(-1)
            saved_weights = mask * weights.detach()
            for sample_i in range(saved_weights.shape[2]):
                if sample_i not in self.agent_attention_weights:
                    self.agent_attention_weights[sample_i] = OrderedDict()
                if token_idx not in self.agent_attention_weights[sample_i]:
                    self.agent_attention_weights[sample_i][token_idx] = OrderedDict()
                for agent_i in range(saved_weights.shape[1]):
                    if agent_i not in self.agent_attention_weights[sample_i][token_idx]:
                        self.agent_attention_weights[sample_i][token_idx][agent_i] = []
                    self.agent_attention_weights[sample_i][token_idx][agent_i].append(
                        saved_weights[:, agent_i, sample_i, ...]
                    )
            weighted_states = (weights.unsqueeze(-1) * agent_states).sum(-2)
            updated_emb = self.agent_token_encoder(weighted_states)
            embeded_tokens[token_idx] = torch.where(mask > 0, updated_emb, embeded_tokens[token_idx])
        return embeded_tokens

    def time_sample(
        self,
        current_agent_state: Tuple[torch.Tensor, torch.Tensor],
        time_idx: int,
        t: torch.Tensor,
        stats: dict,
        agent_i: int,
        all_agent_states: list,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        """This implementation overrides the original method and is based on Luong attention.
        It is possible to use other attention mechanisms.
        """
        n_tokens = self.embeded_tokens.shape[0]
        current_agent_state = current_agent_state[0].squeeze(0)
        agent_state_shape = list(self.embeded_tokens.shape[1:-1]) + [current_agent_state[0].shape[-1]]
        current_agent_state = current_agent_state.view(agent_state_shape)
        if self.params["drop_output"]:
            output_shape = list(current_agent_state.shape[:-1]) + [self.params["encoder_hidden_size"]]
            return torch.zeros(output_shape)
        if self.params["ablate_attention"]:
            encoded_tokens = self.encode_tokens(self.embeded_tokens)
            return encoded_tokens[-1]
        # Generate embeddings for agent tokens and encode it.
        if self.params["ablate_agent_attention"]:
            embeded_tokens = self.embeded_tokens
        else:
            embeded_tokens = self.generate_agent_token_embedding(all_agent_states, agent_i)
        encoded_tokens = self.encode_tokens(embeded_tokens)
        # Uses a bilinear function for attention scoring.
        attn_scoring = self.attention_scoring(current_agent_state)
        attn_scoring = attn_scoring.view(-1, self.params["encoder_hidden_size"])
        encoded_tokens = encoded_tokens.view(n_tokens, -1, encoded_tokens.shape[-1])
        attn_prob = torch.bmm(attn_scoring.repeat(n_tokens, 1, 1), encoded_tokens.permute(0, 2, 1))
        # Takes diagonal to only consider an agent's association with its own tokens and then switches to batch-first.
        attn_weights = F.softmax(torch.diagonal(attn_prob, dim1=1, dim2=2).permute(1, 0), dim=1)
        # Combines with the token embeddings to generate the context tensor.
        context = torch.bmm(attn_weights.unsqueeze(1), encoded_tokens.permute(1, 0, 2))
        context_shape = list(current_agent_state.shape[:-1]) + [self.params["encoder_hidden_size"]]
        # Save attention weights
        att_shape = list(current_agent_state.shape[:-1]) + [n_tokens]
        saved_weights = attn_weights.view(att_shape).detach()
        for sample_i in range(saved_weights.shape[2]):
            if sample_i not in self.attention_weights:
                self.attention_weights[sample_i] = OrderedDict()
            for agent_i in range(saved_weights.shape[1]):
                if agent_i not in self.attention_weights[sample_i]:
                    self.attention_weights[sample_i][agent_i] = []
                self.attention_weights[sample_i][agent_i].append(saved_weights[:, agent_i, sample_i, ...])
        return context.squeeze(1).view(context_shape)

    def update_stats(self, stats: list, sample_i: int) -> None:
        if sample_i is None:
            stats["attention_weights"] = self.attention_weights.copy()
            stats["agent_attention_weights"] = self.agent_attention_weights.copy()
            stats["language_tokens"] = torch.stack(self.generated_tokens_dist).permute(1, 2, 0, 3)
            stats["decoded_tokens"] = self.decoded_tokens
        else:
            stats["attention_weights"] = {}
            if sample_i in self.attention_weights:
                stats["attention_weights"] = self.attention_weights[sample_i].copy()
            stats["agent_attention_weights"] = {}
            if sample_i in self.agent_attention_weights:  # Only set it when it has agent tokens
                stats["agent_attention_weights"] = self.agent_attention_weights[sample_i].copy()
            stats["language_tokens"] = self.generated_tokens_dist[:, :, :, sample_i, ...]
            # Update the decoded tokens.
            # The dimension of decoded_tokens is agent first: [num_agents, batch, tokens].
            tokens_agent = []
            for agent_i in range(self.decoded_tokens.shape[1]):
                tokens_batch = []
                for batch_i in range(self.decoded_tokens.shape[0]):
                    tokens = [
                        self.vocab.get_itos()[token_i]
                        for token_i in self.decoded_tokens[batch_i][agent_i][sample_i]
                        if token_i != self.vocab["<pad>"]
                    ]
                    tokens_batch.append(tokens)
                tokens_agent.append(tokens_batch)
            stats["decoded_tokens"] = tokens_agent
