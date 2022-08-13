import json
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

from intent.multiagents.latent_factors import compute_semantic_costs


class PredictionModelCodec:
    """
    A class wrapper for generator and discriminator-based predictors.

    Parameters
    ----------
    models : dict
        A dictionary of models.
    params : dict
        A dictionary of parameters.
    intermediate_dim : int
        Intermediate dimensions.
    additional_structure_callbacks : list
        List of additional structure callbacks relevant to cost (i.e. hybrid, language).
    additional_model_callbacks : list
        List of additional model callbacks (i.e. hybrid).
    """

    def __init__(
        self,
        models: dict,
        params: dict = {},
        intermediate_dim: int = 0,
        additional_structure_callbacks: list = [],
        additional_model_callbacks: list = [],
    ) -> None:
        """
        Initialize class parameters.
        """
        super().__init__()
        self.params = params
        self.dropout_ratio = params["dropout_ratio"]

        # Get flags.
        # Flag to activate latent factor.
        self.use_latent_factors = params["use_latent_factors"]
        # Flag to activate latent factor.
        self.use_semantics = params["use_semantics"]

        # Flag to activate discriminator.
        self.use_discriminator = params["use_discriminator"]

        self.additional_model_callbacks = additional_model_callbacks
        self.additional_structure_callbacks = additional_structure_callbacks

        # Load models
        self.encoder = models["encoder_model"]
        self.decoder = models["decoder_model"]
        if self.use_discriminator:
            self.discriminator_encoder = models["discriminator_encoder"]
            self.discriminator_future_encoder = models["discriminator_future_encoder"]
            self.discriminator_head = models["discriminator_head"]

        # Load latent factors.
        self.latent_factors = None
        if self.use_latent_factors:
            self.latent_factors_generator = self.params["latent_factors_generator"]
            self.latent_factors_keys = self.params["latent_factors_keys"]
            self.latent_factors = self.params["latent_factors"]

        # Load semantics models.
        if self.use_semantics:
            self.semantic_emissions = nn.ModuleDict()
            self.semantic_index = OrderedDict()

        if "latent_factors_file" in self.params and self.params["latent_factors_file"] is not None:
            with open(self.params["latent_factors_file"]) as fp:
                self.semantic_definitions = json.load(fp)
            intermediate_dim2 = intermediate_dim + len(params["latent_factors_keys"])
            for i, defn in enumerate(self.semantic_definitions):
                if i >= self.params["max_semantic_targets"]:
                    break
                annotation_q = defn["annotator_question"]
                if self.use_semantics:
                    self.semantic_emissions[annotation_q] = nn.Sequential(
                        nn.Linear(intermediate_dim2, intermediate_dim2),
                        nn.ReLU(),
                        nn.Dropout(p=self.dropout_ratio),
                        nn.Linear(intermediate_dim2, intermediate_dim2),
                        nn.ReLU(),
                        nn.Dropout(p=self.dropout_ratio),
                        nn.Linear(intermediate_dim2, 1),
                    )
                    self.semantic_index[defn["id"]] = annotation_q

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        if self.use_latent_factors and self.latent_factors is not None:
            self.latent_factors = self.latent_factors.to(device)

        if self.use_semantics:
            for s_id in self.semantic_index:
                self.semantic_emissions[self.semantic_index[s_id]] = self.semantic_emissions[
                    self.semantic_index[s_id]
                ].to(device)

        return self

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_discriminator_encoder(self):
        return self.discriminator_encoder

    def get_discriminator_future_encoder(self):
        return self.discriminator_future_encoder

    def get_discriminator_head(self):
        return self.discriminator_head

    def generate_trajectory_sample(
        self,
        trajectory_data,
        dropped_trajectory_data,
        normalized_trajectory_data,
        dropped_normalized_trajectory_data,
        additional_inputs,
        agent_additional_inputs,
        transforms,
        agent_type,
        is_valid,
        scene_inputs_tensor,
        agent_inputs_tensor,
        relevant_agents,
        additional_params=None,
    ):
        """Generate trajectories.

        Parameters
        ----------
        :param trajectory_data: Original trajectory data.
        :param dropped_trajectory_data: Dropped original trajectory data.
        :param normalized_trajectory_data: Normalized trajectory data.
        :param dropped_normalized_trajectory_data: Dropped normalized trajectory data.
        :param additional_inputs: dict of additional inputs. Contains "sample_index" which is an index or list of sample indices to generate.
        :param agent_additional_inputs: dict of additional agent inputs.
        :param transforms: transforms for normalization.
        :param agent_type: type of agent to predict.
        :param is_valid: whether each timestep is valid.
        :param scene_inputs_tensor: tensor including scene info.
        :param agent_inputs_tensor: tensor including nearby agent info.
        :param relevant_agents: tensor indicating whether an agent is relevant.
        :param additional_params: additional params populated through out the encoder/decoder/latent factors,
                for customized operations.

        Returns
        -------
        results_list: list
          A num_samples-long list of result tensors (each tensor is of size [B x num_agents x T x 2 x 1])
        decoding_list: list
          A num_samples-long list of decoding tensors (each tensor is of size [B x T x feature_size x 1])
        stats: dict
          A list of dictionary of output statistics for each sample.

        """
        if additional_params is None:
            additional_params = {}

        if self.params["scene_inputs_detach"]:
            if scene_inputs_tensor is not None:
                scene_inputs_tensor = scene_inputs_tensor.detach()
        if self.params["agent_inputs_detach"]:
            if agent_inputs_tensor is not None:
                agent_inputs_tensor = agent_inputs_tensor.detach()

        # Run an encoder to obtain encoded state from observations.
        encoder_inputs = {
            "trajectories": dropped_trajectory_data,
            "normalized_trajectories": dropped_normalized_trajectory_data,
            "agent_type": agent_type,
            "is_valid": is_valid,
            "scene_data": scene_inputs_tensor,
            "agent_data": agent_inputs_tensor,
            "relevant_agents": relevant_agents,
            "initial_state": None,
            "additional_params": additional_params,
        }
        # Only add map to the encoder inputs when using global map.
        if "encoded_map" in additional_inputs:
            encoder_inputs["map_data"] = additional_inputs["encoded_map"]

        # intermediate_result: [batch_size, num_agents, num_past_steps, encoder_dim].
        # node_states: num_past_steps x num_agents list of lstm tuples ([batch_size, lstm_dim], [batch_size, lstm_dim])
        intermediate_result, node_states, auxiliary_encoder_outputs = self.encoder(encoder_inputs)

        # Read the last time step of node states if graph encoder is used.
        if self.params["encoder_decoder_type"] in ["gnn", "transformer"]:
            node_states = node_states[-1]
        # Run a decoder to obtain predictions.
        decoder_inputs = {
            "intermediate_encoder_result": intermediate_result,
            "state_tuple_enc_last": node_states,
            "last_position": normalized_trajectory_data[:, :, -1, :],
            "latent_factors": self.latent_factors,
            "agent_type": agent_type,
            "agent_additional_inputs": agent_additional_inputs,
            "additional_params": additional_params,
            # 'latent_factor_input':additional_params['latent_factor_input']
        }
        decoder_inputs.update(additional_inputs)
        decoder_inputs.update(auxiliary_encoder_outputs)
        # Verify noise samples and index exist.
        assert "noise_samples" in decoder_inputs, "noise samples not available"
        assert "sample_index" in decoder_inputs, "sample index not available"
        sample_indices = additional_inputs["sample_index"]

        if not (type(sample_indices) == list):
            sample_indices = [sample_indices]
        results_list = []
        decoding_list = []
        stats_list = []

        # Add additional callbacks to augment sample index, if available.
        for cb in self.additional_model_callbacks:
            cb.augment_sample_index(self.params, sample_indices)

        decoder_inputs["sample_indices"] = sample_indices

        # res_traj: [batch, agents, samples, future_timestamps, xy]
        # joined_decoding: [batch, samples, future_timestamps, intermediate_dim]
        res_traj, joined_decoding, stats_l = self.decoder(decoder_inputs)

        stats_list.extend(stats_l)

        # The decoder used to loop over sample_indices, this code is to keep the res_traj consistent with old code.
        # Should be removed when possible. Note, the outer function keep the [sample] dim as the last dim.
        for i, sample_index in enumerate(sample_indices):
            # List item shape: [B x num_agents x T x 2 x 1]
            results_list.append(res_traj[:, :, i].unsqueeze(-1))
            if joined_decoding is not None:
                # List item shape: [B x T x features x 1]
                decoding_list.append(joined_decoding[:, i].unsqueeze(-1))

        # Add additional callbacks to update stats list, if available.
        for cb in self.additional_model_callbacks:
            cb.update_stats_list(self.params, stats_list)

        return results_list, decoding_list, stats_list

    def discriminate_trajectory(
        self,
        dropped_past_trajectory_data,
        normalized_past_trajectory,
        agent_type,
        is_past_valid,
        scene_data,
        agent_data,
        relevant_agents,
        future_trajectory,
        normalized_future_trajectory,
        is_future_valid,
    ):
        discriminator_encoder_inputs = {
            "trajectories": dropped_past_trajectory_data,
            "normalized_trajectories": normalized_past_trajectory,
            "agent_type": agent_type,
            "is_valid": is_past_valid,
            "scene_data": scene_data,
            "agent_data": agent_data,
            "relevant_agents": relevant_agents,
            "initial_state": None,
        }

        intermediate_result, node_states, auxiliary_encoder_outputs = self.discriminator_encoder(
            discriminator_encoder_inputs
        )

        discriminator_future_encoder_inputs = {
            "intermediate_encoder_result": intermediate_result,
            "trajectories": torch.cat([future_trajectory, is_future_valid.unsqueeze(3).float()], 3),
            "normalized_trajectories": torch.cat(
                [normalized_future_trajectory, is_future_valid.unsqueeze(3).float()], 3
            ),
            "agent_type": agent_type,
            "is_valid": is_future_valid,
            "scene_data": None,
            "agent_data": None,
            "relevant_agents": None,
            # batch_size x N_agents x N_time_points x hidden_state
            "initial_state": node_states,
        }
        # discriminator_encoder_inputs.update(intermediate_result)
        discriminator_encoder_inputs.update(auxiliary_encoder_outputs)

        intermediate_result2, node_states2, auxiliary_encoder_outputs2 = self.discriminator_future_encoder(
            discriminator_future_encoder_inputs
        )

        # Return the result of the discriminator head applied to the sum across all agents of the
        # last timepoint's discriminator encoder. Should be between 0 (fake) and 1 (real).
        if self.params["linear_discriminator"]:
            result = torch.mean(self.discriminator_head(intermediate_result2[:, :, -1, :].sum(dim=1)), dim=1).clamp(
                -1, 1
            )
        elif "learn_reward_model" in self.params.keys():
            combined_result = torch.cat((intermediate_result, intermediate_result2), axis=2)
            result = self.discriminator_head(combined_result[:, :, -1, :].sum(dim=1))
            result = result.clamp(min=-100, max=100)
        else:
            result = torch.nn.Sigmoid()(
                torch.mean(self.discriminator_head(intermediate_result2[:, :, -1, :].sum(dim=1)), dim=1)
            )

        if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
            import IPython

            IPython.embed(header="nan/inf in discriminate_trajectory")
        return result

    def compute_extra_cost(
        self,
        predicted,
        expected,
        is_future_valid,
        weighted_validity,
        semantic_labels,
        future_encoding,
        timestamps,
        prediction_timestamp,
        label_weights,
        sqr_errs,
        stats,
    ):
        additional_stats = {}
        extra_context = {
            "timestamps": timestamps,
            "prediction_timestamp": prediction_timestamp,
            "stats": stats,
            "sqr_errs": sqr_errs,
            "is_future_valid": is_future_valid,
        }
        # Compute semantics cost.
        for cb in self.additional_structure_callbacks:
            cb.update_costs(
                additional_stats=additional_stats,
                params=self.params,
                predicted=predicted,
                expected=expected,
                future_encoding=future_encoding,
                extra_context=extra_context,
            )

        if self.use_semantics:
            compute_semantic_costs(
                additional_stats,
                self.params,
                timestamps,
                prediction_timestamp,
                future_encoding,
                self.semantic_emissions,
                semantic_labels,
                self.semantic_index,
                label_weights,
            )

        return additional_stats

    def count_parameters(self):
        models_to_count = {}
        if self.use_latent_factors and self.latent_factors is not None:
            # Add latent factors to model count.
            new_key = "latent_factors_" + str(type(self.latent_factors).__name__)
            models_to_count[new_key] = self.latent_factors

            # Add sub factors to model count -- we allow duplicates from latent factors.
            for key in self.latent_factors_keys:
                if hasattr(self.latent_factors, "factors"):
                    if str(type(self.latent_factors.factors[key]).__name__) == "ExplicitDurationTruncatedFunction":
                        new_key = "latent_factors_ExplicitDuration_" + key
                        models_to_count[new_key] = self.latent_factors.factors[key]
                    elif str(type(self.latent_factors.factors[key]).__name__) == "ConstantTemporalFunction":
                        new_key = "latent_factors_ConstantTemporal_" + key
                        models_to_count[new_key] = self.latent_factors.factors[key]

        if self.use_semantics:
            for key in self.semantic_emissions:
                new_key = "semantic_emissions_" + key
                models_to_count[new_key] = self.semantic_emissions[key]
        return models_to_count

    def get_parameters(self, require_grad=True):
        p_list = []
        if self.use_semantics:
            for p in self.semantic_emissions.parameters():
                if p.requires_grad or not require_grad:
                    p_list.append(p)

        if self.use_latent_factors and self.latent_factors is not None:
            # Add latent factors, including its submodules, to model parameters.
            for p in self.latent_factors.parameters():
                if p.requires_grad or not require_grad:
                    p_list.append(p)

        return p_list

    def get_semantic_keys(self) -> List[str]:
        keys = []
        if self.use_semantics and self.semantic_index is not None:
            for s_id in self.semantic_index:
                keys.append(str(s_id) + "_" + self.semantic_index[s_id])
        return keys
