from collections import OrderedDict

import numpy as np
import torch
from torch import nn

try:
    from model_zoo.intent.language_models import LanguageFactors
except ModuleNotFoundError:
    LanguageFactors = None

from model_zoo.intent.multiagent_decoder import AugmentedMultiAgentDecoder
from model_zoo.intent.prediction_latent_factors import ConstantLatentFactors, ExplicitDurationTruncatedFactors
from model_zoo.intent.prediction_latent_factors_attention import AttentionFactorsBatch

try:
    from model_zoo.intent.prediction_latent_factors_causality import CausalityFactors
except ModuleNotFoundError:
    CausalityFactors = None


class AugmentedMultiAgentDecoderAccelerated(AugmentedMultiAgentDecoder):
    """
    An accelerated version of AugmentedMultiAgentDecoder, collapsed the samples by agents loops.
    Should produce similar result.
    """

    def __init__(self, intermediate_dim, coordinate_dim, params, cumulative_decoding=True, truncated_steps=-1):
        super().__init__(intermediate_dim, coordinate_dim, params, cumulative_decoding, truncated_steps)
        if self.params["state_dropout_ratio"] > 0:
            self.state_dropout = nn.Dropout(self.params["state_dropout_ratio"])

        if self.params["additional_dropout_ratio"] > 0:
            self.additional_factors_dropout = nn.Dropout(self.params["additional_dropout_ratio"])

    def forward(self, inputs):
        """Decode trajectories given results from encoder.
        Note: currently don't support 'maps', 'use-hybrid-outputs', 'latent-factors'
        NOTE: the return tensor put 'sample_size' at 3rd dimension instead of the last (used in outer scope),
        but it's okay since legacy code in prediction_model_codec.py:generate_trajectory_sample will extract samples
        into a list.

        Parameters
        ----------
        inputs : dict
            a dictionary including inputs to decoder

        Returns
        -------
        torch.Tensor
            predicted positions of shape (batch_size, num_agents, sample_size, num_future_timepoints, 2)
        torch.Tensor
            joint decoding of shape (batch_size, sample_size, num_future_timepoints, intermediate_dimension)
        List[dict]
            Additional statistics information.
        """
        additional_params = inputs["additional_params"]
        if self.use_latent_factors:
            latent_factors = inputs["latent_factors"]
            latent_factors.reset_generated_samples()

        if self.use_latent_factors:
            latent_factor_classes = (
                ExplicitDurationTruncatedFactors,
                ConstantLatentFactors,
                AttentionFactorsBatch,
            )
            if CausalityFactors:
                latent_factor_classes = (CausalityFactors, *latent_factor_classes)
            if LanguageFactors:
                # isinstance breaks if one of the types is None
                latent_factor_classes = (LanguageFactors, *latent_factor_classes)

            assert isinstance(
                inputs["latent_factors"],
                latent_factor_classes,
            ), "Only certain latent factors can use accelerated decoder, since some operations are not support"
        # Note, this is not possible with this class since all samples are predicted at the same time
        # if self.use_latent_factors:
        # latent_factors.update_generated_samples(res_traj)

        intermediate_output = inputs["state_tuple_enc_last"]  # [2][agent, batch, hidden_dim]
        last_observed_position = inputs["last_position"]
        latent_factors = inputs["latent_factors"]
        agent_type = inputs["agent_type"]

        if self.params["use_batch_graph_encoder"]:
            batch_size = intermediate_output[0].shape[1]
            intermediate_size = intermediate_output[0].shape[2]
            device = intermediate_output[0].device
        else:
            batch_size = intermediate_output[0][0].shape[0]
            intermediate_size = intermediate_output[0][0].shape[1]
            device = intermediate_output[0][0].device
        decoder_output_dim = self.intermediate_dim
        sample_num = len(inputs["sample_indices"])

        if self.use_latent_factors:
            latent_factors.initialize_latent_data(
                batch_size, device, inputs, self.num_agents, self.num_timepoints, additional_params
            )

        stat = {}
        latent_factors_states = {}
        last_lstm_states = None

        # Continuous predictions
        # [batch, agents, samples, future_timestamps, xy]
        generated = torch.zeros(batch_size, self.num_agents, sample_num, self.num_timepoints, self.coordinate_dim).to(
            device
        )

        # [batch, sample, future_timestamps, intermediate]
        joint_decoding = torch.zeros(batch_size, sample_num, self.num_timepoints, self.intermediate_dim).to(device)
        state_dropout_p = 0.0
        additional_factors_dropout_p = 0.0
        if self.training:
            state_dropout_p = self.params["state_dropout_ratio"]
            additional_factors_dropout_p = self.params["additional_dropout_ratio"]
        elif self.use_latent_factors:
            overwritten_dropouts = latent_factors.overwrite_dropouts()
            if overwritten_dropouts is not None:
                state_dropout_p, additional_factors_dropout_p = overwritten_dropouts
                self.state_dropout.p = state_dropout_p
                self.additional_factors_dropout.p = additional_factors_dropout_p

        # [batch, sample, intermediate]
        if self.params["zero_generator_noise"]:
            joint_noise = torch.zeros([batch_size, self.num_agents, sample_num, intermediate_size], device=device)
        else:
            joint_noise = torch.normal(
                0, 1, [batch_size, self.num_agents, sample_num, intermediate_size], device=device
            )

        if self.params["use_batch_graph_encoder"]:
            # [batch, agents, samples, intermediate_dim]
            s = intermediate_output[0].shape
            intermediate_output_0 = (
                intermediate_output[0].permute(1, 0, 2).unsqueeze(2).expand(s[1], s[0], sample_num, s[2])
            )
            intermediate_output_1 = (
                intermediate_output[1].permute(1, 0, 2).unsqueeze(2).expand(s[1], s[0], sample_num, s[2])
            )
        else:
            intermediate_output_0 = (
                torch.stack([inter[0] for inter in intermediate_output], 1).unsqueeze(2).repeat([1, 1, sample_num, 1])
            )
            intermediate_output_1 = (
                torch.stack([inter[1] for inter in intermediate_output], 1).unsqueeze(2).repeat([1, 1, sample_num, 1])
            )

        # Set up initial lstm states
        # Add noise to encoder state.
        intermediate_output_with_noise = [intermediate_output_0, intermediate_output_1]

        intermediate_output_with_noise[0] = self.state_adapter(torch.cat([intermediate_output_0, joint_noise], -1))
        last_lstm_states = [
            torch.nn.functional.pad(x, (0, self.intermediate_dim - self.intermediate_dim0, 0, 0)).unsqueeze(0)
            for x in intermediate_output_with_noise
        ]

        # Setup finished
        # LSTM starts
        for time_idx in range(self.num_timepoints):
            tensor_t = intermediate_output[0][0][0].new_tensor([time_idx * self.timestep])
            # TODO(guy.rosman): add usage of latent factor as a latent_factors.time_sample(timepoint).
            # [batch, agents, samples, intermediate_dim]
            if time_idx == 0:
                # Read current position from last observed position at t=0.
                # [num_batch, agents, samples, 2]
                current_position = last_observed_position[:, :, :2].unsqueeze(2).repeat([1, 1, sample_num, 1])
            else:
                # Read current position from saved position predicted from previous step.
                current_position = last_positions

            # Obtain latent factors states.
            if self.use_latent_factors:
                # Generate initial latent factors before decoding
                # TODO(guy.rosman): merge w/ reset_generated_samples()
                if time_idx == 0:
                    # Use the stats to pass out latent_factor_output, additional_params is accessible at TrainingStrategy level.
                    stats = {}
                    latent_factors_states = latent_factors.generate_initial_factors(intermediate_output_0, stats)
                    additional_params["latent_factor_output"] = stats.get("latent_factor_output", None)

                else:
                    latent_factors.set_initial_factors(latent_factors_states)

            time_point = time_idx * self.timestep
            tensor_t = intermediate_output[0][0][0].new_tensor([time_idx * self.timestep])
            if self.params["zero_generator_noise"]:
                augmented_intermediate_output = torch.zeros(intermediate_output_0.shape, device=device)
            else:
                augmented_intermediate_output = torch.normal(0, 1, intermediate_output_0.shape, device=device)

            # Augment agent state with additional info.
            augmented_state = current_position
            if self.use_latent_factors:
                augmented_state_latent = latent_factors.augment_state(current_position, time_idx, None)

                if augmented_state_latent is not None:
                    augmented_state = augmented_state_latent

            # Update augmented list with a coordinate encoder.
            if self.coordinate_encoder is None:
                current_position_embedding = augmented_state
            else:
                current_position_embedding = self.coordinate_encoder(augmented_state)
            augmented_list = [current_position_embedding]

            # New
            # Update lstm state tuple.
            lstm_state = last_lstm_states
            if time_idx == 0:
                # Overwrite lstm state if necessary.
                if self.use_latent_factors:
                    # This function is currently a no-op, returns none.  It will need to be modified to use the new
                    # batched shape.
                    overwritten_initial_lstm_state = latent_factors.overwrite_initial_lstm_state(
                        intermediate_output, None, tensor_t
                    )
                    if overwritten_initial_lstm_state is not None:
                        lstm_state = overwritten_initial_lstm_state

            # Update augmented list with latent factors.
            if latent_factors is not None:
                assert self.latent_factors_keys is not None
                sampled_values = OrderedDict()
                if self.params["latent_factors_drop"]:
                    for key in self.latent_factors_keys:
                        # TODO: make more efficient
                        sampled_values[key] = latent_factors.get_dimensionality(key)
                    augmented_list.append(
                        augmented_intermediate_output.new_zeros(batch_size, np.sum(list(sampled_values.values())))
                    )
                else:
                    for key in self.latent_factors_keys:
                        sampled_values[key] = latent_factors.time_sample(
                            lstm_state, time_idx, tensor_t, stat, None, last_lstm_states, key
                        )
                        if self.params["latent_factors_detach"]:
                            sampled_values[key] = sampled_values[key].detach()
                    augmented_list.extend(list(sampled_values.values()))

            # Update the augmented intermediate output.
            # [batch, agent, sample, intermediate+latent]
            augmented_intermediate_output = torch.cat(augmented_list, -1)

            # Perform dropout to states, to encourage redundancy.
            if state_dropout_p > 0:
                lstm_state = self.state_dropout(lstm_state)

            # Perform dropout to additional inputs.
            if additional_factors_dropout_p > 0:
                # [batch, agents, sample, intermediate]
                input_vec = self.additional_factors_dropout(augmented_intermediate_output)
            else:
                input_vec = augmented_intermediate_output

            # Process input vec through another MLP if necessary.
            if self.augmented_input_encoder:
                # encoded_input: [1, batch, agents, sample, intermediate]
                encoded_input = self.augmented_input_encoder(input_vec)
            else:
                encoded_input = input_vec
            encoded_input = encoded_input.unsqueeze(0)

            # collapse encoded_input and state
            # from [1, batch, agents, sample, intermediate] -> [1, batch*agents*sample, intermediate]
            encoded_input = encoded_input.view(1, -1, intermediate_size)
            lstm_state = [s.view(1, -1, s.shape[-1]) for s in lstm_state]
            # out: [1, batch*agents*sample, 64]
            out, lstm_state = self.temporal_model(encoded_input, lstm_state, agent_type)
            out = out.squeeze(0).view(batch_size, self.num_agents, sample_num, decoder_output_dim)
            lstm_state = [s.view(1, batch_size, self.num_agents, sample_num, -1) for s in lstm_state]
            joint_decoding[:, :, time_idx, :] += out.sum(1)

            # Predict continuous trajectory, and other state with non default functions if necessary.
            predicted_agent_state = None
            if self.use_latent_factors:
                inputs["training"] = self.training
                predicted_agent_state = latent_factors.predict_agent_state(
                    inputs, out, lstm_state, None, time_idx, time_point, batch_size, self.coordinate_decoder
                )
            if predicted_agent_state is None:
                predicted_agent_state = self.coordinate_decoder(out)
            generated[:, :, :, time_idx, :] = predicted_agent_state
            last_positions = predicted_agent_state
            last_lstm_states = lstm_state

        # End of agent loop

        if self.cumulative_decoding:
            result = last_positions[:, :, :2].unsqueeze(2).repeat([1, 1, sample_num, 1]) + generated.cumsum(dim=2)
        else:
            result = generated

        # TODO(guy.rosman): move to update_stats.
        if latent_factors is not None:
            cumulative_durations_dict = {}
            if hasattr(latent_factors, "factor_names"):
                for key in latent_factors.factor_names:
                    cumulative_durations = []
                    if str(type(latent_factors.factors[key]).__name__) == "ExplicitDurationTruncatedFunction":
                        for i in range(self.num_agents):
                            if self.params["use_batch_graph_encoder"]:
                                # If we are using the batch encoder, the output format is different.
                                # [2][agents, batch, features] instead of [agents][2][1, batch, features].
                                encoder_output = intermediate_output[0][i, :, :]
                            else:
                                encoder_output = intermediate_output[i][0]

                            cumulative_durations_, _, _, _ = latent_factors.factors[key].get_segment_stats(
                                encoder_output
                            )
                            cumulative_durations.append(cumulative_durations_)
                    if cumulative_durations:
                        cumulative_durations_dict[key] = {"mean": torch.cat(cumulative_durations, 1)}
        if self.truncated_steps > 0:
            result[:, :, :, -self.truncated_steps, :] = result[:, :, :, -self.truncated_steps, :].detach()

        stats_list = []
        # Stats are per sample
        for i in range(sample_num):
            stats = {}
            x = generated[:, :, i, :, 0]
            y = generated[:, :, i, :, 1]
            stats["decoder/decoder_x_mean"] = x.mean()
            stats["decoder/decoder_y_mean"] = y.mean()
            stats["decoder/decoder_x_std"] = x.std()
            stats["decoder/decoder_y_std"] = y.std()
            stats["decoder/decoder_x0"] = x[..., 0]
            stats["decoder/decoder_y0"] = y[..., 0]
            if latent_factors is not None:
                stats["cumulative_durations"] = cumulative_durations_dict
            stats_list.append(stats)
            if self.use_latent_factors:
                latent_factors.update_stats(stats, i)

        return result, joint_decoding, stats_list
