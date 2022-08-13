from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from model_zoo.intent.create_networks import create_mlp
from model_zoo.intent.enc_dec_temporal_models import AgentTemporalModel, TypeConditionedTemporalModel
from triceps.protobuf import protobuf_training_parameter_names


class AugmentedMultiAgentDecoder(nn.Module):
    """
    A neural network to decode a tensor into trajectories.
    """

    def __init__(self, intermediate_dim, coordinate_dim, params, cumulative_decoding=True, truncated_steps=-1):
        """[summary]

        Args:
            intermediate_dim ([type]): [description]
            coordinate_dim ([type]): [description]
            cumulative_decoding (bool, optional): [description]. Defaults to True.
            truncated_steps (int, optional): [description]. Defaults to -1.
            params (dict, optional): [description].
        """
        super().__init__()
        self.dropout_ratio = params["dropout_ratio"]
        self.num_agents = None
        self.num_timepoints = None
        self.latent_factors_keys = params["latent_factors_keys"]
        self.intermediate_dim0 = intermediate_dim
        self.latent_factor_output_dim = params["latent_factors_output_dim"]
        if params["use_latent_factors"]:
            self.latent_factor_keys_size = len(self.latent_factors_keys)
            self.intermediate_dim = intermediate_dim + self.latent_factor_keys_size * self.latent_factor_output_dim
        else:
            self.intermediate_dim = intermediate_dim
            self.latent_factor_keys_size = 0
        self.agent_types = params["agent_types"]
        # Define state adapter.
        self.state_adapter = create_mlp(
            input_dim=self.intermediate_dim0 * 2,
            layers_dim=[self.intermediate_dim0],
            dropout_ratio=0.0,
            batch_norm=False,
        )

        self.params = params
        self.truncated_steps = truncated_steps
        self.coordinate_encoder_dim = params["predictor_hidden_state_dim"]
        self.coordinate_dim = coordinate_dim
        self.state_dim = coordinate_dim

        # Use latent factors if it is added.
        self.use_latent_factors = "use_latent_factors" in params and params["use_latent_factors"]

        if self.use_latent_factors:
            # By default, latent_factors should come from inputs from prediction_model_codec.
            # Since we do not have access to inputs, we obtain latent factors from param to access its attributes.
            latent_factors = params["latent_factors"]
            self.state_dim += latent_factors.get_augmented_state_dim()

        # Define coordinate encoder.
        if self.params["decoder_embed_position"]:
            if self.params["coordinate_encoder_widths"] is None:
                linear_net = nn.Linear(self.state_dim, self.coordinate_encoder_dim)
                self.coordinate_encoder = nn.Sequential(linear_net, nn.ReLU())
            else:
                self.coordinate_encoder = create_mlp(
                    input_dim=self.state_dim,
                    layers_dim=self.params["coordinate_encoder_widths"] + [self.coordinate_encoder_dim],
                    dropout_ratio=self.dropout_ratio,
                    leaky_relu=False,
                    pre_bn=True,
                    batch_norm=False,
                )
        else:
            self.coordinate_encoder_dim = self.state_dim
            self.coordinate_encoder = None

        # Add map embedding info.
        if (
            (not self.params["disable_map_decoder"])
            and (not self.params["disable_map_input"])
            and self.params["map_input_type"] == "point"
            and params["map_encoder_type"] == "gnn"
        ):
            self.use_map_embeddings = True
            self.map_embedding_dim = self.params["map_layer_features"][-1]
        else:
            self.use_map_embeddings = False
            self.map_embedding_dim = 0

        # Define input encoder.
        if self.params["decoder_embed_input"]:
            self.augmented_input_encoder = create_mlp(
                input_dim=self.coordinate_encoder_dim
                + self.latent_factor_keys_size * self.latent_factor_output_dim
                + self.map_embedding_dim,
                layers_dim=[self.coordinate_encoder_dim],
                dropout_ratio=self.dropout_ratio,
            )
            temporal_model_input_size = self.coordinate_encoder_dim
        else:
            self.augmented_input_encoder = None
            temporal_model_input_size = (
                self.coordinate_encoder_dim
                + self.latent_factor_keys_size * self.latent_factor_output_dim
                + self.map_embedding_dim
            )

        self.type_dim = len(self.agent_types)
        # This assumes that self.coordinate_input_encoder is not None
        if params["type_conditioned_temporal_model"]:
            self.temporal_model = TypeConditionedTemporalModel(
                input_size=temporal_model_input_size, hidden_size=self.intermediate_dim, num_agent_types=self.type_dim
            )
        else:
            self.temporal_model = AgentTemporalModel(
                input_size=temporal_model_input_size, hidden_size=self.intermediate_dim, num_agent_types=self.type_dim
            )

        # Edge module for inputting node states to edge states.
        coordinate_input_dim = self.intermediate_dim
        if self.use_latent_factors:
            coordinate_input_dim += latent_factors.get_augmented_state_dim()

        if self.params["coordinate_decoder_widths"] is None:
            self.coordinate_decoder = nn.Sequential(nn.Linear(coordinate_input_dim, coordinate_dim))
        else:
            self.coordinate_decoder = create_mlp(
                input_dim=coordinate_input_dim,
                layers_dim=self.params["coordinate_decoder_widths"] + [coordinate_dim],
                dropout_ratio=self.dropout_ratio,
            )

        self.cumulative_decoding = cumulative_decoding
        self.timestep = self.params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEP_SIZE]

    def forward(self, inputs):
        """Decode trajectories given results from encoder.

        :param inputs: for input, see self.decode()
        :return:
            torch.Tensor
                predicted positions of shape (batch_size, num_agents, sample_size, num_future_timepoints, 2)
            torch.Tensor
                joint decoding of shape (batch_size, sample_size, num_future_timepoints, intermediate_dimension)
            List[dict]
                Additional statistics information.
        """
        sample_indices = inputs["sample_indices"]
        results_list = []
        decoding_list = []
        stats_list = []

        # Save generated samples for adaptive sampling.
        if self.use_latent_factors:
            latent_factors = inputs["latent_factors"]
            latent_factors.reset_generated_samples()

        for sample_index in sample_indices:
            inputs["sample_index"] = sample_index
            res_traj, joined_decoding, stats = self.decode(inputs)
            # List item shape: [B x num_agents x T x 2]
            results_list.append(res_traj)
            if joined_decoding is not None:
                # List item shape: [B x T x features x 1]
                decoding_list.append(joined_decoding)
            stats_list.append(stats)

            # Update samples list within latent factor interface given a new sample.
            if self.use_latent_factors:
                latent_factors.update_generated_samples(res_traj)

        # [batch, agents, sample, future_timestamps, xy]
        result = torch.stack(results_list, 2)
        # [batch, sample, future_timestamps, intermediate]
        joined_decoding = torch.stack(decoding_list, 1)
        return result, joined_decoding, stats_list

    def decode(self, inputs):
        """Decode trajectories given results from encoder for one sample.
        NOTE: the return tensor put 'sample_size' at 3rd dimension instead of the last (used in outter scope), but it's
        okay since legacy code in prediction_model_codec.py:generate_trajectory_sample will extract samples into a list.

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
        intermediate_output = inputs["state_tuple_enc_last"]
        last_observed_position = inputs["last_position"]
        latent_factors = inputs["latent_factors"]
        agent_type = inputs["agent_type"]
        agent_additional_inputs = inputs["agent_additional_inputs"]

        batch_size = intermediate_output[0][0].shape[0]
        device = intermediate_output[0][0].device

        # Set up continuous predictions.
        generated = torch.zeros(batch_size, self.num_agents, self.num_timepoints, self.coordinate_dim).to(device)

        if self.use_latent_factors:
            latent_factors.initialize_latent_data(batch_size, device, inputs, self.num_agents, self.num_timepoints, {})

        stats = {}
        joint_decoding = torch.zeros(batch_size, self.num_timepoints, self.intermediate_dim).to(device)

        # Set up dropouts.
        state_dropout_p = 0.0
        additional_factors_dropout_p = 0.0
        if self.training:
            state_dropout_p = self.params["state_dropout_ratio"]
            additional_factors_dropout_p = self.params["additional_dropout_ratio"]
        elif self.use_latent_factors:
            overwritten_dropouts = latent_factors.overwrite_dropouts()
            if overwritten_dropouts is not None:
                state_dropout_p, additional_factors_dropout_p = overwritten_dropouts

        # Set up joint noise for all agents.
        joint_noise = torch.normal(0, 1, intermediate_output[0][0].shape, device=device)

        # Set up dictionaries to save relevant information for each agent.
        last_lstm_states = {}
        last_positions = {}
        latent_factors_states = {}

        # Set up initial lstm state
        # Add noise to encoder state.
        for agent_i in range(self.num_agents):
            intermediate_output_with_noise = list(intermediate_output[agent_i])
            intermediate_output_with_noise[0] = self.state_adapter(
                torch.cat([intermediate_output_with_noise[0], joint_noise], 1)
            )
            lstm_state = [
                torch.nn.functional.pad(x, (0, self.intermediate_dim - self.intermediate_dim0, 0, 0)).unsqueeze(0)
                for x in intermediate_output_with_noise
            ]
            last_lstm_states[agent_i] = lstm_state

        # Predict for each future time step.
        for time_idx in range(self.num_timepoints):

            # Predict trajectory sequence for each agent.
            for agent_i in range(self.num_agents):
                if time_idx == 0:
                    # Read current position from last observed position at t=0.
                    # [num_batch, num_pos_dim].
                    current_position = last_observed_position[:, agent_i, :2]
                else:
                    # Read current position from saved position predicted from previous step.
                    current_position = last_positions[agent_i]

                # Obtain latent factors states.
                if self.use_latent_factors:
                    # Generate initial latent factors before decoding
                    if time_idx == 0:
                        latent_factors_states[agent_i] = latent_factors.generate_initial_factors(
                            intermediate_output[agent_i][0], stats
                        )
                    else:
                        latent_factors.set_initial_factors(latent_factors_states[agent_i])

                time_point = time_idx * self.timestep
                tensor_t = intermediate_output[0][0][0].new_tensor([time_idx * self.timestep])
                augmented_intermediate_output = torch.normal(0, 1, intermediate_output[agent_i][0].shape, device=device)

                # Augment agent state with additional info.
                augmented_state = current_position
                if self.use_latent_factors:
                    augmented_state_latent = latent_factors.augment_state(current_position, time_idx, agent_i)
                    if augmented_state_latent is not None:
                        augmented_state = augmented_state_latent

                # Update augmented list with a coordinate encoder.
                if self.coordinate_encoder is None:
                    current_position_embedding = augmented_state
                else:
                    current_position_embedding = self.coordinate_encoder(augmented_state)
                augmented_list = [current_position_embedding]

                # Update augmented list with map embeddings.
                if self.use_map_embeddings:
                    assert agent_additional_inputs is not None, "agent_additional_inputs is None"
                    assert "get_traj_map_embeddings" in agent_additional_inputs, "get_traj_map_embeddings do not exist"
                    # Obtain map embedding. Need to artificially create a trajectory with step 1 in the second dim.
                    # [num_batch, map_embed_size]
                    current_position_map_embedding = agent_additional_inputs["get_traj_map_embeddings"](
                        current_position.unsqueeze(1).detach(), agent_i
                    )
                    current_position_map_embedding = current_position_map_embedding[:, 0]
                    augmented_list.append(current_position_map_embedding)

                # Update lstm state tuple.
                lstm_state = last_lstm_states[agent_i]
                if time_idx == 0:
                    # Overwrite lstm state if necessary.
                    if self.use_latent_factors:
                        overwritten_initial_lstm_state = latent_factors.overwrite_initial_lstm_state(
                            intermediate_output, agent_i, int(tensor_t)
                        )
                        if overwritten_initial_lstm_state is not None:
                            lstm_state = overwritten_initial_lstm_state

                # Update augmented list with latent factors.
                if latent_factors is not None:
                    assert self.latent_factors_keys is not None
                    sampled_values = OrderedDict()
                    if self.params["latent_factors_drop"]:
                        for key in self.latent_factors_keys:
                            sampled_values[key] = latent_factors.get_dimensionality(key)
                        augmented_list.append(
                            augmented_intermediate_output.new_zeros(batch_size, np.sum(list(sampled_values.values())))
                        )
                    else:
                        for key in self.latent_factors_keys:
                            sampled_values[key] = latent_factors.time_sample(
                                lstm_state, time_idx, tensor_t, stats, agent_i, last_lstm_states, key
                            )
                            if self.params["latent_factors_detach"]:
                                sampled_values[key] = sampled_values[key].detach()
                        augmented_list.extend(list(sampled_values.values()))

                # Update the augmented intermediate output.
                augmented_intermediate_output = torch.cat(augmented_list, 1)
                lstm_state = list(lstm_state)

                # Perform dropout to states, to encourage redundancy.
                if state_dropout_p > 0:
                    dropout_tensor = lstm_state[0].clone().uniform_() > state_dropout_p
                    lstm_state = [x * dropout_tensor for x in lstm_state]

                # Perform dropout to additional inputs.
                if additional_factors_dropout_p > 0:
                    dropout_tensor = augmented_intermediate_output.clone().uniform_() > additional_factors_dropout_p
                    input_vec = augmented_intermediate_output * dropout_tensor
                else:
                    input_vec = augmented_intermediate_output

                # Process input vec through another MLP if necessary.
                if self.augmented_input_encoder:
                    encoded_input = self.augmented_input_encoder(input_vec.unsqueeze(0))

                else:
                    encoded_input = input_vec.unsqueeze(0)

                # Call lstm module to get new lstm state.
                assert encoded_input.shape[2] == self.coordinate_encoder_dim
                out, lstm_state = self.temporal_model(encoded_input, lstm_state, agent_type[:, agent_i, :])
                joint_decoding[:, time_idx, :] += out.squeeze(0)

                # Predict continuous trajectory, and other state with non default functions if necessary.
                predicted_agent_state = None
                if self.use_latent_factors:
                    inputs["training"] = self.training
                    predicted_agent_state = latent_factors.predict_agent_state(
                        inputs, out, lstm_state, agent_i, time_idx, time_point, batch_size, self.coordinate_decoder
                    )
                if predicted_agent_state is None:
                    predicted_agent_state = self.coordinate_decoder(out.squeeze(0)).squeeze()
                generated[:, agent_i, time_idx] = predicted_agent_state

                # Update LSTM state for next prediction.
                last_lstm_states[agent_i] = lstm_state

                # Update position for next prediction.
                current_position = generated[:, agent_i, time_idx, :].clone()
                last_positions[agent_i] = current_position

        # Predict cumulative offsets.
        if self.cumulative_decoding:
            result = last_observed_position[:, :, :2].unsqueeze(2) + generated.cumsum(dim=2)
        else:
            result = generated

        if latent_factors is not None:
            stats["cumulative_durations"] = {}
            if hasattr(latent_factors, "factor_names"):
                for key in latent_factors.factor_names:
                    cumulative_durations = []
                    for agent_i in range(self.num_agents):
                        if str(type(latent_factors.factors[key]).__name__) == "ExplicitDurationTruncatedFunction":
                            cumulative_durations_, *_ = latent_factors.factors[key].get_segment_stats(
                                intermediate_output[agent_i][0]
                            )
                            cumulative_durations.append(cumulative_durations_)
                            stats["cumulative_durations"][key] = {"mean": torch.cat(cumulative_durations, 1)}

        if self.truncated_steps > 0:
            result[:, :, -self.truncated_steps] = result[:, :, -self.truncated_steps].detach()

        # Update stats for debugging and additional loss computation.
        stats["decoder/decoder_x_mean"] = generated[..., 0].mean()
        stats["decoder/decoder_y_mean"] = generated[..., 1].mean()
        stats["decoder/decoder_x_std"] = generated[..., 0].std()
        stats["decoder/decoder_y_std"] = generated[..., 1].std()
        stats["decoder/decoder_x0"] = generated[..., 0, 0]
        stats["decoder/decoder_y0"] = generated[..., 0, 1]

        if self.use_latent_factors:
            latent_factors.update_stats(stats, None)

        return result, joint_decoding, stats

    def set_num_agents(self, num_agents, num_timepoints):
        self.num_agents = num_agents
        self.num_timepoints = num_timepoints

    def save_model(self, data, is_valid):
        pass
