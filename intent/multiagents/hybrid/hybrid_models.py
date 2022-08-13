from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model_zoo.intent.prediction_latent_factors import LatentFactors
from triceps.protobuf import protobuf_training_parameter_names
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset


class HybridFactors(LatentFactors):
    def __init__(self, params: dict):
        """HybridFactors class that encapsulate all hybrid-based operations in the decoder.

        Parameters
        ----------
        params : dict
            The parameters for hybrid-based operations.
        """

        super().__init__()
        self.params = params

        self.discrete_label_type = params["discrete_label_type"]
        discrete_size = {
            # 'LT': 0, 'RT': 1, 'FF': 2, 'SF': 3, 'ST': 4
            ProtobufPredictionDataset.DATASET_KEY_MANEUVERS: self.params["discrete_label_size"],
            # 'LK': 0, 'LL': 1, 'LR': 2
            ProtobufPredictionDataset.DATASET_KEY_LANE_CHANGES: 3,
            ProtobufPredictionDataset.DATASET_KEY_LANE_INDICES: 0,
            "": 0,
        }
        assert self.discrete_label_type in discrete_size, "Bad value for --discrete-label-type."
        self.discrete_label_domain_size = discrete_size[self.discrete_label_type]
        # Continuous (x, y) + discrete.
        self.state_dim = 2 + self.discrete_label_domain_size

        self.latent_factors_keys = params["latent_factors_keys"]
        self.intermediate_dim = params["predictor_hidden_state_dim"] + len(self.latent_factors_keys)

        self.learn_discrete_proposal = self.params["learn_discrete_proposal"]
        self.proposal_adaptive_sampling = self.params["proposal_adaptive_sampling"]
        self.proposal_samples_lstm = self.params["proposal_samples_lstm"]
        self.hybrid_teacher_forcing = self.params["hybrid_teacher_forcing"]

        self.hybrid_fixed_mode = self.params["hybrid_fixed_mode"]
        self.hybrid_fixed_mode_type = self.params["hybrid_fixed_mode_type"]
        self.hybrid_dropout_validation = self.params["hybrid_dropout_validation"]
        self.hybrid_disable_mode = self.params["hybrid_disable_mode"]

        self.hybrid_discrete_dropout = self.params["hybrid_discrete_dropout"]
        self.hybrid_discrete_dropout_type = self.params["hybrid_discrete_dropout_type"]

        self.augmented_state_dim = self.discrete_label_domain_size
        self.generated_samples = []

        # Define models.
        # Predict distribution over discrete modes T.
        self.maneuver_decoder = nn.Sequential(
            nn.Linear(self.intermediate_dim, self.discrete_label_domain_size),
        )
        if self.learn_discrete_proposal:
            # Predict Q adaptively based on previous samples.
            if self.proposal_adaptive_sampling:
                future_steps = self.params[protobuf_training_parameter_names.PARAM_FUTURE_TIMESTEPS]
                if self.proposal_samples_lstm:
                    # Encode previous samples through LSTM.
                    self.generated_samples_encoder = nn.LSTM(self.state_dim, self.intermediate_dim, batch_first=True)
                else:
                    # Encode previous samples through MLP.
                    self.generated_samples_encoder = nn.Sequential(
                        nn.Linear(future_steps * self.state_dim, self.intermediate_dim),
                    )

                self.maneuver_proposal_decoder = nn.Sequential(
                    nn.Linear(self.intermediate_dim * 2, self.discrete_label_domain_size),
                )
            else:
                self.maneuver_proposal_decoder = nn.Sequential(
                    nn.Linear(self.intermediate_dim, self.discrete_label_domain_size),
                )

        # Initialize data.
        self.last_mode = None
        self.generated_samples = []
        self.proposal_logits_total = None
        self.generated_d = None
        self.generated_sample_log_weight = None
        self.gt_future_mode = None
        self.gt_future_positions = None
        self.gt_sample_log_weight = None

    def forward(self):
        return None

    def time_sample(
        self,
        current_agent_state: torch.Tensor,
        time_idx: int,
        t: torch.Tensor,
        agent_i: int,
        all_agent_states: OrderedDict,
        additional_info: Optional[object] = None,
    ) -> torch.Tensor:
        return None

    def get_augmented_state_dim(self):
        """
        Get additional state dim.
        """
        return self.discrete_label_domain_size

    def reset_generated_samples(self):
        """
        Reset generated samples.
        """
        self.generated_samples = []

    def update_generated_samples(self, res_traj: torch.Tensor):
        """
        Update generated samples.

        Parameters
        ----------
        res_traj: torch.Tensor
            Predicted trajectory.
        """
        # Combine continuous and discrete states.
        generated_sample = torch.cat([self.generated_d.detach(), res_traj.detach()], -1)
        self.generated_samples.append(generated_sample)

    def overwrite_dropouts(self):
        """
        Overwrite dropouts.
        """
        if self.hybrid_dropout_validation:
            # Add dropout to support diversity.
            state_dropout_p = self.params["state_dropout_ratio"]
            additional_factors_dropout_p = self.params["additional_dropout_ratio"]
        else:
            state_dropout_p = 0.0
            additional_factors_dropout_p = 0.0
        return state_dropout_p, additional_factors_dropout_p

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
        Initialize latent data before prediction.

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
        # Placeholder for discrete predictions.
        self.generated_d = torch.zeros(batch_size, num_agents, num_timepoints, self.discrete_label_domain_size).to(
            device
        )

        # Keep log probability for each sample.
        self.generated_sample_log_weight = torch.zeros(batch_size, num_agents).to(device)

        # Weight for ground truth discrete sample.
        # This is used to optimize the model.
        self.gt_sample_log_weight = torch.zeros(batch_size, num_agents).to(device)

        # Keep logits of proposal function for regularization.
        self.proposal_logits_total = torch.zeros(
            batch_size, num_agents, num_timepoints, self.discrete_label_domain_size
        ).to(device)

        # Obtain future labels for supervised loss computation.
        self.gt_future_mode = inputs[self.discrete_label_type + "_future"].clone()

        # Perturb discrete labels with a ratio.
        if self.params["perturb_discrete_ratio"] > 0:
            random_gt_future_mode = torch.randint(0, 5, self.gt_future_mode.shape).to(self.gt_future_mode.device)
            perturb_flag = self.gt_future_mode.clone().uniform_() <= self.params["perturb_discrete_ratio"]
            perturb_flag = perturb_flag.int()
            self.gt_future_mode = self.gt_future_mode * (1 - perturb_flag) + random_gt_future_mode * perturb_flag

        # Use fixed future mode for ManeuverLSTM, which can be chosen from the first mode or the most frequent mode.
        if self.hybrid_fixed_mode:
            if self.hybrid_fixed_mode_type == "first":
                self.gt_future_mode = torch.ones_like(self.gt_future_mode) * self.gt_future_mode[:, :1]
            else:
                self.gt_future_mode = torch.ones_like(self.gt_future_mode) * self.gt_future_mode.mode(1)[0].unsqueeze(1)

        # Get past modes, as one-hot vector.
        past_mode = inputs[self.discrete_label_type + "_past"]
        self.gt_future_positions = inputs["future_positions"][..., :2]
        if self.discrete_label_type == ProtobufPredictionDataset.DATASET_KEY_MANEUVERS or (
            self.discrete_label_type == ProtobufPredictionDataset.DATASET_KEY_LANE_CHANGES
        ):
            self.gt_future_mode = torch.nn.functional.one_hot(
                self.gt_future_mode.long(), num_classes=self.discrete_label_domain_size
            ).float()
            past_mode = torch.nn.functional.one_hot(
                past_mode.long(), num_classes=self.discrete_label_domain_size
            ).float()

        # Get the most recent mode.
        self.last_mode = past_mode[:, :, -1]

    def generate_initial_factors(self, initial_state_tensor, stats) -> None:
        """Update initial latent factors.

        Parameters
        ----------
        initial_state_tensor : torch.Tensor
            The intermediate state of the target agents. The dimension is (batch_size, intermediate_dim).
        stats : dict
            The statistics dictionary for additional latent factor statistics.
        """
        pass

    def set_initial_factors(self, initial_state_tensor) -> None:
        pass

    def augment_state(self, current_position: torch.Tensor, t: int, agent_i: int) -> torch.Tensor:
        """
        Augment agent state with additional info.

        Parameters
        ----------
        current_position : torch.Tensor
            Current agent position.

        Returns
        -------
        augmented_state_latent : torch.Tensor
            Updated agent state (i.e. positions + modes).
        """
        # Augment position with discrete mode.
        if t == 0:
            augmented_state_latent = torch.cat((current_position, self.last_mode[:, agent_i]), -1)
        else:
            augmented_state_latent = torch.cat((current_position, self.generated_d[:, agent_i, t - 1, :].clone()), -1)
        return augmented_state_latent

    def overwrite_initial_lstm_state(self, intermediate_output: List[List[torch.Tensor]], i: int, t: int) -> list:
        """
        Overwrite initial lstm state at decoder when t=0.

        Parameters
        ----------
        intermediate_output : torch.Tensor
            Intermediate output from LSTM.
        i : int
            Agent index.
        t : int
            Time index

        Returns
        -------
        lstm_state: torch.Tensor
            Updated state.
        """
        # Skip noise for hybrid prediction.
        lstm_state = [x.unsqueeze(0) for x in intermediate_output[i]]
        return lstm_state

    def predict_agent_state(
        self,
        inputs: dict,
        out: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        agent_i: int,
        t: int,
        timepoint: float,
        batch_size: int,
        coordinate_decoder: torch.nn.modules.container.Sequential,
    ) -> torch.Tensor:
        """
        Predict agent state, including positions and discrete modes, for agent i at time step t.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs.
        out : torch.Tensor
            LSTM output.
        agent_i : int
            Agent index.
        t : int
            Time index.
        batch_size : int
            Batch size.
        coordinate_decoder : torch.nn.modules.container.Sequential
            Coordinate decoder.

        Returns
        -------
        generated_i_t : torch.Tensor
            Generated coordinate/state at time t for agent i.
        """
        # Whether to predict using ground truth future mode.
        predict_use_gt_mode = inputs["sample_index"] < 0

        # Get logits from transition function T.
        discrete_probability_logits = self.maneuver_decoder(out.squeeze(0))
        discrete_probability = nn.Softmax(dim=-1)(discrete_probability_logits)

        # Get logits from proposal function Q.
        if self.learn_discrete_proposal:

            if self.proposal_adaptive_sampling:

                if self.generated_samples:
                    # Encode previously generated samples if they exist.
                    generated_samples_i = [generated_sample[:, agent_i] for generated_sample in self.generated_samples]
                    # [batch_size, num_samples, num_past_steps, state_size)
                    generated_samples_i = torch.stack(generated_samples_i, 1)
                    state_dim = generated_samples_i.shape[-1]

                    if self.proposal_samples_lstm:
                        # Encode previously generated samples with LSTM.
                        generated_samples_i = generated_samples_i.view(
                            batch_size * len(self.generated_samples), -1, state_dim
                        )
                        generated_samples_i_embedding = self.generated_samples_encoder(generated_samples_i)[1][0][0]
                        # [batch_size, num_samples, encode_dim].
                        generated_samples_i_embedding = generated_samples_i_embedding.view(
                            batch_size, len(self.generated_samples), -1
                        )
                    else:
                        # Otherwise, encode previously generated samples with MLP.
                        generated_samples_i = generated_samples_i.view(batch_size, len(self.generated_samples), -1)
                        # [batch_size, num_samples, encode_dim].
                        generated_samples_i_embedding = self.generated_samples_encoder(generated_samples_i)

                    # Maxpool over all generated samples.
                    # [batch_size, encode_dim]
                    generated_samples_i_embedding = generated_samples_i_embedding.max(1)[0]
                else:
                    generated_samples_i_embedding = torch.zeros_like(out.squeeze(0))

                proposal_input = torch.cat([out.squeeze(0), generated_samples_i_embedding], -1)
            else:
                proposal_input = out.squeeze(0)

            proposal_probability_logits = self.maneuver_proposal_decoder(proposal_input)

            # Save values for regularization.
            self.proposal_logits_total[:, agent_i, t] = proposal_probability_logits

            # Add proposal logits by discrete logits.
            proposal_probability_logits = proposal_probability_logits + discrete_probability_logits
        else:
            # Use discrete transition as surrogate if proposal distribution is not predicted.
            self.proposal_logits_total[:, agent_i, t] = discrete_probability_logits
            proposal_probability_logits = discrete_probability_logits

        # Get samples from gumbel softmax, which takes unnormalized logits as input.
        discrete_sample = torch.nn.functional.gumbel_softmax(proposal_probability_logits, hard=True)

        if self.hybrid_disable_mode:
            # Disable discrete mode and set to a fixed number.
            discrete_sample = discrete_sample * 0

        if self.hybrid_discrete_dropout > 0 and inputs["training"]:
            # Add dropout to discrete sample.
            dropout_samples = torch.rand(discrete_sample.shape[:1]) < self.hybrid_discrete_dropout
            dropout_samples = dropout_samples.to(discrete_sample.device)
            if self.hybrid_discrete_dropout_type == "zero":
                # Set dropout discrete sample to 0.
                discrete_sample = discrete_sample * (~dropout_samples).unsqueeze(1)
            elif self.hybrid_discrete_dropout_type == "random":
                # Set dropout samples to a random value.
                new_samples = (
                    torch.nn.functional.one_hot(
                        torch.randint(0, self.discrete_label_domain_size, discrete_sample.shape[:1]),
                        num_classes=self.discrete_label_domain_size,
                    )
                    .float()
                    .to(discrete_sample.device)
                )
                discrete_sample = discrete_sample * (~dropout_samples).unsqueeze(
                    1
                ) + new_samples * dropout_samples.unsqueeze(1)
            else:
                raise NotImplementedError(
                    "Bad --hybrid-discrete-dropout-type {}".format(self.hybrid_discrete_dropout_type)
                )

        # For fixed mode prediction (i.e. ManeuverLSTM), always generate the same set of samples covering each maneuver.
        if self.hybrid_fixed_mode:
            sample_index = max(0, inputs["sample_index"])
            discrete_sample = (
                torch.nn.functional.one_hot(
                    torch.Tensor([sample_index]).long(), num_classes=self.discrete_label_domain_size
                )
                .float()
                .to(discrete_sample.device)
            )
            discrete_sample = discrete_sample.repeat(proposal_probability_logits.shape[0], 1)

        # Save predicted discrete mode as results.
        self.generated_d[:, agent_i, t, :] = discrete_sample

        # Update sample weights as accumulated sum of log probability of the predicted sample.
        self.generated_sample_log_weight[:, agent_i] += torch.log(
            torch.sum(discrete_probability * discrete_sample, -1) + 1e-20
        )

        # Obtain continuous samples given current mode.
        generated_i_t = coordinate_decoder(torch.cat((out.squeeze(0), discrete_sample), -1)).squeeze()

        # Inject current mode with ground truth mode if provided, and compute transition likelihood.
        if predict_use_gt_mode:
            next_gt_discrete = self.gt_future_mode[:, agent_i, t]

            # Compute weight of transitioning to next ground truth mode.
            # For fixed mode, we only compute the transition at t=0.
            if self.hybrid_fixed_mode:
                if t == 0:
                    self.gt_sample_log_weight[:, agent_i] = torch.log(
                        torch.sum(discrete_probability * next_gt_discrete, -1) + 1e-20
                    )
            else:
                self.gt_sample_log_weight[:, agent_i] += torch.log(
                    torch.sum(discrete_probability * next_gt_discrete, -1) + 1e-20
                )

            # Inject current mode with ground truth value.
            self.generated_d[:, agent_i, t, :] = next_gt_discrete.clone()

            # Predict continuous samples given ground truth current mode.
            generated_i_t = coordinate_decoder(torch.cat((out.squeeze(0), next_gt_discrete), -1)).squeeze()

        # Use ground truth future positions if teacher forcing is enabled.
        if predict_use_gt_mode and self.hybrid_teacher_forcing:
            generated_i_t = self.gt_future_positions[:, agent_i, t]

        # Only return predicted positions, and keep predicted modes hidden within the hybrid class.
        return generated_i_t

    def update_stats(self, stats: dict, sample_i: Optional[int]) -> None:
        """
        Update stats from latent predictions.

        Parameters
        ----------
        stats : dict
            Dictionary of statistics.

        Returns
        -------
        stats : dict
            Updated dictionary of statistics.
        """
        stats["discrete_samples"] = self.generated_d
        stats["discrete_samples_log_weight"] = self.generated_sample_log_weight
        stats["gt_sample_log_weight"] = self.gt_sample_log_weight
        stats["proposal_logits_total"] = self.proposal_logits_total
