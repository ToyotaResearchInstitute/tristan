import copy
import os
import pickle
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from intent.multiagents.trainer_utils import reformat_tensor
from loaders.ado_key_names import AGENT_TYPE_NAME_MAP
from model_zoo.intent.prediction_model_interface import PredictionModelInterface
from radutils.torch.async_saver import AsyncModelSaver
from radutils.torch.torch_utils import apply_2d_coordinate_rotation_transform
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from util.prediction_metrics import displacement_errors


def accumulate(tgt_dict, key, value):
    if key not in tgt_dict:
        tgt_dict[key] = 0
    tgt_dict[key] += value


def copy_encoder(enc: nn.Module) -> nn.Module:
    """Deep-copy an encoder object, does not deep-copy the parameters.

    Parameters
    ----------
    enc: nn.Module
      an encoder object

    Returns
    -------
      encoder_copy, a copy of the encoder object.
    """
    has_params = hasattr(enc, "params")
    if has_params:
        params = enc.params
        enc.params = {}
    enc_copy = copy.deepcopy(enc)
    if has_params:
        enc_copy.params = params
        enc.params = params
    return enc_copy


def add_samples_to_batch(tensor, num_agents):
    """
    Make a tensor compatible with the [batch * num_agents, ...] format, starting with a tensor shaped like [batch, ...].
    The tensor is expanded to accomplish this.
    """
    if tensor is None:
        return None
    tensor_shape = tensor.shape
    expanded_tensor = tensor.unsqueeze(1).expand(tensor_shape[0:1] + torch.Size([num_agents]) + tensor_shape[1:])
    return expanded_tensor.contiguous().view(tensor_shape[0] * num_agents, *list(tensor_shape[1:]))


class PredictionModel(PredictionModelInterface):
    """
    A general class for prediction models that generate trajectory predictions.

    Parameters
    ----------
    models : dict
        A dictionary of models.
    params : dict
        A dictionary of parameters.
    additional_model_callbacks : list
        List of additional model callbacks (i.e. hybrid).
    """

    def __init__(
        self,
        models: dict,
        params: dict,
        device,
        input_encoders,
        agent_input_encoders,
        agent_input_handlers,
        additional_model_callbacks=None,
    ) -> None:
        """
        Initialize class parameters.
        """
        super().__init__(params=params)
        # TODO(guy.rosman): add a dictionary mapping from input key to child network type
        # TODO(guy.rosman): instantiate, plug the child networks into the encoder
        # TODO(guy.rosman): instantiate, plug the child networks into the discriminator (past) encoder
        # TODO(guy.rosman): consider -- do we sometime need to plug into future decoder/ discriminator encoder? e.g. maps

        self.dropout_ratio = params["dropout_ratio"]
        self.device = device

        self.input_encoders = input_encoders
        self.agent_input_encoders = agent_input_encoders
        self.agent_input_handlers = agent_input_handlers

        self.bceloss = nn.BCELoss()

        self.model_encap = models["model_encap"]
        self.encoder = self.model_encap.get_encoder()
        self.decoder = self.model_encap.get_decoder()

        # Flag to activate discriminator.
        self.use_discriminator = params["use_discriminator"]

        # Additional model callbacks.
        self.additional_model_callbacks = additional_model_callbacks or []

        if self.params["multigpu"]:
            self.agent_input_encoders.update(
                (key, nn.DataParallel(encoder)) for key, encoder in self.agent_input_encoders.items()
            )
            self.input_encoders.update((key, nn.DataParallel(encoder)) for key, encoder in self.input_encoders.items())
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.model_encap = nn.DataParallel(self.model_encap)

        if self.use_discriminator:
            self.discriminator_input_encoders = nn.ModuleDict(
                (key, copy_encoder(self.input_encoders[key])) for key, encoder in self.input_encoders.items()
            )
            self.discriminator_agent_input_encoders = nn.ModuleDict(
                (key, copy_encoder(self.agent_input_encoders[key]))
                for key, encoder in self.agent_input_encoders.items()
            )

            if self.params["multigpu"]:
                self.discriminator_agent_input_encoders.update(
                    (key, nn.DataParallel(encoder)) for key, encoder in self.discriminator_agent_input_encoders.items()
                )
                self.discriminator_input_encoders.update(
                    (key, nn.DataParallel(encoder)) for key, encoder in self.discriminator_input_encoders.items()
                )
                self.discriminator_encoder = self.model_encap.module.get_discriminator_encoder()
                self.discriminator_future_encoder = self.model_encap.module.get_discriminator_future_encoder()
                self.discriminator_head = self.model_encap.module.get_discriminator_head()
            else:
                self.discriminator_encoder = self.model_encap.get_discriminator_encoder()
                self.discriminator_future_encoder = self.model_encap.get_discriminator_future_encoder()
                self.discriminator_head = self.model_encap.get_discriminator_head()

        if self.params["special_init"] and self.use_discriminator:
            nn.init.normal_(self.discriminator_head[0].bias, 0.0, 0.1)
            nn.init.xavier_normal_(self.discriminator_head[0].weight)

            nn.init.normal_(self.discriminator_head[-1].bias, 0.0, 0.1)
            nn.init.xavier_normal_(self.discriminator_head[-1].weight)

        self.use_dropout = False
        self.child_network_dropout = params["child_network_dropout"]

        self.model_saver = AsyncModelSaver()

        self.current_num_agents = 0

    def set_logger(self, logger):
        super().set_logger(logger)

        encoders_to_update = (
            *self.input_encoders,
            *self.agent_input_encoders,
        )
        if self.use_discriminator:
            encoders_to_update = (
                *encoders_to_update,
                *self.discriminator_input_encoders,
                *self.discriminator_agent_input_encoders,
            )
        for encoder in encoders_to_update:
            if hasattr(encoder, "set_logger"):
                encoder.set_logger(logger)

    def get_input_encoders(self):
        return self.input_encoders

    def get_agent_input_encoders(self):
        return self.agent_input_encoders

    def get_discriminator_input_encoders(self):
        return self.discriminator_input_encoders

    def get_discriminator_agent_input_encoders(self):
        return self.discriminator_agent_input_encoders

    # # TODO (igor.gilitscheski): This method should not be necessary. If everything is
    # # TODO  placed in the right containers, PyTorch knows on its own what attributes are child modules.
    # def to(self, device):
    #     self.device = device
    #     self.model_encap = self.model_encap.to(device)
    #
    #     self.input_encoders.update((key, encoder.to(device)) for key, encoder in self.input_encoders.items())
    #     self.agent_input_encoders.update(
    #         (key, encoder.to(device)) for key, encoder in self.agent_input_encoders.items()
    #     )
    #
    #     if self.use_discriminator:
    #         self.discriminator_encoder = self.discriminator_encoder.to(device)
    #         self.discriminator_future_encoder = self.discriminator_future_encoder.to(device)
    #         self.discriminator_head = self.discriminator_head.to(device)
    #         self.discriminator_input_encoders.update(
    #             (key, encoder.to(device)) for key, encoder in self.discriminator_input_encoders.items()
    #         )
    #         self.discriminator_agent_input_encoders.update(
    #             (key, encoder.to(device)) for key, encoder in self.discriminator_agent_input_encoders.items()
    #         )
    #
    #     self.bceloss = self.bceloss.to(device)
    #     return self

    def child_net_train(self):
        self.use_dropout = True

    def child_net_eval(self):
        self.use_dropout = False

    def get_dropout_draw(self, batch_size, num_inputs):
        if not self.use_dropout:
            return np.float32(np.ones([batch_size, num_inputs]))
        else:
            result = np.zeros([batch_size, num_inputs])
            for b in range(batch_size):
                while not np.any(result[b, :]):
                    result[b, :] = np.random.uniform(size=result[b, :].shape) > self.child_network_dropout
            result = np.float32(result)
            return result

    def fuse_scene_tensor(
        self, additional_inputs: dict, batch_size: float, trajectory_data, discriminator=False, skip_visualization=None
    ):
        """
        Fuse additional inputs, after encoding them. Does child dropout if it is set.
        :param additional_inputs: a list of input tensors
        :param batch_size:
        :return: scene_inputs_tensor of size [batch_size,time, total dimensionality of encoded inputs]
        """
        num_total_timesteps = self.params["past_timesteps"] + self.params["future_timesteps"]
        dropout_draws = self.get_dropout_draw(batch_size, len(additional_inputs) + 1)
        _, num_past_timepoints = trajectory_data.shape[1:3]
        overall_additional_costs = {}
        if skip_visualization is None:
            skip_visualization = False
        additional_params = {"skip_visualization": skip_visualization}
        if len(additional_inputs) > 0:
            # save the N+1 dropout for trajectories.
            additional_inputs_enc = []
            for key_i, key in enumerate(self.input_encoders.keys()):
                if discriminator:
                    enc, additional_costs = self.discriminator_input_encoders[key](
                        additional_inputs[key], trajectory_data, additional_params=additional_params
                    )
                else:
                    enc, additional_costs = self.input_encoders[key](additional_inputs[key], trajectory_data)

                # TODO(igor.gilitschenski): In the philosophy of our structure, this should be done in ImageEncoder,
                # TODO  which will require a change to the interface of our encoders allowing them to know the key.
                if key == ProtobufPredictionDataset.DATASET_KEY_IMAGES:
                    # Shape (batch_size, num_total_timepoints, enc_output_dim)
                    enc = reformat_tensor(
                        enc,
                        additional_inputs[ProtobufPredictionDataset.DATASET_KEY_IMAGES_MAPPING],
                        (num_total_timesteps,),
                    )

                    # Shape (batch_size, num_past_timepoints)
                    enc = enc[:, :num_past_timepoints]

                # We need to do pooling here because the scene tensor expects a fixed sized tensor for
                # all types of keys. It might be better to do it in the map encoder.
                if key in [ProtobufPredictionDataset.DATASET_KEY_MAP, ProtobufPredictionDataset.DATASET_KEY_MAP_STUB]:
                    # Update the encoded global map features to the inputs
                    additional_inputs["encoded_map"] = enc
                    # Average pooling over map segments
                    enc = torch.mean(enc, dim=1)
                    # Shape (batch_size, num_total_timepoints, enc_output_dim)
                    enc = enc.repeat(num_total_timesteps, 1, 1).transpose(0, 1)

                if key in self.params["scene_inputs_nullify"]:
                    additional_inputs_enc.append(torch.zeros_like(enc))
                else:
                    for costs_key in additional_costs:
                        accumulate(overall_additional_costs, costs_key, additional_costs[costs_key])
                    enc_dropped = (
                        enc.transpose(0, -1) * additional_inputs[key].new_tensor(dropout_draws[:, key_i])
                    ).transpose(0, -1)
                    additional_inputs_enc.append(enc_dropped)

            if len(additional_inputs_enc) > 0:
                scene_inputs_tensor = torch.cat(additional_inputs_enc, dim=-1)
            else:
                scene_inputs_tensor = None
        else:
            scene_inputs_tensor = None
        return scene_inputs_tensor, dropout_draws, overall_additional_costs

    # TODO(guy.rosman) fix this and scene tensor
    def fuse_agent_tensor(
        self, agent_additional_inputs, trajectory_data, transforms, discriminator=False, skip_visualization=None
    ):
        """
        Fuse agent_additional inputs, after encoding them.
        :param agent_additional_inputs: a list of input tensors.
        :param trajectory_data: agent trajectory data.
        :param transforms: transforms from global coordinate to local coordinate.
        :param discriminator: whether discriminator is called.
        :return: agent_inputs_tensor of size [batch_size,N_agents, time,total dimensionality of encoded inputs]
        """
        # Apply input encoder handler if necessary.
        batch_size, num_agents, num_past_timepoints = trajectory_data.shape[0:3]
        num_total_timesteps = self.params["past_timesteps"] + self.params["future_timesteps"]
        dropout_draws = self.get_dropout_draw(batch_size, len(agent_additional_inputs) + 1)
        dropout_draws = trajectory_data.new_tensor(dropout_draws)
        for handler in self.agent_input_handlers:
            handler(agent_additional_inputs, transforms, self.should_normalize, self.params)
        if skip_visualization is None:
            skip_visualization = True
        additional_params = {"skip_visualization": skip_visualization}

        overall_additional_costs = {}
        if len(agent_additional_inputs) > 0:
            agent_additional_inputs_enc = []
            for key_i, key in enumerate(self.agent_input_encoders.keys()):
                # TODO(igor.gilitschenski): Here we are injecting encoder_input explicitly and another time implicitly
                # TODO  via agent_additional_inputs. This doubling seems not necessary.
                if discriminator:
                    enc_input = agent_additional_inputs[key]
                    enc, additional_costs = self.discriminator_agent_input_encoders[key](
                        enc_input,
                        trajectory_data=trajectory_data,
                        agent_additional_inputs=agent_additional_inputs,
                        additional_params=additional_params,
                    )
                else:
                    # import IPython;IPython.embed(header='fuse_agent_tensor')
                    enc_input = agent_additional_inputs[key]
                    enc, additional_costs = self.agent_input_encoders[key](
                        enc_input,
                        trajectory_data=trajectory_data,
                        agent_additional_inputs=agent_additional_inputs,
                        additional_params=additional_params,
                    )

                # Check for agent images.
                if key == ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES:
                    # Shape (batch_size, num_agents, num_total_timepoints, enc_output_dim)
                    enc = reformat_tensor(
                        enc,
                        agent_additional_inputs[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES_MAPPING],
                        (num_agents, num_total_timesteps),
                    )

                    # Shape (batch_size, num_agents, num_past_timepoints)
                    enc = enc[:, :, :num_past_timepoints]

                if key in self.params["agent_inputs_nullify"]:
                    agent_additional_inputs_enc.append(torch.zeros_like(enc))
                else:
                    for key2 in additional_costs:
                        accumulate(overall_additional_costs, key2, additional_costs[key2])
                    enc_dropped = enc * dropout_draws[:, key_i].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    agent_additional_inputs_enc.append(enc_dropped)

            if len(agent_additional_inputs_enc) > 0:
                agent_inputs_tensor = torch.cat(agent_additional_inputs_enc, dim=-1)
            else:
                agent_inputs_tensor = None
        else:
            agent_inputs_tensor = None
        return agent_inputs_tensor, overall_additional_costs

    def normalize_trajectories(self, trajectory_data, is_valid, transforms, invert=False):
        """
        Normalizes trajectory with a given set of affine transforms. Can be forward or inverse transformation.
        :param trajectory_data: The trajectories, [B x N_agents x N_timepoints x 3] tensor.
        :param is_valid: [B x N_agents x N_timepoints] tensor.
        :param transforms: [B x N_agents x 3 x 2], affine transform matrices.
        :param invert: boolean.
        :return: normalized trajectory data.
        """
        if not self.should_normalize:
            return trajectory_data

        normalized_trajectory_data = trajectory_data.clone()
        if invert:
            # Rotate each agent.
            normalized_trajectory_data[..., :2] = apply_2d_coordinate_rotation_transform(
                torch.linalg.inv(transforms[:, :, :2]),
                trajectory_data[..., :2],
                result_einsum_prefix="bat",
                rotation_einsum_prefix="ba",
            )
            # Shift each agent.
            normalized_trajectory_data[..., :2] = normalized_trajectory_data[..., :2] - transforms[:, :, 2].unsqueeze(2)
        else:
            # Shift each agent.
            normalized_trajectory_data[..., :2] = trajectory_data[..., :2] + transforms[:, :, 2].unsqueeze(2)
            # Rotate each agent.
            normalized_trajectory_data[..., :2] = apply_2d_coordinate_rotation_transform(
                transforms[:, :, :2],
                normalized_trajectory_data[..., :2],
                result_einsum_prefix="bat",
                rotation_einsum_prefix="ba",
            )
        return normalized_trajectory_data

    def forward(
        self,
        trajectory_data,
        additional_inputs,
        agent_additional_inputs,
        relevant_agents,
        agent_type,
        is_valid,
        timestamps,
        prediction_timestamp,
        skip_visualization: Optional[bool] = None,
        additional_params=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        :param trajectory_data: a B x N_agents x N_timepoints x3 tensor
        :param additional_inputs: a dictionary of key -> additional input
        :param is_valid: [B x N_agents x N_timepoints] tensor.
        :return: A tuple of [B x N_agents x N_timepoints x2 x num_samples] tensor, [B x N_timepoints x joint decoding space x num_samples] tensor, dictionary of auxiliary stats.
        """
        # Generate scene relevant data (consistent across all agents).
        scene_inputs_tensor, dropout, scene_additional_costs = self.fuse_scene_tensor(
            additional_inputs,
            batch_size=is_valid.shape[0],
            trajectory_data=trajectory_data / self.params["predictor_normalization_scale"],
            skip_visualization=skip_visualization,
        )
        # Apply local transformation.
        transforms_local_scene = self.compute_normalizing_transforms(
            trajectory_data, is_valid, timestamps, prediction_timestamp
        )
        normalized_trajectory_data = self.normalize_trajectories(trajectory_data, is_valid, transforms_local_scene)

        # Normalize ground truth future positions if they exist.
        if "future_positions" in additional_inputs:
            additional_inputs["future_positions"] = self.normalize_trajectories(
                additional_inputs["future_positions"], is_valid, transforms_local_scene
            )

        for cb in self.additional_model_callbacks:
            cb.update_agent_additional_inputs(
                agent_additional_inputs, transforms_local_scene, self.normalize_trajectories
            )

        # Fuse agent state.
        # [batch_size, num_agent, num_past_steps, agent_inputs_tensor_dim]
        # TODO(guy.rosman): fix agents to handle agents' coordinates
        agent_inputs_tensor, agent_additional_costs = self.fuse_agent_tensor(
            agent_additional_inputs,
            trajectory_data=normalized_trajectory_data,
            transforms=transforms_local_scene,
            skip_visualization=skip_visualization,
        )

        # Compute trajectory data after dropout.
        dropped_trajectory_data = torch.mul(
            trajectory_data.new_tensor(dropout[:, -1:])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(torch.Size([1]) + trajectory_data.shape[1:]),
            trajectory_data,
        )
        dropped_normalized_trajectory_data = torch.mul(
            trajectory_data.new_tensor(dropout[:, -1:])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(torch.Size([1]) + trajectory_data.shape[1:]),
            normalized_trajectory_data,
        )

        additional_costs = copy.copy(scene_additional_costs)
        additional_costs.update(agent_additional_costs)
        # TODO(cyrushx): Add map input to discriminator.
        results_list, decoding_list, stats = self.model_encap.generate_trajectory_sample(
            trajectory_data,
            dropped_trajectory_data,
            normalized_trajectory_data,
            dropped_normalized_trajectory_data,
            additional_inputs,
            agent_additional_inputs,
            transforms_local_scene,
            agent_type,
            is_valid,
            scene_inputs_tensor,
            agent_inputs_tensor,
            relevant_agents,
            additional_params=additional_params,
        )
        # Unnormalize trajectories, depending on whether a single sample is returned,
        # or multiple samples are returned.
        if results_list:
            result = torch.cat(results_list, 4)
            # import IPython;IPython.embed()
        if decoding_list is not None:
            joined_decoding = torch.cat(decoding_list, -1)
        else:
            joined_decoding = None

        if len(tuple(result.shape)) == 5:
            sample_size = result.shape[-1]
            result_multiple_samples = []
            for s in range(sample_size):
                result_s = self.normalize_trajectories(result[..., s], is_valid, transforms_local_scene, invert=True)
                result_multiple_samples.append(result_s.unsqueeze(4))
            # Shape: [batch_size, num_agents, num_future_steps, traj_dim, num_samples]
            result = torch.cat(result_multiple_samples, dim=4)
        else:
            result = self.normalize_trajectories(result, is_valid, transforms_local_scene, invert=True)

        for cb in self.additional_model_callbacks:
            cb.update_prediction_stats(
                self.params, stats, is_valid, sample_size, transforms_local_scene, self.normalize_trajectories
            )

        # Pass agent rotation and velocity for computing metrics.
        stats[0]["agent_transforms"] = transforms_local_scene

        return result, joined_decoding, stats, additional_costs

    # TODO (igor.gilitscheski): This method should not be necessary and has been a source of bugs.
    # If everything is placed in the right containers, PyTorch knows on its own what attributes
    # are child modules. Storing of command line arguments can be done in the trainer.
    def save_model(
        self,
        data,
        is_valid: bool,
        folder: str,
        checkpoint: torch.Tensor = None,
        use_async: bool = False,
        save_to_s3: bool = False,
    ):
        # TODO(guy.rosman): use jit.trace to save modules for TensorRT deployment (see intent/TrajectoryEstimator.py).
        # TODO(guy.rosman): add loading for training/inspection.
        models_to_save = {
            "decoder": self.decoder,
            "encoder": self.encoder,
            "inputs_encoders": self.input_encoders,
            "agent_inputs_encoders": self.agent_input_encoders,
        }
        if self.use_discriminator:
            models_to_save.update(
                {
                    "discriminator_encoder": self.discriminator_encoder,
                    "discriminator_future_encoder": self.discriminator_future_encoder,
                    "discriminator_inputs_encoders": self.discriminator_input_encoders,
                    "discriminator_agent_inputs_encoders": self.discriminator_agent_input_encoders,
                }
            )
        if self.params["use_latent_factors"]:
            if "latent_factors" in self.params:
                models_to_save.update({"latent_factors": self.params["latent_factors"]})
            elif self.model_encap.latent_factors is not None:
                # Backward compatible to original latent factors.
                models_to_save.update({"latent_factors": self.model_encap.latent_factors})

        if not use_async:
            self.model_saver.save_model_sync(models_to_save, checkpoint, folder, save_to_s3)
        else:
            self.model_saver.set_model_to_save(models_to_save, checkpoint, folder, save_to_s3)

        # Save language vocabs
        if "language_vocab" in self.params:
            with open(folder + "/language_vocab.pkl", "wb") as fp:
                pickle.dump(self.params["language_vocab"], fp)

        # TODO(guy.rosman): add child network saving

    # TODO (igor.gilitschenski): This method can be deprecated once save_model is deprecated.
    def load_model(self, folder):
        models_to_load = {
            "decoder": self.decoder,
            "encoder": self.encoder,
            "inputs_encoders": self.input_encoders,
            "agent_inputs_encoders": self.agent_input_encoders,
        }
        if self.use_discriminator:
            models_to_load.update(
                {
                    "discriminator_encoder": self.discriminator_encoder,
                    "discriminator_future_encoder": self.discriminator_future_encoder,
                    "discriminator_inputs_encoders": self.discriminator_input_encoders,
                    "discriminator_agent_inputs_encoders": self.discriminator_agent_input_encoders,
                }
            )
        if self.params["use_latent_factors"]:
            if "latent_factors" in self.params:
                models_to_load.update({"latent_factors": self.params["latent_factors"]})
            elif self.model_encap.latent_factors is not None:
                # Backward compatible to original latent factors.
                models_to_load.update({"latent_factors": self.model_encap.latent_factors})
        for modelname in models_to_load:
            model = models_to_load[modelname]
            fname_model = os.path.join(folder, "model_{}.pth".format(modelname))

            if not os.path.exists(fname_model) and "encoders" in modelname:
                print(f"WARNING: Loading legacy model without parameters for {modelname}")
            print("Loading model from {}".format(fname_model))
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            d = torch.load(fname_model, map_location=self.device)
            model.load_state_dict(d, strict=False)
        # Loab language vocabs
        if "language_vocab" in self.params:
            vocab_path = folder + "/language_vocab.pkl"
            with open(vocab_path, "rb") as fp:
                print("Loading vocabulary from {}".format(vocab_path))
                self.params["language_vocab"] = pickle.load(fp)

    def count_parameters(self):
        text_net_params = ""
        models_to_count = {
            "decoder": self.decoder,
            "encoder": self.encoder,
        }
        for key in self.input_encoders:
            new_key = key + "_encoder"
            models_to_count[new_key] = self.input_encoders[key]
        for key in self.agent_input_encoders:
            new_key = "agent_" + key + "_encoder"
            models_to_count[new_key] = self.agent_input_encoders[key]
        generator_models_to_count = self.model_encap.count_parameters()
        models_to_count.update(generator_models_to_count)

        if self.use_discriminator:
            models_to_count.update(
                {
                    "discriminator_encoder": self.discriminator_encoder,
                    "discriminator_future_encoder": self.discriminator_future_encoder,
                    "discriminator_head": self.discriminator_head,
                }
            )
            for key in self.discriminator_input_encoders:
                new_key = "discriminator_" + key + "_encoder"
                models_to_count[new_key] = self.discriminator_input_encoders[key]
            for key in self.discriminator_agent_input_encoders:
                new_key = "discriminator_agent_" + key + "_encoder"
                models_to_count[new_key] = self.discriminator_agent_input_encoders[key]

        for modelname, model in models_to_count.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # if modelname.find('images')>-1:
            # import IPython;IPython.embed(header='check_learnables: {}'.format(modelname))
            txt_sum = "  \n{}  \nlearnable params: [{}]  \ntotal params: [{}]  \n".format(
                modelname, trainable_params, total_params
            )
            text_net_params += txt_sum
            for layer, layer_state in model.state_dict().items():
                txt_layer = "{}: [{}] params, {}  \n".format(layer, layer_state.numel(), layer_state.size())
                text_net_params += txt_layer
        return text_net_params

    def get_generator_parameters(self, require_grad=True):
        p_dict = OrderedDict()

        for p_i, p in enumerate(self.encoder.parameters()):
            if p.requires_grad or not require_grad:
                p_dict["encoder_" + str(p_i)] = p
        for p_i, p in enumerate(self.decoder.parameters()):
            if p.requires_grad or not require_grad:
                p_dict["decoder_" + str(p_i)] = p
        if not self.params["scene_inputs_detach"]:
            for key in self.input_encoders:
                for p_i, p in enumerate(self.input_encoders[key].parameters()):
                    if p.requires_grad or not require_grad:
                        p_dict["input_encoders_" + key + "_" + str(p_i)] = p
        if not self.params["agent_inputs_detach"]:
            for key in self.agent_input_encoders:
                for p_i, p in enumerate(self.agent_input_encoders[key].parameters()):
                    if p.requires_grad or not require_grad:
                        p_dict["input_encoders_" + key + "_" + str(p_i)] = p
        # import IPython;
        # IPython.embed(header='check what is frozen')
        # TODO(guy.rosman) -- Cyrus, can we verify?
        for p_i, p in enumerate(list(self.model_encap.get_parameters(require_grad))):
            p_dict["encap_" + str(p_i)] = p

        return p_dict

    def get_discriminator_parameters(self, require_grad=True):
        p_dict = OrderedDict()
        if self.use_discriminator:
            for p_i, p in enumerate(self.discriminator_encoder.parameters()):
                if p.requires_grad or not require_grad:
                    p_dict["discriminator_encoder" + "_" + str(p_i)] = p
            for p_i, p in enumerate(self.discriminator_future_encoder.parameters()):
                if p.requires_grad or not require_grad:
                    p_dict["discriminator_future_encoder" + "_" + str(p_i)] = p

            for p_i, p in enumerate(self.discriminator_head.parameters()):
                if p.requires_grad or not require_grad:
                    p_dict["discriminator_head" + "_" + str(p_i)] = p
            for key in self.discriminator_input_encoders:
                for p_i, p in enumerate(self.discriminator_input_encoders[key].parameters()):
                    if p.requires_grad or not require_grad:
                        p_dict["discriminator_input_encoders" + "_" + str(p_i)] = p
            for key in self.discriminator_agent_input_encoders:
                for p_i, p in enumerate(self.discriminator_agent_input_encoders[key].parameters()):
                    if p.requires_grad or not require_grad:
                        p_dict["discriminator_agent_input_encoders" + "_" + str(p_i)] = p

        return p_dict

    def get_latent_factors_parameters(self, require_grad=True):
        p_dict = OrderedDict()
        for p_i, p in enumerate(list(self.model_encap.latent_factors.parameters())):
            if p.requires_grad or not require_grad:
                p_dict["latent_factors_" + str(p_i)] = p
        return p_dict

    def generate_noise_samples(self, batch_size, num_samples, sample_dim) -> torch.Tensor:
        return torch.randn(batch_size, num_samples, sample_dim)

    def generate_trajectory(
        self,
        input_trajectory,
        additional_inputs,
        agent_additional_inputs,
        relevant_agents,
        agent_type,
        is_valid,
        timestamps,
        prediction_timestamp,
        num_samples=1,
        additional_params=None,
    ):
        """
        Generate a trajectory sample
        :param input_trajectory: [B x N_agents x N_timepoints x3] tensor, 3 is for: x,y, is_valid
        :param additional_inputs: A dictionary key -> BxN_timepoints x d tensor, depending on the input type
        :param agent_additional_inputs: A dictionary key -> BxN_agentsxN_timepoints x d tensor
        :param relevant_agents:[B x N_agents] Relevant agent mask
        :param agent_type: [B x N_agents N_types] one-hot vector of agents types
        :param is_valid: [B x N_agents x N_timepoints] tensor of validity bits for agents
        :param timestamps: [B x N_timepoints] tensor of timestamps
        :param prediction_timestamp: the timestamp at which predictions are expected.
        :param num_samples:int, how many samples to return for each trajectory.
        :param additional_params:
        :return: results, decoding, stats where results is [B x N_agents x N_timepoints x 2 x num_samples]
        tensor of coordinates, decoding is predictor-specific, stats is a predictor-specific dictionary of statistics.
        """
        stats = None
        decoding = None

        # Create noise samples for decoder.
        noise_samples = self.generate_noise_samples(
            batch_size=input_trajectory.shape[0], num_samples=num_samples, sample_dim=32
        ).to(input_trajectory.device)
        if self.params["deterministic_prediction"]:
            noise_samples = torch.zeros_like(noise_samples)
        additional_inputs["noise_samples"] = noise_samples
        skip_visualization = additional_params is not None and additional_params["skip_visualization"]
        # Generate predictions.
        additional_inputs["sample_index"] = list(range(num_samples))
        results, decoding, stats, additional_costs = self.forward(
            input_trajectory,
            additional_inputs,
            agent_additional_inputs,
            relevant_agents,
            agent_type,
            is_valid,
            timestamps,
            prediction_timestamp,
            skip_visualization=skip_visualization,
        )

        if "decoder/decoder_x0" in stats[0]:
            distance_x0, _ = (
                (stats[0]["decoder/decoder_x0"] ** 2 + stats[0]["decoder/decoder_y0"] ** 2).sqrt().max(dim=1)
            )
            stats[0]["distance_x0"] = distance_x0

        return results, decoding, stats, additional_costs

    def discriminate_trajectory_samples(
        self,
        past_trajectory,
        past_inputs,
        agent_additional_inputs,
        future_trajectory_samples,
        expected,
        agent_type,
        is_past_valid,
        is_future_valid,
        timestamps,
        prediction_timestamp,
        relevant_agents,
        additional_param=None,
    ):
        """
        This is basically a wrapper to call prediction_model_codec.discriminate_trajectory.
        This function differs from discriminate_trajectory in that it will combine trajectory samples per agent
        in the batch dimension to allow batching of multi-sample discrimination.
        :param past_trajectory: [batch, agent, timepoint, 2]
        :param past_inputs: ['images', 'images_mapping', 'future_positions', 'noise_samples', 'sample_index']
        :param agent_additional_inputs: ['agent_images', 'agent_images_mapping', 'map', 'get_traj_map_embeddings']
        :param future_trajectory_samples: [batch, agent, timepoint, 2, samples]
        :param expected: [batch, agent, timepoint, 2]
        :param agent_type: [batch, agent, num_agent_types]
        :param is_past_valid: [batch, agent, timepoint]
        :param is_future_valid: [batch, agent, timepoint]
        :param timestamps: [batch, timepoint * 2] All timestamps forward and backward in time.
        :param prediction_timestamp: [batch]
        :param relevant_agents: [batch, agent]
        :param additional_param=None: Optionally a dictionary with "skip_visualization" true or false.
        :return: [batch]
        """
        num_past_timesteps = past_trajectory.shape[2]
        batch_size, num_agents, num_future_timesteps, traj_dim, num_samples = future_trajectory_samples.shape
        relevant_agents = relevant_agents.float()
        irrelevant_agents = 1.0 - relevant_agents
        # Include the trajectory samples in the batch dimension so we don't loop over samples in python
        future_trajectory_batch_sample = (
            future_trajectory_samples.permute(0, 4, 1, 2, 3)
            .contiguous()
            .view(batch_size * num_samples, num_agents, num_future_timesteps, traj_dim)
        )
        # replace the future trajectories of irrelevant agents with the acausal data
        # TODO(cyrushx): [relevant_agents] Why we want to use expected values for irrelevant agents (they might be invalid)?
        future_trajectory_batch_sample = future_trajectory_batch_sample * add_samples_to_batch(
            relevant_agents[:, :, None, None], num_samples
        )
        if not self.params["nullify_irrelevant_discriminator_agents"]:
            future_trajectory_batch_sample = future_trajectory_batch_sample + add_samples_to_batch(
                expected * irrelevant_agents[:, :, None, None], num_samples
            )

        # modified_is_future_valid = is_future_valid
        modified_is_future_valid = is_future_valid * relevant_agents[:, :, None]

        past_traj_with_validity = torch.cat([past_trajectory, is_past_valid.unsqueeze(3).float()], 3)

        transforms_local_scene = self.compute_normalizing_transforms(
            past_traj_with_validity, is_past_valid, timestamps[:, :num_past_timesteps], prediction_timestamp
        )
        normalized_past_trajectory = self.normalize_trajectories(
            past_traj_with_validity, is_past_valid, transforms_local_scene
        )
        normalized_future_trajectory_batch_sample = self.normalize_trajectories(
            future_trajectory_batch_sample,
            add_samples_to_batch(is_future_valid, num_samples),
            add_samples_to_batch(transforms_local_scene, num_samples),
        )
        skip_visualization = additional_param is not None and additional_param["skip_visualization"]

        scene_data, dropout, _ = self.fuse_scene_tensor(
            past_inputs, batch_size=batch_size, trajectory_data=past_traj_with_validity, discriminator=True
        )
        agent_data, _ = self.fuse_agent_tensor(
            agent_additional_inputs,
            trajectory_data=normalized_past_trajectory,
            transforms=transforms_local_scene,
            discriminator=True,
            skip_visualization=skip_visualization,
        )
        dropped_past_trajectory_data = torch.mul(
            past_traj_with_validity.new_tensor(dropout[:, -1:])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(torch.Size([1]) + past_traj_with_validity.shape[1:]),
            past_traj_with_validity,
        )
        normalized_past_trajectory = torch.mul(
            normalized_past_trajectory.new_tensor(dropout[:, -1:])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(torch.Size([1]) + normalized_past_trajectory.shape[1:]),
            normalized_past_trajectory,
        )

        result = self.model_encap.discriminate_trajectory(
            add_samples_to_batch(dropped_past_trajectory_data, num_samples),  # [batch * samples, agent, timepoint, 3]
            add_samples_to_batch(normalized_past_trajectory, num_samples),  # [batch * samples, agent, timepoint, 3]
            add_samples_to_batch(agent_type, num_samples),  # [batch * samples, agent, num_agent_types]
            add_samples_to_batch(is_past_valid, num_samples),  # [batch * samples, agent, timepoints]
            add_samples_to_batch(scene_data, num_samples),  # [batch * samples, agent, scene_dim]
            add_samples_to_batch(agent_data, num_samples),  # [batch * samples, agent, agent_dim]
            add_samples_to_batch(relevant_agents, num_samples),  # [batch * samples, agent]
            future_trajectory_batch_sample,  # [batch * samples, agent, timepoint, 2]
            normalized_future_trajectory_batch_sample,  # [batch * samples, agent, timepoint, 2]
            add_samples_to_batch(modified_is_future_valid, num_samples),  # [batch * samples, agent, timepoint]
        )
        additional_costs = {}
        return result, additional_costs  # [batch], {}

    def discriminate_trajectory(
        self,
        past_trajectory,
        past_inputs,
        agent_additional_inputs,
        future_trajectory,
        expected,
        agent_type,
        is_past_valid,
        is_future_valid,
        timestamps,
        prediction_timestamp,
        relevant_agents,
        additional_param=None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        :param past_trajectory: [batch, agent, timepoint, 2]
        :param past_inputs: ['images', 'images_mapping', 'future_positions', 'noise_samples', 'sample_index']
        :param agent_additional_inputs: ['agent_images', 'agent_images_mapping', 'map', 'get_traj_map_embeddings']
        :param future_trajectory: [batch, agent, timepoint, 2]
        :param expected: [batch, agent, timepoint, 2]
        :param agent_type: [batch, agent, num_agent_types]
        :param is_past_valid: [batch, agent, timepoint]
        :param is_future_valid: [batch, agent, timepoint]
        :param timestamps: [batch, timepoint * 2] All timestamps forward and backward in time.
        :param prediction_timestamp: [batch]
        :param relevant_agents: [batch, agent]
        :param additional_param=None: Optionally a dictionary with "skip_visualization" true or false.
        :return: [batch]
        """
        print("Discriminating regular trajectory")
        batch_size, num_agents, num_past_timesteps = past_trajectory.shape[0:3]
        relevant_agents = relevant_agents.float()
        irrelevant_agents = 1.0 - relevant_agents
        # replace the future trajectories of irrelevant agents with the acausal data
        # TODO(cyrushx): [relevant_agents] Why we want to use expected values for irrelevant agents (they might be invalid)?
        modified_future_trajectory = future_trajectory * relevant_agents[:, :, None, None]
        if not self.params["nullify_irrelevant_discriminator_agents"]:
            modified_future_trajectory = modified_future_trajectory + expected * irrelevant_agents[:, :, None, None]

        modified_is_future_valid = is_future_valid * relevant_agents[:, :, None]

        past_traj_with_validity = torch.cat([past_trajectory, is_past_valid.unsqueeze(3).float()], 3)

        transforms_local_scene = self.compute_normalizing_transforms(
            past_traj_with_validity, is_past_valid, timestamps[:, :num_past_timesteps], prediction_timestamp
        )
        normalized_past_trajectory = self.normalize_trajectories(
            past_traj_with_validity, is_past_valid, transforms_local_scene
        )
        normalized_future_trajectory = self.normalize_trajectories(
            future_trajectory, is_future_valid, transforms_local_scene
        )
        skip_visualization = additional_param is not None and additional_param["skip_visualization"]

        scene_data, dropout, _ = self.fuse_scene_tensor(
            past_inputs, batch_size=batch_size, trajectory_data=past_traj_with_validity, discriminator=True
        )
        agent_data, _ = self.fuse_agent_tensor(
            agent_additional_inputs,
            trajectory_data=normalized_past_trajectory,
            transforms=transforms_local_scene,
            discriminator=True,
            skip_visualization=skip_visualization,
        )
        dropped_past_trajectory_data = torch.mul(
            past_traj_with_validity.new_tensor(dropout[:, -1:])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(torch.Size([1]) + past_traj_with_validity.shape[1:]),
            past_traj_with_validity,
        )
        normalized_past_trajectory = torch.mul(
            normalized_past_trajectory.new_tensor(dropout[:, -1:])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(torch.Size([1]) + normalized_past_trajectory.shape[1:]),
            normalized_past_trajectory,
        )

        result = self.model_encap.discriminate_trajectory(
            dropped_past_trajectory_data,  # [batch, agent, timepoint, 3]
            normalized_past_trajectory,  # [batch, agent, timepoint, 3]
            agent_type,  # [batch, agent, num_agent_types]
            is_past_valid,  # [batch, agent, timepoints]
            scene_data,  # [batch, agent, scene_dim]
            agent_data,  # [batch, agent, agent_dim]
            relevant_agents,  # [batch, agent]
            modified_future_trajectory,  # [batch, agent, timepoint, 2]
            normalized_future_trajectory,  # [batch, agent, timepoint, 2]
            modified_is_future_valid,  # [batch, agent, timepoint]
        )
        additional_costs = {}
        return result, additional_costs

    def set_num_agents(self, num_agents, num_past_points, num_future_timepoints):
        """
        Reset the graph structure to match the number of agents, timepoints
        :param num_agents:
        :param num_past_points:
        :param num_future_timepoints:
        :return:
        """
        if num_agents != self.current_num_agents:
            self.current_num_agents = num_agents
            if type(self.encoder) == nn.DataParallel:
                self.encoder.module.set_num_agents(num_agents, num_past_points)
                self.decoder.module.set_num_agents(num_agents, num_future_timepoints)
            else:
                self.encoder.set_num_agents(num_agents, num_past_points)
                self.decoder.set_num_agents(num_agents, num_future_timepoints)

            if self.use_discriminator:
                self.discriminator_encoder.set_num_agents(num_agents, num_past_points)
                if "learn_reward_model" in self.params.keys():
                    self.discriminator_future_encoder.set_num_agents(num_agents, 1)
                else:
                    self.discriminator_future_encoder.set_num_agents(num_agents, num_future_timepoints)

    def compute_generator_cost(
        self,
        past_trajectory,
        past_additional_inputs,
        agent_additional_inputs,
        predicted,
        expected,
        agent_type,
        is_valid,
        is_future_valid,
        timestamps,
        prediction_timestamp,
        semantic_labels,
        future_encoding,
        relevant_agents,
        label_weights,
        param,
        stats,
    ):
        """Compute generator training cost

        TODO(igor.gilitschenski): The lists below are incomplete but correct. They have to be completed in a later PR

        Parameters
        ----------
        past_trajectory
        past_additional_inputs
        agent_additional_inputs
        predicted
        expected : dict
            Dictioanry containing a 'trajectories' key which contains a tensor of the expected trajectories
            of shape (batch_size, num_agents, num_future_timesteps, 2). The trajectories are expected to be
            normalized.
        agent_type
        is_valid
        is_future_valid
        timestamps
        prediction_timestamp
        semantic_labels
        future_encoding
        relevant_agents
        label_weights
        param: dict
            Dictionary with command line arguments.
        stats: list
            List of stats for each sample.


        Returns
        -------
        g_cost : torch.Tensor
            Cost for each predicted scene / datapoint. Shape: (batch_size,)
        g_stats : dict
            Different datapoint statistics. The tensors therein are not detached
            allowing use as additional cost terms if desired. The statistics involve:
              * "agent_mon_fde": Minimum over all smples of per-agent FDEs which
                can be nan for non-relevant agents. Shape (batch_size x num_agents)
              * "agent_mon_fdes_partial": Dict of minimum FDEs for different horizons
                 indexed by horizon. Can be nan for non-relevant agents and not observed
                 horizons. Shape (batch_size x num_agents)
        """
        # FIX(igor.gilitschenski): Visibility infomration seems to be encoded twice, in past_trajectory[:,:,:,2] and in
        # is_valid. Is this intended?
        predicted_trajectories_scene = predicted["trajectories"]
        expected_trajectories_scene = expected["trajectories"]
        skip_visualization = False
        if "skip_visualization" in param:
            skip_visualization = param["skip_visualization"]
        additional_param = {"skip_visualization": skip_visualization}

        _, _, num_pred_timesteps, _, num_samples = predicted_trajectories_scene.shape

        additional_stats = {}
        instance_l2_error = 0.0
        robust_error = 0.0
        weighted_validity = (is_future_valid > 0).float()
        weighted_validity = weighted_validity * relevant_agents[..., None]

        # ADE/FDE Errors
        instance_ade_error = []
        instance_fde_error = []
        sqr_error = []
        agent_sqr_error = []
        agent_visibility = []
        err_horizons_timepoints = param["err_horizons_timepoints"]
        fde_errs = {}
        fde_errs_full = {}
        ade_errs = {}
        MoN_fde_errs = {}
        for fde_point in err_horizons_timepoints:
            fde_errs[fde_point] = []
            fde_errs_full[fde_point] = []
            ade_errs[fde_point] = []
            MoN_fde_errs[fde_point] = []

        report_agent_type_metrics = param["report_agent_type_metrics"]
        fde_errs_type = {}
        ade_errs_type = {}
        MoN_fde_errs_type = {}
        MoN_ade_errs_type = {}
        if report_agent_type_metrics:
            assert "agent_types" in param, "Agent types need to be provided if report_agent_type_metrics is set."
            for agent_type_id in param["agent_types"]:
                agent_type_name = AGENT_TYPE_NAME_MAP[agent_type_id]
                fde_errs_type[agent_type_name] = {}
                ade_errs_type[agent_type_name] = {}
                MoN_fde_errs_type[agent_type_name] = {}
                MoN_ade_errs_type[agent_type_name] = {}

                for fde_point in err_horizons_timepoints:
                    fde_errs_type[agent_type_name][fde_point] = []
                    ade_errs_type[agent_type_name][fde_point] = []
                    MoN_fde_errs_type[agent_type_name][fde_point] = []
                    MoN_ade_errs_type[agent_type_name][fde_point] = []

        # Add additional callbacks to compute additional metrics and update variables.
        for cb in self.additional_model_callbacks:
            num_samples = cb.update_model_stats(
                param,
                stats,
                num_samples,
                num_pred_timesteps,
                predicted_trajectories_scene,
                expected_trajectories_scene,
                timestamps,
                prediction_timestamp,
                relevant_agents,
                is_future_valid,
                err_horizons_timepoints,
                agent_type,
            )

        # Report metrics per sample.
        report_sample_metrics = param["report_sample_metrics"]
        if report_sample_metrics:
            ade_errs_sample = {}
            for i in range(num_samples):
                ade_errs_sample[i] = []

        # Compute error for each sample in predicted_trajectories, with shape
        # [batch_size, num_agents, pred_seq, traj_dim, num_samples].
        for i in range(num_samples):
            displ_errors = displacement_errors(
                predicted_trajectories_scene[..., 0:2, i],
                expected_trajectories_scene[:, :, :, 0:2],
                timestamps[:, -num_pred_timesteps:],
                prediction_timestamp,
                relevant_agents,
                is_future_valid,
                err_horizons_timepoints,
                param["miss_thresholds"],
                param,
                relevant_agent_types=param["relevant_agent_types"],
                agent_types=agent_type,
            )
            sqr_error.append(displ_errors["square_error"])
            # Get raw square error without any operations (i.e. averaging, summing).
            agent_sqr_error.append(displ_errors["agent_square_error_raw"])
            agent_visibility.append(displ_errors["agent_visibility"])
            robust_error += displ_errors["robust_error"]
            instance_l2_error += displ_errors["square_error"]
            instance_fde_error.append(displ_errors["individual_fde_err"])
            instance_ade_error.append(displ_errors["individual_ade_err"])

            for fde_point in err_horizons_timepoints:
                fde_errs[fde_point].append(displ_errors["agent_fdes_partial"][fde_point].view(-1))
                fde_errs_full[fde_point].append(displ_errors["agent_fdes"][fde_point].view(-1))
                ade_errs[fde_point].append(displ_errors["agent_ades_partial"][fde_point].view(-1))
                # Note: this assumes MoN is jointly over the trajectories.
                if i == 0:
                    MoN_fde_errs[fde_point] = fde_errs[fde_point][-1]
                else:
                    MoN_fde_errs[fde_point] = torch.min(MoN_fde_errs[fde_point], fde_errs[fde_point][-1])

            # Update agent ade and fde errors per agent type.
            if report_agent_type_metrics:
                for agent_type_id in param["agent_types"]:
                    agent_type_name = AGENT_TYPE_NAME_MAP[agent_type_id]
                    for fde_point in err_horizons_timepoints:
                        fde_errs_type[agent_type_name][fde_point].append(
                            displ_errors["agent_fdes_partial_{}".format(agent_type_name)][fde_point]
                        )
                        ade_errs_type[agent_type_name][fde_point].append(
                            displ_errors["agent_ades_partial_{}".format(agent_type_name)][fde_point]
                        )

            # Update agent ade errors per sample.
            if report_sample_metrics:
                ade_errs_sample[i] += displ_errors["individual_ade_err"].tolist()

            if i == 0:
                agent_fdes = [displ_errors["agent_fde"]]
                agent_ades = [displ_errors["agent_ade"]]

                agent_fdes_partial = {}
                for key in displ_errors["agent_fdes_partial"]:
                    agent_fdes_partial[key] = [displ_errors["agent_fdes_partial"][key]]

                MoN_error = displ_errors["mon_individual_error"]
                MoN_ade_error = displ_errors["individual_ade_err"]
                MoN_fde_error = displ_errors["individual_fde_err"]
            else:
                agent_fdes.append(displ_errors["agent_fde"])
                agent_ades.append(displ_errors["agent_ade"])

                for key in displ_errors["agent_fdes_partial"]:
                    agent_fdes_partial[key].append(displ_errors["agent_fdes_partial"][key])
                MoN_error = torch.min(MoN_error, displ_errors["mon_individual_error"])  # Minimum over N (MoN) loss
                MoN_ade_error = torch.min(MoN_ade_error, displ_errors["individual_ade_err"])
                MoN_fde_error = torch.min(MoN_fde_error, displ_errors["individual_fde_err"])

        # Compute agent wise ade and fde, which could include NaNs, with shape (batch_size x num_agents).
        agent_mon_fde = torch.stack(agent_fdes).min(0)[0]
        agent_mon_ade = torch.stack(agent_ades).min(0)[0]
        additional_stats["agent_mon_fde"] = agent_mon_fde
        additional_stats["agent_mon_ade"] = agent_mon_ade

        # Compute marginal ade and fde averaged over each individual agent, ignoring NaNs.
        agent_mon_fde_no_nan = agent_mon_fde[~agent_mon_fde.isnan()]
        agent_mon_ade_no_nan = agent_mon_ade[~agent_mon_ade.isnan()]
        additional_stats["MoN_fde_marginal"] = agent_mon_fde_no_nan.mean(-1)
        additional_stats["MoN_ade_marginal"] = agent_mon_ade_no_nan.mean(-1)

        additional_stats["agent_mon_fdes_partial"] = {}
        for key, val in agent_fdes_partial.items():
            additional_stats["agent_mon_fdes_partial"][key] = torch.stack(val).min(0)[0]

        # Compute marginal agent errors per agent type.
        if report_agent_type_metrics:
            for agent_type_id in param["agent_types"]:
                agent_type_name = AGENT_TYPE_NAME_MAP[agent_type_id]
                additional_stats["MoN_fde_marginal_{}".format(agent_type_name)] = {}
                additional_stats["MoN_ade_marginal_{}".format(agent_type_name)] = {}
                for fde_point in err_horizons_timepoints:
                    # Get marginal errors, with shape (batch_size x num_agents).
                    MoN_fde_marginal_type_fde = torch.stack(fde_errs_type[agent_type_name][fde_point], -1).min(-1)[0]
                    additional_stats["MoN_fde_marginal/{}/{}_sec".format(agent_type_name, fde_point)] = [
                        x for x in MoN_fde_marginal_type_fde.view(-1).tolist() if not np.isnan(x)
                    ]
                    MoN_ade_marginal_type_fde = torch.stack(ade_errs_type[agent_type_name][fde_point], -1).min(-1)[0]
                    additional_stats["MoN_ade_marginal/{}/{}_sec".format(agent_type_name, fde_point)] = [
                        x for x in MoN_ade_marginal_type_fde.view(-1).tolist() if not np.isnan(x)
                    ]

        # Compute ade per sample.
        if report_sample_metrics:
            for i in range(num_samples):
                additional_stats["MoN_ade_sample/sample_{}".format(i)] = ade_errs_sample[i]

        instance_fde_error = torch.cat(instance_fde_error)
        instance_ade_error = torch.cat(instance_ade_error)
        instance_l2_error /= num_samples
        robust_error /= num_samples
        sqr_error = torch.stack(sqr_error, 1)  # [num_batch, num_samples].

        # Compute marginal MoN error.
        agent_sqr_error = torch.stack(agent_sqr_error, -1)  # [num_batch, num_agents, num_steps, num_samples].
        agent_visibility = torch.stack(agent_visibility, -1)  # [num_batch, num_agents, num_steps, num_samples].
        # Set invalid error to 0.
        agent_sqr_error = agent_sqr_error * agent_visibility

        # Aggregate square error over time steps.
        mon_agent_sqr_error_agg = agent_sqr_error.sum(-2)
        # TODO(cyrushx): Shall we take the average over steps (see below), since trajectories come with different valid steps.
        # mon_agent_sqr_error_agg = agent_sqr_error.sum(-2) / (agent_visibility.sum(-2) + 1e-10)

        # Compute MoN error over samples.
        mon_agent_sqr_error = mon_agent_sqr_error_agg.min(-1)[0]
        agent_visibility_validity = agent_visibility.sum(-2)[..., 0] > 0
        # Compute mean MoN over valid agents.
        MoN_error_marginal = mon_agent_sqr_error[agent_visibility_validity].mean()
        # Make fake batch size.
        MoN_error_marginal = MoN_error_marginal.unsqueeze(0).repeat(MoN_error.shape[0])

        for fde_point in err_horizons_timepoints:
            fde_errs[fde_point] = torch.stack(fde_errs[fde_point])
            fde_errs_full[fde_point] = torch.stack(fde_errs_full[fde_point])
            ade_errs[fde_point] = torch.stack(ade_errs[fde_point])

        expected["agent_type"] = agent_type
        additional_stats.update(
            self.model_encap.compute_extra_cost(
                predicted,
                expected,
                is_future_valid,
                weighted_validity,
                semantic_labels,
                future_encoding,
                timestamps,
                prediction_timestamp,
                label_weights,
                sqr_error,
                stats,
            )
        )

        if param["l2_error_only"]:
            data_cost = instance_l2_error * param["l2_term_coeff"]
            additional_stats["l2_cost"] = instance_l2_error * param["l2_term_coeff"]
            additional_stats["MoN_error"] = instance_l2_error.new_tensor(0.0)
        else:
            if param["use_marginal_error"]:
                data_cost = MoN_error_marginal * param["mon_term_coeff"]
            else:
                data_cost = MoN_error * param["mon_term_coeff"]
            if param["l2_term_coeff"] > 0:
                if param["raw_l2_for_mon"]:
                    data_cost += instance_l2_error * param["l2_term_coeff"]
                else:
                    data_cost += robust_error * param["l2_term_coeff"]
            if "semantic_cost" in additional_stats and not param["disable_label_weights"]:
                data_cost += additional_stats["semantic_cost"]
            if "acceleration_cost" in additional_stats:
                data_cost += additional_stats["acceleration_cost"]

            if param["raw_l2_for_mon"]:
                additional_stats["l2_cost"] = instance_l2_error * param["l2_term_coeff"]
            else:
                additional_stats["l2_cost"] = robust_error * param["l2_term_coeff"]

            additional_stats["MoN_cost"] = MoN_error * param["mon_term_coeff"]
            additional_stats["MoN_cost_marginal"] = MoN_error_marginal * param["mon_term_coeff"]

        # Add additional callbacks to compute additional data costs.
        for cb in self.additional_model_callbacks:
            data_cost = cb.update_data_cost(param, additional_stats, data_cost)

        # Add token generator cost if it exists.
        token_loss_name = "token_cost"
        if token_loss_name in additional_stats:
            data_cost += additional_stats[token_loss_name] * param["token_generator_coeff"]

        if param["use_discriminator"]:
            discrimination = 0.0
            if param["discriminator_term_coeff"] > 0 and "learn_reward_model" not in param.keys():
                if self.params["linear_discriminator"]:
                    disc, _ = self.discriminate_trajectory_samples(
                        past_trajectory,
                        past_additional_inputs,
                        agent_additional_inputs,
                        predicted_trajectories_scene,
                        expected_trajectories_scene,
                        agent_type,
                        is_valid,
                        is_future_valid,
                        timestamps,
                        prediction_timestamp,
                        relevant_agents,
                        additional_param=additional_param,
                    )
                    discrimination -= disc
                else:
                    disc, _ = self.discriminate_trajectory_samples(
                        past_trajectory,
                        past_additional_inputs,
                        agent_additional_inputs,
                        predicted_trajectories_scene,
                        expected_trajectories_scene,
                        agent_type,
                        is_valid,
                        is_future_valid,
                        timestamps,
                        prediction_timestamp,
                        relevant_agents,
                        additional_param=additional_param,
                    )
                    discrimination += self.bceloss(disc.clamp(0, 1), disc.new_ones(disc.shape))
            else:
                discrimination = torch.zeros_like(predicted_trajectories_scene[:, 0, 0, 0, :])
            # discrimination has batch and samples combined into one dimension, undo this and sum along the samples dimension.
            discrimination = (
                discrimination.view(predicted_trajectories_scene.shape[0], predicted_trajectories_scene.shape[4])
                .contiguous()
                .sum(1)
            )
            discrimination /= predicted_trajectories_scene.shape[4]

            g_cost = data_cost - discrimination * param["discriminator_term_coeff"]
        else:
            g_cost = data_cost

        if torch.isinf(g_cost).sum() > 0 or torch.isnan(g_cost).sum() > 0:
            import IPython

            IPython.embed(header="nan/inf in g_cost computation")

        additional_stats["l2_error"] = [x for x in instance_l2_error.view(-1).tolist() if not np.isnan(x)]
        additional_stats["robust_error"] = [x for x in robust_error.view(-1).tolist() if not np.isnan(x)]
        additional_stats["MoN_error"] = [x for x in MoN_error.view(-1).tolist() if not np.isnan(x)]
        additional_stats["data_cost"] = [x for x in data_cost.view(-1).tolist() if not np.isnan(x)]

        additional_stats["ade_error"] = instance_ade_error
        for fde_point in err_horizons_timepoints:
            additional_stats["fde_error/{}_sec".format(fde_point)] = [
                x for x in fde_errs[fde_point].view(-1).tolist() if not np.isnan(x)
            ]
            additional_stats["fde_error_full/{}_sec".format(fde_point)] = [
                x for x in fde_errs_full[fde_point].view(-1).tolist() if not np.isnan(x)
            ]
            additional_stats["ade_error/{}_sec".format(fde_point)] = [
                x for x in ade_errs[fde_point].view(-1).tolist() if not np.isnan(x)
            ]
            # TODO(igor.gilitschenski): Each of these entries contains still num_batch numbers. This does not make sene.
            additional_stats["MoN_fde_error/{}_sec".format(fde_point)] = [
                x for x in MoN_fde_errs[fde_point].view(-1).tolist() if not np.isnan(x)
            ]

        additional_stats["fde_error"] = [x for x in instance_fde_error.view(-1).tolist() if not np.isnan(x)]
        additional_stats["MoN_ade_error"] = [x for x in MoN_ade_error.view(-1).tolist() if not np.isnan(x)]
        additional_stats["MoN_fde_error"] = [x for x in MoN_fde_error.view(-1).tolist() if not np.isnan(x)]

        if param["use_discriminator"]:
            additional_stats["discrimination_cost"] = [
                x for x in (-discrimination).view(-1).tolist() if not np.isnan(x)
            ]
        return g_cost, additional_stats

    def get_semantic_keys(self) -> List[str]:
        return self.model_encap.get_semantic_keys()

    def compute_discriminator_cost(
        self,
        past_trajectory,
        additional_inputs,
        agent_additional_inputs,
        predicted_trajectories_scene,
        expected,
        agent_type,
        is_valid,
        is_future_valid,
        is_fake,
        timestamps,
        prediction_timestamp,
        relevant_agents,
        param,
    ):
        """
        Get discriminator cost for training.
        :param past_trajectory: [B x N_agents x timesteps x 3] tensor, past positions.
        :param additional_inputs: A dictionary of additional inputs. Each input should be of size [B x timesteps x..].
        :param predicted_trajectories_scene: [B x N_agents x timesteps x 2 x sample set size], the set of predicted trajectories.
        :param expected: [B x N_agents x timesteps x 3]
        :param agent_type: [B x N_agents x N_types]
        :param is_valid: [B x N_agents x past_timesteps] tensor, are the past positions valid?
        :param is_future_valid: [B x N_agents x timesteps] tensor, are the future positions valid?
        :param is_fake: boolean, are these trajectories fake or not?
        :param timestamps:
        :param prediction_timestamp:
        :param relevant_agents:
        :param param: d_cost a B-vector of the costs, additional stats dictionary.
        :return:
        """
        discrimination = 0.0
        skip_visualization = False
        if "skip_visualization" in param:
            skip_visualization = param["skip_visualization"]
        additional_param = {"skip_visualization": skip_visualization}

        if param["linear_discriminator"]:
            disc, _ = self.discriminate_trajectory_samples(
                past_trajectory[:, :, :, :2],
                additional_inputs,
                agent_additional_inputs,
                predicted_trajectories_scene,
                expected,
                agent_type,
                is_valid,
                is_future_valid,
                timestamps,
                prediction_timestamp,
                relevant_agents,
                additional_param=additional_param,
            )
            if is_fake:
                discrimination += disc
            else:
                discrimination -= disc
        else:
            disc, _ = self.discriminate_trajectory_samples(
                past_trajectory[:, :, :, :2],
                additional_inputs,
                agent_additional_inputs,
                predicted_trajectories_scene,
                expected,
                agent_type,
                is_valid,
                is_future_valid,
                timestamps,
                prediction_timestamp,
                relevant_agents,
                additional_param=additional_param,
            )
            if is_fake:
                discrimination += self.bceloss(disc.clamp(0, 1), disc.new_ones(disc.shape))
            else:
                discrimination += self.bceloss(disc.clamp(0, 1), disc.new_zeros(disc.shape))
        discrimination = (
            discrimination.view(predicted_trajectories_scene.shape[0], predicted_trajectories_scene.shape[4])
            .contiguous()
            .sum(1)
        )
        discrimination /= predicted_trajectories_scene.shape[4]
        d_cost = discrimination * param["discriminator_term_coeff"]
        additional_stats = {}
        if torch.isinf(d_cost).sum() > 0 or torch.isnan(d_cost).sum() > 0:
            import IPython

            IPython.embed(header="nan/inf in d_cost computation")

        return d_cost, additional_stats
