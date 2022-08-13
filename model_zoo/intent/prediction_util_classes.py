import functools
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from torch import nn

from intent.multiagents.cache_utils import split_reading_hash
from intent.multiagents.trainer_logging import TrainingLogger
from intent.multiagents.trainer_visualization import visualize_map_process
from model_zoo.intent.create_networks import MLP, CNNModel, create_mlp, load_cnn
from triceps.protobuf.prediction_dataset_cache import InMemoryCache
from triceps.protobuf.prediction_dataset_map_handlers import MapDataIndices, MapPointType


class AdditionalInputEncoder(ABC, nn.Module):
    """Interface class for additional inputs encoding."""

    logger: Optional[TrainingLogger]

    def __init__(self):
        super().__init__()
        self.logger = None

    @abstractmethod
    def forward(
        self,
        additional_input: torch.Tensor,
        trajectory_data: torch.Tensor,
        agent_additional_inputs: Optional[dict] = None,
        additional_params: Optional[dict] = None,
    ) -> tuple:
        pass

    def set_logger(self, logger: TrainingLogger):
        self.logger = logger


class ImageEncoder(AdditionalInputEncoder):
    def __init__(
        self,
        embed_size: int = 64,
        width: Optional[int] = None,
        height: Optional[int] = None,
        channels: int = 3,
        backbone_model: CNNModel = CNNModel.VGG11,
        pretrained: bool = False,
        frozen_params: int = 0,
        use_checkpoint: bool = False,
        fc_widths: list = None,
        params: Optional[dict] = None,
        nan_result_retries: int = 0,
    ) -> None:
        """Encodes images for prediction.

        Creates a CNN (with a fully-connected head) for computing image embeddings.

        Parameters
        ----------
        embed_size : int
            Size of the created image embedding.
        width: int, optional
            Width of the encoded image.
        height: int
            Height of the encoded image.
        backbone_model : CNNModel
            The CNN backbone model to use.
        pretrained : bool
            Use a pretrained CNN.
        frozen_params : int
            Describes how many layers of the backbone should have frozen parameters
            (i.e requires_grad = False).
        use_checkpoint : bool
            Switch indicating whether checkpointing should be used for sequential
            backbones. Activating this reduces memory demand while increasing
            computation time. See https://pytorch.org/docs/stable/checkpoint.html.
        fc_widths : list
            List containing the widths of the fully connected layer. If this is not given,
            the network simply uses one linear layer.
        params: Optional[dict]
            the parameters for the network/encoder.
        """
        super().__init__()
        self.out_features = embed_size
        self.back_bone, cnn_output_dim = load_cnn(
            backbone_model, height, width, channels, pretrained, frozen_params, params
        )
        if fc_widths is None:
            self.fc_layer = nn.Linear(cnn_output_dim, self.out_features)
        else:
            self.fc_layer = create_mlp(cnn_output_dim, fc_widths + [self.out_features], dropout_ratio=0.5)
        self.use_checkpoint = use_checkpoint
        self.nan_result_retries = nan_result_retries

    def forward(
        self,
        img: torch.Tensor,
        trajectory_data: torch.Tensor,
        agent_additional_inputs: Optional[dict] = None,
        additional_params: Optional[dict] = None,
    ) -> tuple:
        """Generates image embeddings

        Deprecation notice:
        Additionally to the parameters / format described below, this method supports 2 legacy modes
        which will be deprecated.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape (batch_size, num_images, 3, height, width)
        trajectory_data : torch.Tensor
            Unused.
        agent_additional_inputs : dict, optional
            Unused.

        Returns
        -------
        torch.Tensor
            Image embeddings of shape (batch_size, max_num_images, output_feature_dim)
        dict
            Unused additional costs dictionary.

        """
        batch_size = img.shape[0]
        result = img.new_zeros([batch_size, img.shape[1], self.out_features])

        # Shape (batch_size, max_images)
        actual_images_mask = torch.count_nonzero(img, dim=(-1, -2, -3)) > 0.0

        batch_idxs, image_idxs = torch.nonzero(actual_images_mask, as_tuple=True)

        if len(batch_idxs) > 0:
            nonzero_tensors = img[batch_idxs, image_idxs, :, :, :]

            if (
                self.training
                and self.use_checkpoint
                and isinstance(self.back_bone, torch.nn.modules.container.Sequential)
            ):
                nonzero_tensors = self.fc_layer(
                    torch.utils.checkpoint.checkpoint_sequential(self.back_bone, 2, nonzero_tensors)
                )
            else:
                original_input = nonzero_tensors.clone()

                # AdaptiveAvgPool2d fails and returns a nan result sometimes, when using multiprocess dataloaders
                # The effect is temporary, so we retry a few times until we get a non-nan result or give up

                def do_encoding(retries):
                    nonzero_tensors = self.fc_layer(self.back_bone(original_input.clone()))
                    if nonzero_tensors.isnan().any() and retries > 0:
                        logging.warning(f"ImageEncoder backbone generated NaN results. {retries} retries remaining...")
                        return do_encoding(retries=retries - 1)
                    else:
                        return nonzero_tensors

                nonzero_tensors = do_encoding(retries=self.nan_result_retries)
                assert not nonzero_tensors.isnan().any()

            assert not nonzero_tensors.isnan().any()
            result[batch_idxs, image_idxs, :] = nonzero_tensors

        additional_costs = {}
        return result, additional_costs


def create_cnn_model(layer_features, output_dim):
    layers = []
    layer_features = [3] + layer_features + [output_dim]
    for i, fts in enumerate(zip(layer_features[:-1], layer_features[1:])):
        f1, f2 = fts
        if i > 0:
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        layers.append(nn.Conv2d(f1, f2, 3, padding=1, padding_mode="replicate"))
    return nn.Sequential(*layers)


class MapPointMLPEncoder(AdditionalInputEncoder):
    def __init__(self, max_point_num=10, input_dim=3, embed_size=[32]):
        """
        Encodes map points using MLPs.
        :param embed_size:
        """
        super().__init__()
        self.max_point_num = max_point_num
        self.input_dim = input_dim

        self.embed_size = embed_size[0]
        self.out_features = embed_size[0]
        self.fc_layer1 = nn.Linear(self.input_dim * self.max_point_num, self.embed_size)
        self.fc_layer2 = nn.Linear(self.embed_size, self.out_features)
        self.relu = nn.ReLU()

    def forward(
        self, points, trajectory_data=None, agent_additional_inputs=None, additional_params: Optional[dict] = None
    ) -> tuple:
        """
        Encode map points into a embed vector.
        :param points: map points of shape [batch_size, num_map_elements, num_points_per_element, 3]
        :return: embed vector [batch_size, embed_size]
        """
        (batch_size, num_map_elements, num_points_per_element, _) = tuple(points.shape)
        # Push second dim to first dim, and merge last two dims
        inputs = points.view(-1, num_points_per_element * 3)

        outputs = self.relu(self.fc_layer1(inputs))
        outputs = self.relu(self.fc_layer2(outputs))
        outputs = outputs.view(batch_size, num_map_elements, -1)
        # Use a sum reducer to pool features from each map element.

        outputs = torch.max(outputs, axis=1, keepdim=True)[0]
        additional_costs = {}
        return outputs, additional_costs


class LocalProcessing(nn.Module):
    def __init__(self, map_dim, embedding_dim):
        super().__init__()
        self.node_processor = nn.Sequential(
            nn.Linear(map_dim + embedding_dim + 2, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.edge_processor = nn.Sequential(nn.Linear(map_dim * 2 + embedding_dim * 2, embedding_dim), nn.ReLU())
        self.edge_attn_processor = nn.Sequential(nn.Linear(map_dim * 2 + embedding_dim * 2, embedding_dim), nn.ReLU())
        # This ModuleDict stores the modules used for additional tasks. These are registering and will be optimized.
        self.additional_tasks = torch.nn.ModuleDict()
        self.additional_tasks["position"] = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 2)
        )
        self.additional_tasks["tangent"] = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 2)
        )
        self.additional_tasks["normal"] = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 2)
        )
        self.future_position_net = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 2)
        )
        self.additional_tasks[("future_position_1")] = self.future_position_net
        self.additional_tasks[("future_position_2")] = self.future_position_net

        # This dictionary is used to store additional parameters used for additional task computation
        # TODO(guy.rosman): Consider refactoring and using functools.partial and a single ModuleDict
        # TODO  with a specific class interface.
        self.additional_tasks_inputs = {}
        self.additional_tasks_inputs[("future_position_1")] = 1.0
        self.additional_tasks_inputs[("future_position_2")] = 2.0
        self.nonlinear = nn.ReLU()

    def forward(self, neighborhood, neighborhood_embedding, representative_id):
        """Example local processing for a Voronoi cell

        Args:
            neighborhood (Tensor): The coordinates of the neighborhood, N_node x 6.
            neighborhood_embedding (Tensor): The embedding of the neighborhood, N_node x embedding_size.
            representative_id (int): The index of the representative point.


        Returns:
            [type]: [description]
        """
        additional_stats = {}
        num_nodes = neighborhood.shape[0]

        result_map = neighborhood
        result_embedding = neighborhood_embedding
        # This is used to normalize input sizes.
        # TODO(guy.rosman): replace with a more general approach.
        map_scale = 10
        for _ in range(2):
            results_aggregate = result_embedding
            edge_inputs = []
            for i in range(num_nodes):
                embed1 = result_embedding[i, :].unsqueeze(0).repeat([num_nodes, 1])
                map1 = result_map[i, :].unsqueeze(0).repeat([num_nodes, 1]) / map_scale
                edge_input = torch.cat([map1, result_map, embed1, result_embedding], 1)
                edge_inputs.append(edge_input)
            edge_attn_weights = (self.edge_attn_processor(torch.stack(edge_inputs)) ** 2 + 1e-2) ** (-1)
            edge_processed = self.edge_processor(torch.stack(edge_inputs))
            results_aggregate2 = results_aggregate + (edge_processed * edge_attn_weights).sum(
                dim=1
            ) / edge_attn_weights.sum(dim=1)
            result_embedding = self.node_processor(
                torch.cat([result_map, result_map[:, :2] - result_map[representative_id, :2], results_aggregate2], 1)
            )
        for key in self.additional_tasks:
            if key in self.additional_tasks_inputs:
                input_value = torch.cat([result_embedding, result_embedding[:, :1] * 0 + 1.0], 1)
                additional_stats[key] = self.additional_tasks[key](input_value)
            else:
                additional_stats[key] = self.additional_tasks[key](result_embedding)

        return (
            result_map[representative_id : representative_id + 1, :],
            result_embedding[representative_id : representative_id + 1, :],
            additional_stats,
        )


def accumulate(dct, key, addition):
    if key not in dct:
        dct[key] = 0
    dct[key] += addition


def compute_distance_matrix(nearest_coord):
    """Compute a distance matrix

    Parameters
    ----------
    nearest_coord: tensor
      A tensor of size batch_size x num_coords x dim, capturing the coordinates.

    Returns
    -------
    distance_matrix: tensor
      A tensor of size batch_size x num_coords x num_coords, capturing the distance matrix per batch element.

    """
    num_map_elements = nearest_coord.shape[1]
    A2 = (nearest_coord**2).sum(2).unsqueeze(2).repeat(1, 1, num_map_elements)
    B2 = A2.transpose(1, 2)
    A_ = nearest_coord.unsqueeze(2).repeat(1, 1, num_map_elements, 1)
    AB = (A_ * (A_.transpose(1, 2))).sum(3)
    distance_matrix = A2 + B2 - 2 * AB
    return distance_matrix


class MapPointGNNEncoder(AdditionalInputEncoder):
    def __init__(
        self,
        map_input_dim,
        traj_input_dim,
        embed_size,
        params,
        device="cpu",
    ):
        """
        Encodes map points using GNNs.
        Implementation based on https://arxiv.org/pdf/1806.01261.pdf
        :param map_input_dim: dimension of map points.super().__init__()
        :param traj_input_dim: dimension of trajectory inputs.
        :param embed_size: size of node and edge embeddings.
        :param params
        :param device: device of attribute initializations.
        """
        super().__init__()
        self.map_input_dim = map_input_dim
        self.traj_input_dim = traj_input_dim
        self.embed_size = embed_size
        self.out_features = embed_size
        self.relu = nn.ReLU()
        self.params = params
        self.device = device
        self.num_nearest_distances = self.params["map_num_nearest_distances"]
        self.map_num_relevant_points = self.params["map_num_relevant_points"]
        self.map_input_normalization_scales = self.params["map_input_normalization_scales"]
        if self.params["map_polyline_feature_degree"] > 0:
            self.map_input_normalization_scales += [1] * self.params["map_polyline_feature_degree"] * 2
            self.map_input_dim += self.params["map_polyline_feature_degree"] * 2

        # Use attention to pool features from each map element, similar to VectorNet.
        self.map_attention_type = params["map_attention_type"]

        # Set up MLPs to encoder coordinates.
        self.traj_encoder = nn.Linear(self.traj_input_dim, self.embed_size)

        # Set up graph.
        # TODO(guy.rosman): consider not storing data in the class but move w/ functions.
        self.node_states = OrderedDict()
        self.edge_states = OrderedDict()
        self.map_data = None
        self.traj_data = None

        self.map_encoding_cache = InMemoryCache()

        # Set up graph attributes.
        self.edge_processor = nn.Linear(self.embed_size * 3, self.embed_size)
        self.node_processor = nn.Linear(self.embed_size * 2, self.embed_size)
        self.edge_input_encoder = nn.Linear(3, self.embed_size)
        self.gnn_layer = params["map_gnn_layer"]
        self.num_map_elements = 0
        self.num_agent_positions = 0
        self.local_processing = LocalProcessing(self.map_input_dim + 1, self.embed_size)
        self.call_cnt = 0
        NUM_MAP_ELEMENT_TYPES = len(MapPointType)
        self.map_type_values = list(range(NUM_MAP_ELEMENT_TYPES))

        self.num_max_map_elements = params["map_elements_max"]
        self.map_id_type = params["map_id_type"]
        if self.map_id_type == "integer":
            self.map_id_size = 1
        elif self.map_id_type == "binary":
            self.map_id_size = int(np.ceil(np.log2(self.num_max_map_elements + 1)))
        elif self.map_id_type == "onehot":
            self.map_id_size = self.num_max_map_elements
        else:
            raise Exception("Bad map id type.")

        self.map_input_normalization_scales += [1] * self.map_id_size
        self.map_expanded_dim = self.map_input_dim - 1 + len(self.map_type_values) + self.map_id_size
        self.embed_size2 = self.embed_size - min(self.num_nearest_distances, self.map_num_relevant_points)
        self.point_encoder = nn.Linear(self.map_expanded_dim, self.embed_size2)

        # This ModuleDict stores the modules used for additional tasks. These are registering and will be optimized.
        self.additional_tasks = torch.nn.ModuleDict()
        self.additional_tasks["position"] = nn.Sequential(
            nn.Linear(self.embed_size2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, 2)
        )
        self.additional_tasks["tangent"] = nn.Sequential(
            nn.Linear(self.embed_size2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, 2)
        )
        self.additional_tasks["type_tensor"] = nn.Sequential(
            nn.Linear(self.embed_size2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, 1)
        )
        self.additional_tasks["normal"] = nn.Sequential(
            nn.Linear(self.embed_size2, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, 2)
        )
        if params["map_polyline_feature_degree"] > 0:
            self.additional_tasks["poly"] = nn.Sequential(
                nn.Linear(self.embed_size2, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, params["map_polyline_feature_degree"] * 2),
            )

        self.future_position_net = nn.Sequential(
            nn.Linear(self.embed_size2 + 1, self.embed_size), nn.ReLU(), nn.Linear(self.embed_size, 2)
        )
        self.additional_tasks[("future_position_1")] = self.future_position_net
        self.additional_tasks[("future_position_2")] = self.future_position_net

        # This dictionary is used to store additional parameters used for additional task computation
        # TODO(guy.rosman): Consider refactoring and using functools.partial and a single ModuleDict
        # TODO  with a specific class interface.
        self.additional_tasks_inputs = {}
        self.additional_tasks_inputs[("future_position_1")] = 1.0
        self.additional_tasks_inputs[("future_position_2")] = 2.0

        # Use attention to pool features from each map element, similar to VectorNet.
        if self.map_attention_type in ("element", "point"):
            self.traj_attention_q = nn.Linear(2, self.embed_size)
            self.map_attention_q = nn.Linear(self.embed_size2, self.embed_size)
            self.map_attention_k = nn.Linear(self.embed_size2, self.embed_size)
            self.map_attention_v = nn.Linear(self.embed_size2, self.embed_size)
            self.map_attention_q_p = nn.Linear(self.embed_size, 1)
        elif self.map_attention_type != "none":
            raise Exception("Bad map attention type.")

    def message_passing(self):
        """
        Perform message passing on the GNN.
        :return:
        """
        profiling = False
        # First update edge states.
        new_edge_states = {}
        edge_inputs = []
        t0 = time.perf_counter()
        for e in self.edge_states:
            i, j = e
            node_i = self.node_states[i]
            node_j = self.node_states[j]
            edge_state_new = torch.cat([self.edge_states[e], node_i, node_j], -1).unsqueeze(0)
            edge_inputs.append(edge_state_new)
        edge_inputs = torch.cat(edge_inputs, 0)
        processed_edges = self.edge_processor(edge_inputs)
        processed_edges = self.relu(processed_edges)
        processed_edges = torch.split(processed_edges, 1, dim=0)
        for i, j in self.edge_states:
            new_edge_states[e] = processed_edges[i].squeeze(0)
        self.edge_states = new_edge_states
        t1 = time.perf_counter()
        if profiling:
            print("message_passing: edge processing: {} s".format(t1 - t0))
        # Next update node states.
        new_node_states = {}
        t2 = time.perf_counter()
        node_inputs = []
        for v in self.node_states:
            edge_aggregation = []
            for w in self.node_states:
                if v != w:
                    edge_aggregation.append(self.edge_states[(w, v)])
            edge_aggregation = torch.stack(edge_aggregation, 1)
            edge_aggregation = torch.sum(edge_aggregation, 1)
            node_state_new = torch.cat((self.node_states[v], edge_aggregation), -1)
            node_inputs.append(node_state_new.unsqueeze(0))
        node_inputs = torch.cat(node_inputs, 0)

        node_states_new = self.node_processor(node_inputs)
        node_states_new = self.relu(node_states_new)
        for v_i, v in enumerate(self.node_states):
            new_node_states[v] = node_states_new[v_i, ...]
        self.node_states = new_node_states
        t3 = time.perf_counter()
        if profiling:
            print("message_passing: node processing: {} s".format(t3 - t2))
            print("message_passing: total {} s".format(t3 - t0))
        # TODO(cyrushx): do we need global states?
        self.edge_states = new_edge_states
        self.node_states = new_node_states

    @staticmethod
    def get_top_k_map_embedding(traj_points, map_points, map_embedding, k=1):
        """
        Get top k relevant map embeddings according to minimum distances.
        Parameters
        ----------
        traj_points: [tensor] with shape [num_traj_points, 2], agent positions.
        map_points: [tensor] with shape [num_map_points, 2], map positions.
        map_embedding: [tensor], embedding of each map position.
        k: int, number of embeddings to pool.

        Returns
        -------
        top_k_map_embedding: [tensor], top k map embedding.
        """
        # Compute pairwise distance between agent positions and map positions.
        dists = ((traj_points.unsqueeze(1) - map_points.unsqueeze(0)) ** 2).sum(-1)
        candidate_indices = torch.argsort(dists, 1)[:, :k]
        top_k_map_embedding = map_embedding[candidate_indices, :]
        return top_k_map_embedding

    def get_trajectory_map_embeddings(
        self,
        map_data: torch.Tensor,
        map_embeddings: torch.Tensor,
        map_validity: torch.Tensor,
        num_agents: int,
        additional_costs: dict = None,
        trajectory: torch.Tensor = None,
        agent_index: int = None,
    ):
        """
        Get top k map embeddings from each lane relevant to a trajectory.

        Parameters
        ----------
        map_data: torch.Tensor
            map position with shape [num_batch * num_agents, num_max_points, num_map_features],
            where map features (dim=9) include (x, y, validity, point type, sin(theta), cos(theta), cos(theta),
            -sin(theta), point id).
        map_embeddings: torch.Tensor
            map embedding tensor with shape [num_batch * num_agents, num_max_points, embed_size].
        map_validity: torch.Tensor
            map validity tensor with shape [num_batch * num_agents, num_max_points, 1]
        num_agents: int
            number of agents.
        additional_costs: dict
            dict of additional costs to collect.
        trajectory: torch.Tensor
            trajectory tensor with shape [num_batch, num_step, 2].
        agent_index: int
            index of agent to get map embedding. If not provided, get embedding for all agents.

        Returns
        -------
        trajectory_map_embedding: torch.Tensor
            map embedding associated with each point in the given trajectory, with shape [num_batch, num_step,
            map_embed_size].
        """
        # Get relevant shapes. Note that the first dim of map_data includes both batch size and agent num.
        (batch_agents_size, num_max_points, _) = tuple(map_data.shape)

        # If a specific agent index is provided, we need to supply the map data specific to that agent.
        if agent_index is not None:
            batch_size = int(batch_agents_size / num_agents)
            map_data = map_data.view(batch_size, num_agents, num_max_points, -1)[:, agent_index]
            map_embeddings = map_embeddings.view(batch_size, num_agents, num_max_points, -1)[:, agent_index]
            map_validity = map_validity.view(batch_size, num_agents, num_max_points, 1)[:, agent_index]
            batch_agents_size = batch_size

        num_step = trajectory.shape[1]
        trajectory_positions = trajectory[..., :2]
        map_validity = map_validity[..., 0]
        top_k = self.params["map_num_relevant_points"]
        map_id = map_data[..., -1]

        # Compute distances between agent positions and map positions.
        map_positions = map_data[..., :2]
        dists = ((trajectory_positions.unsqueeze(2) - map_positions.unsqueeze(1)) ** 2).sum(-1)
        dists = dists.view(batch_agents_size, num_step, num_max_points)

        # Obtain indices of top k point in each lane to each trajectory point.
        map_validity_agent_steps = map_validity.unsqueeze(1).repeat(1, num_step, 1)
        invalid_dist_penalty = (1.0 - map_validity_agent_steps) * 1e10
        # Add large numbers to invalid points.
        dists = dists + invalid_dist_penalty
        # Sort points based on distance to trajectory.
        # [num_batch_agent, num_step, num_map_points).
        closest_dist_indices = torch.argsort(dists, -1)

        # Reorder the indices so that we can closest k elements from each unique map element.
        # TODO(cyrushx): Find a more efficient way to do this.
        closest_dist_indices_per_element = torch.zeros_like(closest_dist_indices)
        for b in range(batch_agents_size):
            map_id_b = map_id[b]
            unique_map_ids = torch.unique(map_id_b)
            pointer = 0

            for i in unique_map_ids:
                map_id_mask = map_id_b == i
                map_id_mask_indices = torch.masked_select(closest_dist_indices[b], map_id_mask).view(
                    closest_dist_indices[b].shape[0], -1
                )
                map_id_mask_indices_size = min(map_id_mask_indices.shape[-1], 2)
                closest_dist_indices_per_element[
                    b, :, pointer : pointer + map_id_mask_indices_size
                ] = map_id_mask_indices[:, :map_id_mask_indices_size]
                pointer += map_id_mask_indices_size

            closest_dist_indices_per_element_left = []
            for t in range(num_step):
                closest_dist_indices_per_element_left_t = closest_dist_indices[b, t][
                    ~closest_dist_indices[b, t].unsqueeze(1).eq(closest_dist_indices_per_element[b, t, :pointer]).any(1)
                ]
                closest_dist_indices_per_element_left.append(closest_dist_indices_per_element_left_t)
            closest_dist_indices_per_element_left = torch.stack(closest_dist_indices_per_element_left, 0)
            closest_dist_indices_per_element[b, :, pointer:] = closest_dist_indices_per_element_left

        # [num_batch_element, num_step, top_k_points].
        closest_dist_indices_per_element = closest_dist_indices_per_element[..., :top_k]

        # Compute mask based on closest indices.
        # [num_batch_element, num_step, num_max_points].
        top_k_mask = torch.nn.functional.one_hot(closest_dist_indices_per_element, num_classes=num_max_points).sum(-2)

        # Update embedding tensor and validity tensor to have the same shape.
        map_embeddings_agent_steps = map_embeddings.unsqueeze(1).repeat(1, num_step, 1, 1)
        map_validity_agent_steps_embed = map_validity_agent_steps.unsqueeze(-1).repeat(
            1, 1, 1, map_embeddings.shape[-1]
        )
        # Set invalid embeddings to 0.
        # [num_batch_element, num_step, num_max_points, map_embed_size].
        map_embeddings_agent_steps_valid = map_embeddings_agent_steps * map_validity_agent_steps_embed

        # Obtain top embeddings. This assumes that each mask has exactly top_k True values.
        top_k_map_embeddings_flat = map_embeddings_agent_steps_valid[top_k_mask.bool()]
        top_k_map_embeddings = top_k_map_embeddings_flat.view(batch_agents_size, num_step, top_k, -1)
        top_k_dists_flat = dists[top_k_mask.bool()]
        top_k_dists = top_k_dists_flat.view(batch_agents_size, num_step, top_k)

        # Get information of the points that are closest to the last observed point.
        nearest_x = map_positions[..., 0][top_k_mask[:, -1, :].bool()].view(top_k_mask.shape[0], -1)
        nearest_y = map_positions[..., 1][top_k_mask[:, -1, :].bool()].view(top_k_mask.shape[0], -1)
        nearest_type = map_data[..., MapDataIndices.MAP_IDX_TYPE][top_k_mask[:, -1, :].bool()].view(
            top_k_mask.shape[0], -1
        )

        # Compute distance among each selected point.
        # TODO(cyrushx): Add point id when computing the distance.
        # [num_batch_element, top_k_points, point_feature_size].
        nearest_coord = torch.cat([nearest_x.unsqueeze(2), nearest_y.unsqueeze(2), nearest_type.unsqueeze(2)], 2)
        nearest_coord_pairwise_distance = compute_distance_matrix(nearest_coord)
        # Add large distance to the same point.
        diag = torch.eye(top_k).to(nearest_coord_pairwise_distance.device).unsqueeze(0).repeat(batch_agents_size, 1, 1)
        nearest_coord_pairwise_distance = nearest_coord_pairwise_distance + diag * 10

        # Create, as features, the distance from each element nearest point to the other element nearest points.
        # Allows the network to know if there are duplicate elements, and suppress their contribution (in the spirit of
        # non-maxima suppression).
        nearest_elements_dists, _ = nearest_coord_pairwise_distance.sort(2)
        nearest_elements_dists = nearest_elements_dists.clamp(0, 10)
        assert not (torch.isnan(nearest_elements_dists).any() or torch.isinf(nearest_elements_dists).any())
        nearest_elements_dists = nearest_elements_dists[:, :, : self.num_nearest_distances] / 1000
        top_k_map_embeddings_augmented = torch.cat(
            [top_k_map_embeddings, nearest_elements_dists.unsqueeze(1).repeat(1, num_step, 1, 1)], 3
        )

        # Add distance kernel to map embeddings.
        # TODO(cyruxh): Add attention / GNN message passing.
        # [num_agent_batch, num_steps, embed_size]
        dist_kernel = 1.0 / (top_k_dists - torch.min(top_k_dists, -1, keepdim=True)[0] + 1.0)
        trajectory_map_embedding = (dist_kernel.unsqueeze(-1) * top_k_map_embeddings_augmented).sum(2)
        # # Perform max pooling over map embeddings.
        # trajectory_map_embedding = top_k_map_embeddings.max(2)[0]

        # Obtain additional stats.
        for b in range(batch_agents_size):
            # Add additional costs info if given.
            if additional_costs is not None:
                point_count_b = map_validity[b].sum().detach().cpu().numpy()
                accumulate(additional_costs, "point_count", point_count_b)
        # [batch_size, num_past_steps, map_dim]
        return trajectory_map_embedding

    def get_trajectory_map_embeddings_element_attention(
        self,
        map_data: torch.Tensor,
        map_embeddings: torch.Tensor,
        map_validity: torch.Tensor,
        num_agents: int,
        additional_costs: dict = None,
        trajectory: torch.Tensor = None,
        agent_index: int = None,
    ):
        """
        Get map embeddings given a trajectory using self-attention over each map element.

        Parameters
        ----------
        map_data: torch.Tensor
            map position with shape [num_batch * num_agents, num_max_points, num_map_features],
            where map features (dim=9) include (x, y, validity, point type, sin(theta), cos(theta), cos(theta),
            -sin(theta), point id).
        map_embeddings: torch.Tensor
            map embedding tensor with shape [num_batch * num_agents, num_max_points, embed_size].
        map_validity: torch.Tensor
            map validity tensor with shape [num_batch * num_agents, num_max_points, 1]
        num_agents: int
            number of agents.
        additional_costs: dict
            dict of additional costs to collect.
        trajectory: torch.Tensor
            trajectory tensor with shape [num_batch, num_step, 2].
        agent_index: int
            index of agent to get map embedding. If not provided, get embedding for all agents.

        Returns
        -------
        trajectory_map_embedding: torch.Tensor
            map embedding associated with each point in the given trajectory, with shape [num_batch, num_step,
            map_embed_size].
        """
        # Get relevant shapes. Note that the first dim of map_data includes both batch size and agent num.
        (batch_agents_size, num_max_points, _) = tuple(map_data.shape)

        # If a specific agent index is provided, we need to supply the map data specific to that agent.
        if agent_index is not None:
            batch_size = int(batch_agents_size / num_agents)
            map_data = map_data.view(batch_size, num_agents, num_max_points, -1)[:, agent_index]
            map_embeddings = map_embeddings.view(batch_size, num_agents, num_max_points, -1)[:, agent_index]
            map_validity = map_validity.view(batch_size, num_agents, num_max_points, 1)[:, agent_index]
            batch_agents_size = batch_size

        num_step = trajectory.shape[1]
        map_validity = map_validity[..., 0]
        map_id = map_data[..., -1]

        # Get embeddings from each unique element through max-pool.
        map_embeddings_attention = []
        for b in range(batch_agents_size):
            map_validity_b = map_validity[b] > 0
            map_embeddings_valid_b = map_embeddings[b][map_validity_b]
            map_id_valid_b = map_id[b][map_validity_b]
            unique_map_ids = torch.unique(map_id_valid_b)
            map_embeddings_max_pool_b = []
            for i in unique_map_ids:
                map_id_mask = map_id_valid_b == i
                map_id_embeddings = map_embeddings_valid_b[map_id_mask]
                map_id_embeddings_max_pool = map_id_embeddings.max(0)[0]
                map_embeddings_max_pool_b.append(map_id_embeddings_max_pool)

            # [num_unique_map_elements, map_embed_size]
            map_embeddings_max_pool_b = torch.stack(map_embeddings_max_pool_b, 0)
            # [num_step, num_unique_map_elements, map_embed_size].
            map_embeddings_max_pool_b = map_embeddings_max_pool_b[None].repeat(num_step, 1, 1)

            # Attend each map element.
            trajectory_positions = trajectory[b, :, :2]
            # [num_past_steps, query_dim].
            traj_embedding_q = self.traj_attention_q(trajectory_positions)

            # [num_past_steps, num_map_elements, query_dim]
            map_embeddings_k = self.map_attention_k(map_embeddings_max_pool_b)
            map_embeddings_v = self.map_attention_v(map_embeddings_max_pool_b)

            attention_score = torch.matmul(traj_embedding_q.unsqueeze(1), map_embeddings_k.transpose(1, 2))
            attention_score_p = nn.Softmax(dim=-1)(attention_score)

            map_embeddings_attention_b = torch.matmul(attention_score_p, map_embeddings_v)
            # [num_past_steps, map_dim]
            map_embeddings_attention_b = map_embeddings_attention_b.squeeze(1)
            map_embeddings_attention.append(map_embeddings_attention_b)

            # Add additional costs info if given.
            if additional_costs is not None:
                point_count_b = map_validity[b].sum().detach().cpu().numpy()
                accumulate(additional_costs, "point_count", point_count_b)

        # [batch_agent_size, num_past_steps, map_dim].
        map_embeddings_attention = torch.stack(map_embeddings_attention, 0)
        return map_embeddings_attention

    def get_trajectory_map_embeddings_point_attention(
        self,
        map_data: torch.Tensor,
        map_embeddings: torch.Tensor,
        map_validity: torch.Tensor,
        num_agents: int,
        additional_costs: dict = None,
        trajectory: torch.Tensor = None,
        agent_index: int = None,
    ):
        """
        Get map embeddings given a trajectory using self-attention over each map point.

        Parameters
        ----------
        map_data: torch.Tensor
            map position with shape [num_batch * num_agents, num_max_points, num_map_features],
            where map features (dim=9) include (x, y, validity, point type, sin(theta), cos(theta), cos(theta),
            -sin(theta), point id).
        map_embeddings: torch.Tensor
            map embedding tensor with shape [num_batch * num_agents, num_max_points, embed_size].
        map_validity: torch.Tensor
            map validity tensor with shape [num_batch * num_agents, num_max_points, 1]
        num_agents: int
            number of agents.
        additional_costs: dict
            dict of additional costs to collect.
        trajectory: torch.Tensor
            trajectory tensor with shape [num_batch, num_step, 2].
        agent_index: int
            index of agent to get map embedding. If not provided, get embedding for all agents.

        Returns
        -------
        trajectory_map_embedding: torch.Tensor
            map embedding associated with each point in the given trajectory, with shape [num_batch, num_step,
            map_embed_size].
        """
        # Subsample map data.
        # TODO(cyrushx): Shall we do it in the dataloader?
        map_data = map_data[:, :: self.params["map_points_subsample_ratio"]]
        map_embeddings = map_embeddings[:, :: self.params["map_points_subsample_ratio"]]
        map_validity = map_validity[:, :: self.params["map_points_subsample_ratio"]]

        # Get relevant shapes. Note that the first dim of map_data includes both batch size and agent num.
        (batch_agents_size, num_max_points, _) = tuple(map_data.shape)

        # If a specific agent index is provided, we need to supply the map data specific to that agent.
        if agent_index is not None:
            batch_size = int(batch_agents_size / num_agents)
            map_embeddings = map_embeddings.view(batch_size, num_agents, num_max_points, -1)[:, agent_index]
            map_validity = map_validity.view(batch_size, num_agents, num_max_points, 1)[:, agent_index]

        num_step = trajectory.shape[1]
        map_validity = map_validity[..., 0]
        # Set invalid embeddings to 0.
        map_embeddings = map_embeddings * map_validity.unsqueeze(-1)

        trajectory_positions = trajectory[..., :2]
        # [batch_agent_size, num_past_steps, query_dim].
        traj_embedding_q = self.traj_attention_q(trajectory_positions)

        map_embeddings_k = self.map_attention_k(map_embeddings)
        map_embeddings_v = self.map_attention_v(map_embeddings)
        s_k = map_embeddings_k.shape
        # [batch_agent_size, num_past_steps, num_map_elements, embed_size]
        map_embeddings_k = map_embeddings_k[:, None].expand(s_k[0], num_step, s_k[1], s_k[2])
        s_v = map_embeddings_v.shape
        # [batch_agent_size, num_past_steps, num_map_elements, embed_size]
        map_embeddings_v = map_embeddings_v[:, None].expand(s_v[0], num_step, s_v[1], s_v[2])

        # TODO(cyrushx): Verify attention vs. maxpool.
        attention_score = torch.matmul(traj_embedding_q.unsqueeze(2), map_embeddings_k.transpose(2, 3))
        # Replace invalid score with a very small number.
        # TODO(cyrushx): Is there a better way to do it?
        attention_score = attention_score * map_validity.unsqueeze(1).unsqueeze(1)
        attention_score = attention_score + (1.0 - map_validity.unsqueeze(1).unsqueeze(1)) * (-10000)
        attention_score_p = nn.Softmax(dim=-1)(attention_score)

        map_embeddings_attention = torch.matmul(attention_score_p, map_embeddings_v)
        # [batch_agent_size, num_past_steps, map_dim].
        map_embeddings_attention = map_embeddings_attention.squeeze(2)

        # TODO(cyrushx): Move this outside of map embedding.
        for b in range(batch_agents_size):
            # Add additional costs info if given.
            if additional_costs is not None:
                point_count_b = map_validity[b].sum().detach().cpu().numpy()
                accumulate(additional_costs, "point_count", point_count_b)

        point_count = map_validity.sum().detach().cpu().numpy()
        accumulate(additional_costs, "point_count", point_count)

        # TODO(guy.rosman): propagate attention_effective_number back so that we can see whether the attention is
        # TODO  reasonable.
        attention_effective_number = (1 / (attention_score_p**2)).sum(-1).mean().detach().cpu().numpy()
        return map_embeddings_attention

    def forward(
        self,
        additional_input: torch.Tensor,
        trajectory_data: torch.Tensor,
        agent_additional_inputs: Optional[dict] = None,
        additional_params: Optional[dict] = None,
    ) -> tuple:
        """
        Encode map points into a embed vector.
        :param additional_input: agent-centric map points of shape [batch_size, num_max_points, 9],
            The elements in the last dimension are structured as follows
            (x, y, validity, point type, sin(theta), cos(theta), cos(theta), -sin(theta), point id)
        :param trajectory_data: agent-centric trajectory data of shape [batch_size, num_agents, num_past_steps, 3]
        :param agent_additional_inputs: dictionary of additional agent inputs.
        :return: embed vector [batch_size, embed_size]
        """
        map_data = additional_input
        batch_size = map_data.shape[0]
        traj_coordinates = trajectory_data
        trajectory_ints = (
            (traj_coordinates / self.params["map_representation_cache_scale"]).int().detach().cpu().numpy()
        )
        map_ints = (map_data[:, :, :, :2] / self.params["map_representation_cache_scale"]).int().detach().cpu().numpy()
        cache_on = self.params["cache_map_encoder"]

        if not cache_on:
            result, additional_costs = self.compute_forward(
                map_data, trajectory_data, agent_additional_inputs, additional_params
            )
        else:
            cached_results = []
            cached_results_idxs = []
            computed_idxs = []
            for b in range(batch_size):
                batch_item_key = f"{trajectory_ints[b,...]}{map_ints[b,...]}"
                reading_hash = split_reading_hash({}, postfix=f"map_encoder{batch_item_key}")
                invalidate_cache = np.random.uniform() < self.params["map_representation_cache_miss_ratio"]
                if not invalidate_cache and self.map_encoding_cache.is_cached(reading_hash):
                    cached_results_idxs.append(b)
                    cached_results.append(self.map_encoding_cache.load(reading_hash))
                else:
                    computed_idxs.append(b)
            additional_costs = {}

            if len(computed_idxs) == 0:
                result = torch.stack(cached_results, 0)
            else:
                computed_results, additional_costs = self.compute_forward(
                    map_data[computed_idxs, ...],
                    trajectory_data[computed_idxs, ...],
                    agent_additional_inputs,
                    additional_params,
                )
                for b_i, b in enumerate(computed_idxs):
                    batch_item_key = f"{trajectory_ints[b, ...]}{map_ints[b, ...]}"
                    reading_hash = split_reading_hash({}, postfix=f"map_encoder{batch_item_key}")
                    self.map_encoding_cache.save(computed_results[b_i, ...].detach(), key=reading_hash)
                if len(cached_results_idxs) == 0:
                    result = computed_results
                else:
                    _, n_agent, n_timepoints, n_features = computed_results.shape
                    result = computed_results.new_zeros([batch_size, n_agent, n_timepoints, n_features])
                    result[computed_idxs, ...] = computed_results
                    result[cached_results_idxs, ...] = torch.stack(cached_results, 0)

        return result, additional_costs

    def compute_forward(
        self,
        map_data: torch.Tensor,
        trajectory_data: torch.Tensor,
        agent_additional_inputs: dict,
        additional_params: dict,
    ):
        """

        Parameters
        ----------
        map_data
        trajectory_data
        agent_additional_inputs
        additional_params

        Returns
        -------

        """
        # Reshape input since we may have multiple agents.
        (batch_size, num_agents, num_max_points, _) = tuple(map_data.shape)
        (_, _, num_past_steps, _) = tuple(trajectory_data.shape)
        batch_agents_size = batch_size * num_agents
        map_data = map_data.view(batch_agents_size, num_max_points, -1)

        # Update map states for encoding.
        map_xy = map_data[..., :2]
        map_validity = map_data[..., MapDataIndices.MAP_IDX_VALIDITY]
        map_type = map_data[..., MapDataIndices.MAP_IDX_TYPE]
        map_type_one_hot = torch.nn.functional.one_hot(map_type.to(torch.int64), len(self.map_type_values))
        map_tangent = map_data[..., MapDataIndices.MAP_IDX_TANGENT : MapDataIndices.MAP_IDX_TANGENT + 2]
        map_normal = map_data[..., MapDataIndices.MAP_IDX_NORMAL : MapDataIndices.MAP_IDX_NORMAL + 2]
        # Assume id is the last element.
        map_id = map_data[..., -1]

        if self.map_id_type == "integer":
            map_id = map_id.unsqueeze(-1)
        elif self.map_id_type == "binary":
            # Convert decimal ids to binary, for a more compact representation.
            mask = 2 ** torch.arange(self.map_id_size).to(map_id.device)
            map_id_binary = map_id.to(torch.int64).unsqueeze(-1).bitwise_and(mask).ne(0).byte()
            map_id = map_id_binary
        elif self.map_id_type == "onehot":
            # Note: Number of map element can be quite large for Waymo (>100).
            map_id_one_hot = torch.nn.functional.one_hot(map_id.to(torch.int64), self.num_max_map_elements)
            map_id = map_id_one_hot
        else:
            raise Exception("Bad map id type.")

        # [batch_agent_size, num_max_point, 11].
        map_data_expanded = torch.cat([map_xy, map_type_one_hot, map_tangent, map_normal, map_id], -1)
        trajectory_data = trajectory_data.view(batch_agents_size, num_past_steps, -1)
        map_data_normalized = map_data_expanded / map_data_expanded.new_tensor(
            self.map_input_normalization_scales
        ).unsqueeze(0).unsqueeze(0)

        # Get point embedding.
        points_embedding = self.point_encoder(map_data_normalized)
        points_embedding = self.relu(points_embedding)

        # Reconstruct map features such as positions and headings.
        # TODO(cyrushx): Make this optional.
        additional_stats = {}
        for key in self.additional_tasks:
            if key in self.additional_tasks_inputs:
                input_value = torch.cat([points_embedding, points_embedding[..., :1] * 0 + 1.0], -1)
                additional_stats[key] = self.additional_tasks[key](input_value)
            else:
                additional_stats[key] = self.additional_tasks[key](points_embedding)

        # Compute reconstruction errors for map features.
        additional_costs = {}

        map_validity = map_data[..., MapDataIndices.MAP_IDX_VALIDITY : MapDataIndices.MAP_IDX_VALIDITY + 1]
        position_coeff = self.params["map_reconstruction_position_coeff"]
        # TODO(guy.rosman): once adding more reconstruction tasks, feed from parameters. This is a placeholder so
        # TODO  that we know where to normalize costs.
        tangent_coeff = 1.0
        normal_coeff = 1.0
        type_coeff = 1.0
        poly_coeff = 1.0

        position_costs = ((additional_stats["position"] - map_data[..., :2]) ** 2) * map_validity * position_coeff
        accumulate(additional_costs, "position", position_costs.sum())

        tangent = map_data[..., MapDataIndices.MAP_IDX_TANGENT : MapDataIndices.MAP_IDX_TANGENT + 2]
        tangent_costs = ((additional_stats["tangent"] - tangent) ** 2) * map_validity * tangent_coeff
        accumulate(additional_costs, "tangent", tangent_costs.sum() * 0.1)

        normal = map_data[..., MapDataIndices.MAP_IDX_NORMAL : MapDataIndices.MAP_IDX_NORMAL + 2]
        normal_costs = ((additional_stats["normal"] - normal) ** 2) * map_validity * normal_coeff
        accumulate(additional_costs, "normal", normal_costs.sum() * 0.1)

        type_tensor = map_data[..., MapDataIndices.MAP_IDX_TYPE : MapDataIndices.MAP_IDX_TYPE + 1]
        type_costs = ((additional_stats["type_tensor"] - type_tensor) ** 2) * map_validity * type_coeff
        accumulate(additional_costs, "type", type_costs.sum() * 0.1)

        if self.params["map_polyline_feature_degree"] > 0:
            poly_tensor = map_data[
                ...,
                MapDataIndices.MAP_IDX_POLY : MapDataIndices.MAP_IDX_POLY
                + self.params["map_polyline_feature_degree"] * 2,
            ]
            poly_costs = ((additional_stats["poly"] - poly_tensor) ** 2) * map_validity * poly_coeff
            accumulate(additional_costs, "poly", poly_costs.sum() * 0.1)

        if self.map_attention_type == "none":
            map_encoder = self.get_trajectory_map_embeddings
        elif self.map_attention_type == "element":
            map_encoder = self.get_trajectory_map_embeddings_element_attention
        elif self.map_attention_type == "point":
            map_encoder = self.get_trajectory_map_embeddings_point_attention
        else:
            raise Exception("Bad map attention type.")

        outputs = map_encoder(
            map_data,
            points_embedding,
            map_validity,
            num_agents,
            additional_costs,
            trajectory_data,
            agent_index=None,
        )
        # Update agent additional inputs with a function that computes relevant map embeddings given a trajectory.
        if agent_additional_inputs is not None:
            agent_additional_inputs["get_traj_map_embeddings"] = functools.partial(
                map_encoder, map_data, points_embedding, map_validity, num_agents, None
            )

        # Plot estimated positions.
        if self.logger is not None:
            # TODO(nicholas.guyett.ctr) Move this code out of the model and into a configurable callback
            map_log_reconstruction_plot = self.params["map_log_reconstruction_plot"]
            map_plotting_epoch = additional_params is None or (
                "skip_visualization" in additional_params and not additional_params["skip_visualization"]
            )
            if map_log_reconstruction_plot and map_plotting_epoch:
                try:
                    for agent_id in range(num_agents):
                        visualize_map_process(
                            self.logger,
                            agent_id,
                            map_data,
                            self.params["predictor_normalization_scale"],
                            additional_stats["position"],
                            additional_stats["tangent"],
                            trajectory_data,
                            self.params.get("writer_global_step", None),
                        )
                except ValueError as e:
                    logging.error(f"map vis exception: {e}")

        # Return map embedding at each traj point, with shape [batch_size, num_agents, num_past_steps, hidden_dim].
        outputs = outputs.view(batch_size, num_agents, num_past_steps, -1)

        # TODO(guy.rosman): add more task to multitask reconstruction, make sure map is learned properly.
        return outputs, additional_costs


class StubEncoder(nn.Module):
    def __init__(self, input_size=64, embed_size=64, requires_grad=True):
        """
        Encoder of additional input stub, for testing purposes.
        :param embed_size:
        """
        super().__init__()
        self.in_features = input_size
        self.out_features = embed_size
        self.fc_layer = nn.Linear(self.in_features, self.out_features)
        for p in self.parameters():
            p.requires_grad = requires_grad

    def forward(self, stub, trajectory_data, agent_additional_inputs=None, additional_params=None):
        """
        Encode stub input into an embed vector.
        :param: the number of params is consistent with other input encoders
        :return:
        """
        output = self.fc_layer(stub)
        additional_costs = {}
        return output, additional_costs


class MLPEncoder(nn.Module):
    def __init__(self, input_size=64, embed_size=64):
        """
        MLP Encoder of additional input.

        Parameters
        ----------
        input_size : int
            Input size.
        embed_size : int
            Embedding size.
        """
        super().__init__()
        self.in_features = input_size
        self.out_features = embed_size
        self.mlp = MLP(input_size, embed_size)

    def forward(self, stub, trajectory_data=None, agent_additional_inputs=None, additional_params=None):
        """
        Encode input into an embed vector.

        Parameters
        ----------
        stub : torch.Tensor
            Input tensor.

        Returns
        -------
        hidden_states: torch.Tensor
            Embedded tensor.

        """
        hidden_states = self.mlp(stub)
        return hidden_states, {}


class MapAttentionEncoder(AdditionalInputEncoder):
    """
    Encode map points using attention.

    Parameters
    ----------
    map_input_dim : int
        Dimension of map input.
    traj_input_dim : int
        Dimension of trajectory input.
    embed_size : int
        Size of embeddings.
    device : str
        Device of model and tensors.
    params : dict
        Dictionary of parameters.
    """

    def __init__(
        self,
        map_input_dim: int = 7,
        traj_input_dim: int = 3,
        embed_size: int = 32,
        device: str = "cpu",
        params: dict = {},
    ):
        super().__init__()
        self.map_input_dim = map_input_dim
        self.traj_input_dim = traj_input_dim
        self.embed_size = embed_size
        self.out_features = embed_size * 2
        self.relu = nn.ReLU()
        self.params = params
        self.device = device

        self.map_encoder = LocalGraph(16, self.embed_size, 1)
        self.local_hidden_dim = self.embed_size * 2
        if params["use_global_map"]:
            self.map_out_linear = nn.Linear(self.local_hidden_dim, self.embed_size)
        else:
            self.agent_encoder = LocalGraph(4, self.embed_size, 1)
            self.global_encoder = GlobalGraph(self.local_hidden_dim)

    def forward(
        self,
        additional_input: torch.Tensor,
        trajectory_data: torch.Tensor,
        agent_additional_inputs: Optional[dict] = None,
        additional_params: Optional[dict] = None,
    ) -> tuple:
        """
        Encode map points into a embed vector.
        :param additional_input: agent-centric map points of shape [batch_size, num_max_points, 9],
            The elements in the last dimension are structured as follows
            (x, y, validity, point type, sin(theta), cos(theta), cos(theta), -sin(theta), point id)
        :param trajectory_data: agent-centric trajectory data of shape [batch_size, num_agents, num_past_steps, 3]
        :param agent_additional_inputs: dictionary of additional agent inputs.
        :return: embed vector [batch_size, embed_size]
        """
        map_data = additional_input.cpu().numpy()
        # Reshape input since we may have multiple agents.
        if len(map_data.shape) == 3:  # global maps
            (batch_size, num_max_points, _) = tuple(map_data.shape)
            batch_agents_size = batch_size
        else:  # agent maps
            (batch_size, num_agents, num_max_points, _) = tuple(map_data.shape)
            batch_agents_size = batch_size * num_agents
        (_, _, num_past_steps, _) = tuple(trajectory_data.shape)
        map_data = map_data.reshape(batch_agents_size, num_max_points, -1)
        trajectory_data = trajectory_data.reshape(batch_agents_size, num_past_steps, -1)

        # Update map states for encoding.
        map_xy = map_data[..., :2]
        map_validity = map_data[..., MapDataIndices.MAP_IDX_VALIDITY] > 0
        map_type = map_data[..., MapDataIndices.MAP_IDX_TYPE : MapDataIndices.MAP_IDX_TYPE + 1]
        map_tangent = map_data[..., MapDataIndices.MAP_IDX_TANGENT : MapDataIndices.MAP_IDX_TANGENT + 2]
        map_normal = map_data[..., MapDataIndices.MAP_IDX_NORMAL : MapDataIndices.MAP_IDX_NORMAL + 2]
        # Assume id is the last element.
        map_id = map_data[..., -1]

        if self.params["map_mask_distance_threshold"] > 0:
            # Filter map elements by a distance threshold.
            # Compute map distance for filters.
            map_distance = np.sqrt((map_xy**2).sum(-1)) / self.params["predictor_normalization_scale"]
            map_distance_mask = map_distance < self.params["map_mask_distance_threshold"]
            map_distance_mask = map_validity & map_distance_mask
        else:
            map_distance_mask = None

        # Combine useful map states.
        map_states = np.concatenate([map_xy, map_type, map_tangent, map_normal, map_id[..., np.newaxis]], -1)
        map_states = map_states * map_validity[..., np.newaxis]
        map_dim = map_states.shape[-1]

        # Divide map states to segments based on ids.
        map_segment_size = self.params["map_segment_size"]
        map_segment_max = self.params["map_segment_max"]
        map_segment_size_subsampled = int(np.ceil(map_segment_size / self.params["map_points_subsample_ratio"]))

        map_segment_states = np.zeros((batch_agents_size, map_segment_max, map_segment_size_subsampled, map_dim))
        map_segment_counts = []
        for b in range(batch_size):
            map_id_b = map_id[b][map_validity[b]]
            map_id_values, map_id_index_start, map_id_counts = np.unique(
                map_id_b, return_index=True, return_counts=True
            )

            map_segment_count = 0
            map_segment_ids = []
            for id_s, id_e in zip(map_id_index_start[:-1], map_id_index_start[1:]):
                while id_s + map_segment_size <= id_e:
                    map_states_seg = map_states[b, id_s : id_s + map_segment_size][
                        :: self.params["map_points_subsample_ratio"]
                    ]
                    map_segment_ids.append([id_s, id_s + map_segment_size])
                    id_s = id_s + map_segment_size
                    # Skip adding map segment if the entire segment is masked.
                    if map_distance_mask is not None:
                        map_states_seg_mask = map_distance_mask[b, id_s : id_s + map_segment_size]
                        if map_states_seg_mask.sum() == 0:
                            continue
                    map_segment_states[b, map_segment_count, : map_states_seg.shape[0]] = map_states_seg
                    map_segment_count += 1

                if id_s != id_e:
                    map_segment_ids.append([id_s, id_e])
                    # Skip adding map segment if the entire segment is masked.
                    if map_distance_mask is not None:
                        map_states_seg_mask = map_distance_mask[b, id_s:id_e]
                        if map_states_seg_mask.sum() == 0:
                            continue
                    map_states_seg_id = map_states[b, id_s, -2]
                    if map_states_seg_id <= 16:
                        map_states_seg = map_states[b, id_s:id_e][:: self.params["map_points_subsample_ratio"]]
                    else:
                        # Do not subsample for map types, including stop sign, cross walk, speed bump.
                        map_states_seg = map_states[b, id_s:id_e][:map_segment_size_subsampled]
                    map_segment_states[b, map_segment_count, : map_states_seg.shape[0]] = map_states_seg
                    map_segment_count += 1
            map_segment_counts.append(map_segment_count)

        # Subsample map states with dynamical segment counts.
        map_segment_states = map_segment_states[:, : np.max(map_segment_counts)]

        # Obtain map vectors.
        map_vectors = np.concatenate([map_segment_states[:, :, :-1], map_segment_states[:, :, 1:]], -1)
        # [batch_size, num_segments, num_segment_size - 1, 16].
        map_vectors = torch.tensor(map_vectors, dtype=torch.float, device=self.device)
        map_embedding_l = self.map_encoder(map_vectors)

        # Only perform agent self-attention and agent-map cross attention if not using global map
        if not self.params["use_global_map"]:
            # Obtain agent vectors.
            agent_states = trajectory_data.unsqueeze(1)[..., :2]
            # [batch_size, 1, num_past_steps - 1, 4].
            agent_vectors = torch.cat([agent_states[:, :, :-1], agent_states[:, :, 1:]], -1)
            agent_embedding_l = self.agent_encoder(agent_vectors)

            # Agent embedding size: [batch_size, num_agents, agent_hidden_dim].
            # Map embedding size: [batch_size, num_map_segments, map_hidden_dim].
            agent_embedding, map_embedding = self.global_encoder(agent_embedding_l, map_embedding_l)

            # Repeat agent embedding along step dimension.
            agent_embedding = agent_embedding.unsqueeze(2).repeat(1, 1, num_past_steps, 1)

            # Return map embedding at each traj point, with shape [batch_size, num_agents, num_past_steps, hidden_dim].
            outputs = agent_embedding.view(batch_size, num_agents, num_past_steps, -1)
        else:
            # Skip agent attention as will be handled in the downstream models, e.g. agent transformer.
            outputs = self.map_out_linear(map_embedding_l)

        return outputs, {}


class LocalGraph(nn.Module):
    """
    Local graph module to process point wise features within an instance such as an agent or a map feature,
    via mlp and map-pooling.

    Parameters
    ----------
    input_size : int
        Dimension of input point state.
    hidden_dim : int
        Dimension of hidden state.
    layers : int
        Number of layers for local graph.
    """

    def __init__(self, input_size, hidden_dim, layers=1):
        super(LocalGraph, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.input_encoders = []
        for i in range(self.layers):
            if i == 0:
                input_encoder = nn.Sequential(MLP(input_size, self.hidden_dim), MLP(self.hidden_dim, self.hidden_dim))
            else:
                hidden_dim_i = self.hidden_dim * 2 * i
                input_encoder = nn.Sequential(MLP(hidden_dim_i, hidden_dim_i), MLP(hidden_dim_i, hidden_dim_i))
            self.input_encoders.append(input_encoder)
        self.input_encoders = nn.ModuleList(self.input_encoders)

    def forward(self, input_states):
        input_embedding = input_states
        # TODO(cyrushx): Try self-attention.
        for i in range(self.layers):
            input_embedding = self.input_encoders[i](input_embedding)
            # Get max pool of neighbor embeddings.
            neighbor_embedding = []
            neighbor_embedding_mask = torch.zeros_like(input_embedding)
            node_size = neighbor_embedding_mask.shape[2]
            for i in range(node_size):
                # Set self node to a large negative number to ignore itself in neighbor max-pooling.
                neighbor_embedding_mask[:, :, i] -= 10000.0
                neighbor_embedding.append(torch.max(input_embedding + neighbor_embedding_mask, 2)[0])
                neighbor_embedding_mask[:, :, i] += 10000.0
            neighbor_embedding = torch.stack(neighbor_embedding, 2)
            input_embedding = torch.cat([input_embedding, neighbor_embedding], -1)
        # Take max-pool over all nodes.
        input_embedding = torch.max(input_embedding, 2)[0]
        return input_embedding


class GlobalGraph(nn.Module):
    """
    A global graph module based on attention to module interactions between agent features and map features,
    where features for each agent/map instance are processed by a local graph module.

    The self/cross attention mechanisms are inspired from LaneGCN paper (https://arxiv.org/abs/2007.13732).

    Parameters
    ----------
    hidden_dim : int
        Dimension of hidden state.
    """

    def __init__(self, hidden_dim):
        super(GlobalGraph, self).__init__()
        self.hidden_dim = hidden_dim

        self.global_attention_head = 1
        self.global_layer = 1

        # Self attention for target agent features.
        self.attention_agent = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=self.global_attention_head, batch_first=True
        )
        # Self attention for map features.
        self.attention_map = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=self.global_attention_head, batch_first=True
        )
        # Cross attention from target agent to map.
        self.attention_a2m = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=self.global_attention_head, batch_first=True
        )
        # Cross attention from target agent to context agent.
        self.attention_a2c = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=self.global_attention_head, batch_first=True
        )
        # Cross attention from map to target agent.
        self.attention_m2a = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=self.global_attention_head, batch_first=True
        )
        # Cross attention from context agent to target agent.
        self.attention_c2a = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=self.global_attention_head, batch_first=True
        )

    def forward(self, agent_embedding, map_embedding):
        target_agent_embedding = agent_embedding[:, :1]
        # context_agent_embedding = agent_embedding[:, 1:]
        for _ in range(self.global_layer):
            # Run cross attention from target agent to map.
            map_embedding_a2m, _ = self.attention_a2m(
                query=map_embedding,
                key=torch.cat([target_agent_embedding, map_embedding], 1),
                value=torch.cat([target_agent_embedding, map_embedding], 1),
            )
            # Add residual.
            map_embedding += map_embedding_a2m

            # TODO(cyrushx): Add attention for context agents.
            # # Run cross attention from target agent to context.
            # context_agent_embedding_a2c, _ = self.attention_a2c(
            #     query=context_agent_embedding,
            #     key=torch.cat([target_agent_embedding, context_agent_embedding], 1),
            #     value=torch.cat([target_agent_embedding, context_agent_embedding], 1),
            # )
            # # Add residual.
            # context_agent_embedding += context_agent_embedding_a2c

            # Run self attention and cross attention and fuse encodings.
            target_agent_embedding_a, _ = self.attention_agent(
                query=target_agent_embedding, key=target_agent_embedding, value=target_agent_embedding
            )
            target_agent_embedding_m2a, _ = self.attention_m2a(
                query=target_agent_embedding, key=map_embedding, value=map_embedding
            )
            target_agent_embedding = target_agent_embedding_a + target_agent_embedding_m2a

            # TODO(cyrushx): Add attention for context agents.
            # target_agent_embedding_c2a, _ = self.attention_c2a(
            #     query=target_agent_embedding, key=context_agent_embedding, value=context_agent_embedding
            # )
            # target_agent_embedding = target_agent_embedding_a + target_agent_embedding_m2a + target_agent_embedding_c2a

        return target_agent_embedding, map_embedding
