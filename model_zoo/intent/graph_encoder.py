import torch
from torch import nn

from model_zoo.intent.create_networks import create_mlp
from model_zoo.intent.enc_dec_temporal_models import AgentTemporalModel, TypeConditionedTemporalModel


class GraphEncoder(nn.Module):
    """
    A Graph neural network to encode a graph into a tensor. Temporal edges are represented via edge modules (all agents)
    and LSTMs (self-edges only)
    """

    def __init__(
        self,
        input_dim,
        node_hidden_dim,
        edge_hidden_dim,
        output_dim,
        type_dim,
        scene_dim,
        agent_dim,
        params,
        nullify_nan_inputs=False,
        temporal_truncated_steps=-1,
        leaky_relu=False,
    ):
        """
        :param input_dim: dimensionality of the main input (positions).
        :param node_hidden_dim: dimensionality of node states.
        :param edge_hidden_dim: dimensionality of edge states.
        :param output_dim: dimensionality of the output layer from the nodes/network.
        :param type_dim: dimensionality of the type input (used for agent/node type)
        :param scene_dim: dimensionality of the scene-global inputs.
        :param agent_dim: dimensionality of the agent-specific inputs.
        :param params: parameters dictionary (new parametrs should go here)
        :param nullify_nan_inputs: whether to set nan inputs to zero (only do this if you have a reason to).
        :param temporal_truncated_steps: whether to truncate temporal steps (i.e. TBPTT)
        :param leaky_relu: whether to use leaky relu's.
        """
        super().__init__()
        self.params = params
        self.dropout_ratio = params["dropout_ratio"]
        self.type_dim = type_dim
        self.input_dim = input_dim
        self.scene_dim = scene_dim
        self.agent_dim = agent_dim
        self.use_layer_norm = self.params["predictor_layer_norm"]

        # Layer for input processing per agent, input_dim+1 - to accommodate valid bit
        if self.params["coordinate_encoder_widths"] is None:
            linear_net = nn.Linear(2 * (input_dim + 1), node_hidden_dim)
            if self.params["special_init"]:
                nn.init.normal_(linear_net.bias, 0.0, 0.01)
                nn.init.xavier_uniform_(linear_net.weight)
            # TODO(guy.rosman) -- followup, see why BatchNorm is causing problems, check w/ Cyrus.
            # self.input_module = nn.Sequential(linear_net,nn.BatchNorm1d(node_hidden_dim))
            if self.use_layer_norm:
                self.input_module = nn.Sequential(linear_net, nn.LayerNorm(node_hidden_dim), nn.ReLU())
            else:
                self.input_module = nn.Sequential(linear_net, nn.ReLU())
        else:
            # TODO(cyrushx): Fix BatchNorm1d in create_mlp
            self.input_module = create_mlp(
                input_dim=2 * (input_dim + 1),
                layers_dim=self.params["coordinate_encoder_widths"] + [node_hidden_dim],
                dropout_ratio=self.dropout_ratio,
                leaky_relu=leaky_relu,
                pre_bn=True,
                batch_norm=self.params["predictor_batch_norm"],
            )
        # Layer for output prediction.
        if self.params["layer_pred_widths"] is None:
            self.layer_pred = nn.Linear(node_hidden_dim, output_dim)
            if self.params["special_init"]:
                nn.init.normal_(self.layer_pred.bias, 0.0, 0.01)
                nn.init.xavier_uniform_(self.layer_pred.weight)

            if self.use_layer_norm:
                self.layer_pred = nn.Sequential(self.layer_pred, nn.LayerNorm(output_dim), nn.ReLU())
            else:
                self.layer_pred = nn.Sequential(self.layer_pred, nn.ReLU())

        else:
            self.layer_pred = create_mlp(
                input_dim=node_hidden_dim,
                layers_dim=self.params["layer_pred_widths"] + [output_dim],
                dropout_ratio=self.dropout_ratio,
                leaky_relu=leaky_relu,
                batch_norm=self.params["predictor_batch_norm"],
            )

            # Edge module for inputting node states to edge states.
        if self.params["edge_encoder_widths"] is None:
            self.edge_module = nn.Sequential(
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(node_hidden_dim * 2 + edge_hidden_dim * 2, edge_hidden_dim),
                nn.BatchNorm1d(edge_hidden_dim),
            )
        else:
            self.edge_module = create_mlp(
                input_dim=node_hidden_dim * 2 + edge_hidden_dim * 2,
                layers_dim=self.params["edge_encoder_widths"] + [edge_hidden_dim],
                dropout_ratio=self.dropout_ratio,
                leaky_relu=leaky_relu,
                pre_bn=True,
            )

        # Update edge state from nodes, for the self-temporal edge.
        if self.params["self_edge_encoder_widths"] is None:
            self_edge_layers = [
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(node_hidden_dim + edge_hidden_dim + type_dim, edge_hidden_dim),
            ]
            if self.params["predictor_batch_norm"]:
                self_edge_layers.append(nn.BatchNorm1d(edge_hidden_dim))

            self.self_edge_module = nn.Sequential(*self_edge_layers)
        else:
            self.self_edge_module = create_mlp(
                input_dim=node_hidden_dim + edge_hidden_dim + type_dim,
                layers_dim=self.params["self_edge_encoder_widths"] + [edge_hidden_dim],
                dropout_ratio=self.dropout_ratio,
                leaky_relu=leaky_relu,
                pre_bn=True,
                batch_norm=self.params["predictor_batch_norm"],
            )
        # Connects coordinate inputs into the edge.
        self.edge_input_module = nn.Sequential(nn.Linear(4 * (input_dim + 1) + 2 * type_dim, edge_hidden_dim))
        # Layer for processing one-hot agent_type as edge input
        self.edge_type_processor = nn.Linear(self.type_dim * 2, edge_hidden_dim)
        self.node_type_processor = nn.Linear(self.type_dim, node_hidden_dim)

        # Layer for processing one-hot agent_type as self-edge input
        self.self_edge_type_processor = None
        # The pooling from nearby(+self) edges to each node
        self.edge_to_node_module = nn.Linear(edge_hidden_dim, node_hidden_dim)
        if self.params["special_init"]:
            nn.init.normal_(self.edge_to_node_module.bias, 0.0, 0.1)
            nn.init.xavier_normal_(self.edge_to_node_module.weight)

        # The temporal update at every timestep of the node. node_dim *3 input - for coordinates, edges, type embedding,
        # agent inputs.
        lstm_node_hidden_dim_multiplier = 3
        self.scene_processor = None
        self.agent_processor = None
        if scene_dim > 0:
            self.scene_processor = nn.Linear(scene_dim, node_hidden_dim)
            lstm_node_hidden_dim_multiplier += 1
        if agent_dim > 0:
            self.agent_processor = nn.Linear(agent_dim, node_hidden_dim)
            lstm_node_hidden_dim_multiplier += 1

        if params["type_conditioned_temporal_model"]:
            self.temporal_model = TypeConditionedTemporalModel(
                input_size=node_hidden_dim * lstm_node_hidden_dim_multiplier,
                hidden_size=node_hidden_dim,
                num_agent_types=self.type_dim,
            )
        else:
            self.temporal_model = AgentTemporalModel(
                input_size=node_hidden_dim * lstm_node_hidden_dim_multiplier,
                hidden_size=node_hidden_dim,
                num_agent_types=self.type_dim,
                use_layer_norm=self.params["predictor_layer_norm"],
            )
        if self.params["predictor_batch_norm"]:
            self.inputs_normalizer = nn.BatchNorm1d(node_hidden_dim * lstm_node_hidden_dim_multiplier)
        else:
            self.inputs_normalizer = None
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.out_dim = output_dim
        self.past_time_points = None
        self.num_nodes = None
        self.edge_types = []
        self.agent_states = {}
        self.edge_states = {}
        self.edges = []
        self.temporal_dropout = None
        self.nullify_nan_inputs = nullify_nan_inputs
        self.temporal_truncated_steps = temporal_truncated_steps

    def forward(self, inputs):
        """
        Runs over the graph, and creates an output from each node.
        :param inputs: a dictionary including inputs to the encoder.
            trajectories: a N_agents x N_timepoints x3 tensor
            normalized_trajectories: a trajectories vector after normalization to the agent frame.
            agent_type: N_agents x N_types
            is_valid: indicator of which agent/timestep is valid. Not used / redundant?
            scene_data: global data.
            initial_state:
        :return: result_tensor, node_states
        result_tensor is  batch_size x num_agents x timesteps x encoding_dim
        node_states is a a timesteps-long list of num_agents-long lists of 2-tuples with encoding_dim in each
        element, to capture temporal model hidden and cell states.
        """

        trajectories = inputs["trajectories"]
        normalized_trajectories = inputs["normalized_trajectories"]
        agent_type = inputs["agent_type"]
        # TODO(guy.rosman): check and if so remove.
        scene_data = inputs["scene_data"]
        agent_data = inputs["agent_data"]
        relevant_agents = inputs["relevant_agents"]
        initial_state = inputs["initial_state"]

        # Output for all agents
        res = []
        node_states_res = []

        # Disable global trajectories in the input.
        if self.params["encoder_normalized_trajectory_only"]:
            both_trajectories = torch.cat([normalized_trajectories, torch.zeros_like(trajectories)], 3)
        else:
            # [batch, agents, timestamps, 3(x, y, valid)]
            both_trajectories = torch.cat([normalized_trajectories, trajectories], 3)

        if self.nullify_nan_inputs:
            invalid_mask = both_trajectories[:, :, :, 2] == 0
            both_trajectories[invalid_mask] = 0
        batch_size = trajectories.shape[0]

        def run_on_4D(net, tensor, out_dim):
            batch_size = tensor.shape[0]
            in_dim = tensor.shape[-1]
            res = net(tensor.view(-1, in_dim))
            return res.view(batch_size, tensor.shape[1], tensor.shape[2], out_dim)

        # [batch, agents, timestamps, hidden_dim]
        node_processed_trajectories = run_on_4D(self.input_module, both_trajectories, self.node_hidden_dim)
        # [batch, agents, hidden_dim]
        node_processed_agent_type = self.node_type_processor(agent_type)
        for t in range(self.past_time_points):
            # Update nodes in current layer
            for i in range(self.num_agents):
                if self.scene_dim > 0:
                    if len(scene_data.shape) == 2:
                        import IPython

                        IPython.embed()
                    processed_scene = self.scene_processor(scene_data[:, t, :])
                if self.agent_dim > 0:
                    # TODO(guy.rosman): add per-agent agent data.
                    processed_agent = self.agent_processor(agent_data[:, i, t, :])
                    # Zero irrelevant agent states.
                    processed_agent = processed_agent * relevant_agents[:, i][:, None].float()
                    if self.params["graph_input_dropout"] > 0:
                        processed_agent = (
                            processed_agent.clone().uniform_() > self.params["graph_input_dropout"]
                        ) * processed_agent

                if t == 0:
                    # First timestep nodes are with just the inputs.
                    # [batch, hidden_dim]
                    node_type_input = node_processed_agent_type[:, i, :]
                    node_edges_input = node_processed_trajectories.new_zeros(batch_size, self.node_hidden_dim)
                    node_coord_input = node_processed_trajectories[:, i, t, :]
                    if initial_state is not None:
                        init_state = initial_state[0][i][0]
                    else:
                        # [batch, hidden_dim]
                        init_state = node_coord_input * 0
                    lstm_input_list = [node_coord_input, node_type_input, node_edges_input]
                    if self.scene_dim > 0:
                        if len(processed_scene.shape) > 2:
                            # use information from the current time frame
                            # ideally, every information bit should be per-time-frame
                            lstm_input_list.append(processed_scene[:, t, :])
                        else:
                            # use global information
                            lstm_input_list.append(processed_scene)

                    if self.agent_dim > 0:
                        lstm_input_list.append(processed_agent)

                    # [batch, hidden_dim*3]
                    lstm_input = torch.cat(lstm_input_list, 1)
                    # [1, batch, hidden_dim]
                    h = init_state.unsqueeze(0)
                    c = h * 0
                    if self.params["predictor_batch_norm"]:
                        lstm_input = self.inputs_normalizer(lstm_input.clone())
                        # import IPython;IPython.embed()
                    _, lstm_hidden = self.temporal_model(
                        lstm_input.view(1, batch_size, -1), (h, c), agent_type[:, i, :]
                    )
                    self.agent_states[i] = {t: lstm_hidden}

                else:
                    # For later timesteps, nodes are created by feeding neighbor edges and inputs to the LSTM.
                    node_coord_input = node_processed_trajectories[:, i, t, :]
                    node_type_input = node_processed_agent_type[:, i, :]
                    node_edges_input = node_processed_trajectories.new_zeros(batch_size, self.node_hidden_dim)
                    if not self.params["disable_gnn_edges"]:
                        for e in self.edges:
                            i1, j1, _ = e
                            if i1 == j1 and self.params["disable_self_edges"]:
                                continue
                            if j1 == i:
                                temporal_update = self.edge_to_node_module(self.edge_states[e][t - 1])
                                if self.temporal_dropout is not None:
                                    temporal_update = self.temporal_dropout(temporal_update)
                                node_edges_input = node_edges_input + temporal_update
                    if self.training:
                        node_coord_input_dropout = (
                            node_coord_input.clone().uniform_() >= self.params["coordinate_encoder_dropout"]
                        ).float()
                        edge_input_dropout = (
                            node_edges_input.clone().uniform_() >= self.params["edge_dropout"]
                        ).float()
                    else:
                        node_coord_input_dropout = node_coord_input.new_ones(node_coord_input.shape)
                        edge_input_dropout = node_coord_input.new_ones(node_edges_input.shape)
                    lstm_input_list = [
                        node_coord_input_dropout * node_coord_input,
                        node_type_input,
                        edge_input_dropout * node_edges_input,
                    ]
                    if self.scene_dim > 0:
                        if len(processed_scene.shape) > 2:
                            # use information from the current time frame
                            # ideally, every information bit should be per-time-frame
                            lstm_input_list.append(processed_scene[:, t, :])
                        else:
                            # use global information
                            lstm_input_list.append(processed_scene)

                    if self.agent_dim > 0:
                        # Add processed agent state to input list.
                        if len(processed_agent.shape) > 2:
                            lstm_input_list.append(processed_agent[:, t, :])
                        else:
                            lstm_input_list.append(processed_agent)

                        # Feed to the recurrent unit a sum of neighbor edge feeds and the current node "observation"
                        # input.
                    lstm_input = torch.cat(lstm_input_list, 1)

                    if self.params["predictor_batch_norm"]:
                        lstm_input = self.inputs_normalizer(lstm_input.clone())

                    _, node_state = self.temporal_model(
                        lstm_input.view(1, batch_size, -1), self.agent_states[i][t - 1], agent_type[:, i, :]
                    )
                    self.agent_states[i][t] = node_state
                    # import IPython;IPython.embed(header='check node state')

            # Update edges in current layer
            for e in self.edges:
                i1, j1, _ = e
                if t == 0:
                    self.edge_states[e] = {}
                agent_input_i = both_trajectories[:, i1, t, :]  # [batch, agent, time, 6]
                agent_input_j = both_trajectories[:, j1, t, :]  # [batch, agent, time, 6]
                agent_type_i = agent_type[:, i1, :]
                agent_type_j = agent_type[:, j1, :]
                agent_inputs = torch.cat([agent_input_i, agent_type_i, agent_input_j, agent_type_j], 1)
                edge_input = self.edge_input_module(agent_inputs)

                node_state_i = self.agent_states[i1][t][0][0, :, :]  # [layer, batch, hidden]
                node_state_j = self.agent_states[j1][t][0][0, :, :]
                if i1 != j1:
                    edge_agent_type = self.edge_type_processor(torch.cat([agent_type_i, agent_type_j], 1))
                    self.edge_states[e][t] = self.edge_module(
                        torch.cat([node_state_i, node_state_j, edge_input, edge_agent_type], 1)
                    )
                else:
                    if self.self_edge_type_processor is not None:
                        edge_agent_type = self.self_edge_type_processor(torch.cat([agent_type_i], 1))
                    else:
                        edge_agent_type = agent_type_i
                    self.edge_states[e][t] = self.self_edge_module(
                        torch.cat([node_state_i, edge_input, edge_agent_type], 1)
                    )

            if t == self.temporal_truncated_steps:
                for e in self.edges:
                    self.edge_states[e][t] = self.edge_states[e][t].detach()

                for i in range(self.num_agents):
                    self.agent_states[i][t] = tuple(x.detach() for x in self.agent_states[i][t])

            res1 = []
            node_states = []
            for i in range(self.num_agents):
                res1.append(self.layer_pred(self.agent_states[i][t][0][0, :, :]))  # [time, batch, intermediate_dim]
                node_states.append([x.squeeze(0) for x in self.agent_states[i][t]])
            res1 = torch.stack(res1, 0)
            node_states_res.append(node_states)
            res.append(res1)  # [time, agent, batch, intermediate_dim]
            if t > 1:
                for e in self.edges:
                    self.edge_states[e].pop(t - 1)
            if t > 1:
                for i in range(self.num_agents):
                    self.agent_states[i].pop(t - 2)

        res = torch.stack(res, 0).permute(2, 1, 0, 3)  # [batch, agent, time, intermediate_dim]
        return res, node_states_res, {}

    def set_num_agents(self, num_agents, num_pastpoints):
        """
        Reset the graph structure to match the number of agents, timepoints
        :param num_agents:
        :param num_pastpoints:
        :return:
        """
        self.num_agents = num_agents
        self.past_time_points = num_pastpoints
        self.agent_states = {}
        for i in range(self.num_agents):
            self.agent_states[i] = []
        self.edge_states = {}
        self.edges = []
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                self.edges.append((i, j, "edge"))

    def save_model(self, data, is_valid):
        # TODO(guy.rosman): Implement using torch.jit.trace.
        pass
