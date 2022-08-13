import torch
from torch import nn as nn

from model_zoo.intent.create_networks import create_mlp
from model_zoo.intent.enc_dec_temporal_models import AgentTemporalModel, TypeConditionedTemporalModel


class BatchGraphEncoder(nn.Module):
    """
    A Graph neural network to encode a graph into a tensor. Temporal edges are represented via edge modules (all agents) and LSTMs (self-edges only)
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
        params={},
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
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.agent_feature_dim = 4 * (input_dim + 1) + 2 * type_dim
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
                self.input_module = nn.Sequential(
                    linear_net, nn.ReLU(), nn.Dropout(self.params["coordinate_encoder_dropout"])
                )
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
                nn.Linear(node_hidden_dim * 2 + edge_hidden_dim, edge_hidden_dim),
                nn.BatchNorm1d(edge_hidden_dim),
            )
        else:
            self.edge_module = create_mlp(
                input_dim=node_hidden_dim * 2 + edge_hidden_dim,
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
        self.edge_input_module = nn.Sequential(nn.Linear(self.agent_feature_dim, edge_hidden_dim))
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

        # The temporal update at every timestep of the node. node_dim *3 input - for coordinates, edges, type embedding, agent inputs.
        lstm_node_hidden_dim_multiplier = 3
        self.scene_processor = None
        self.agent_processor = None
        if scene_dim > 0:
            if self.params["graph_input_dropout"] > 0:
                self.scene_processor = nn.Sequential(
                    nn.Linear(scene_dim, node_hidden_dim), nn.Dropout(self.params["graph_input_dropout"])
                )
            else:
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
        self.agent_states = []
        self.edge_states = {}
        self.edges = []
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
        node_states is a a timesteps-long list of 2-tuples with encoding_dim in each
        element, to capture temporal model hidden and cell states.  Each element is [agent, batch, encoding_dim]
        """

        trajectories = inputs["trajectories"]
        normalized_trajectories = inputs["normalized_trajectories"]
        agent_type = inputs["agent_type"]
        is_valid = inputs["is_valid"]
        # TODO(guy.rosman): check and if so remove.
        scene_data = inputs["scene_data"]
        agent_data = inputs["agent_data"]
        relevant_agents = inputs["relevant_agents"]
        initial_state = inputs["initial_state"]

        # Output for all agents
        res = []
        node_states_res = []
        device = normalized_trajectories.device
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

        # Compute global graph inputs, trajectories and agent types, that don't change per time step.

        # [batch, agents, timestamps, hidden_dim]
        node_processed_trajectories = run_on_4D(self.input_module, both_trajectories, self.node_hidden_dim)
        # [batch, agents, hidden_dim]
        node_processed_agent_type = self.node_type_processor(agent_type)
        # reshaped_agent_data = agent_data.view(agent_data.shape[0] * agent_data.shape[1], agent_data.shape[2], agent_data.shape[3])

        # TODO(guy.rosman): add per-agent agent data.
        if self.agent_dim > 0:
            # Batch and process agent state
            batched_processed_agent = self.agent_processor(agent_data)
            # Zero irrelevant agent states.
            batched_processed_agent = batched_processed_agent * relevant_agents[:, :, None, None].float()

        # TODO(paul.drews): Encapsulate the phases of this loop to make reading easier.
        # Try to clean/batch the agent loop and remove run_on_4D.
        # Computation structure (where each node in the graph is an agent state at time t,
        # graph edges connect all agents within timestep t, LSTMs update agent states through time,
        # and edge states persist from t to t+1):
        #
        # for each time point
        #   update agent state LSTMs given global agent state and edge inputs
        #   calculate edge inputs for input and output node
        #   update edge state given input and output node state and edge inputs
        # Calculate output given final timestamp node hidden state
        for t in range(self.past_time_points):
            if self.scene_dim > 0:
                if len(scene_data.shape) == 2:
                    import IPython

                    IPython.embed()
                processed_scene = self.scene_processor(scene_data[:, t, :])

            #########################
            # Update agent state LSTMs given global agent state and edge inputs
            for i in range(self.num_agents):
                if t == 0:
                    # First timestep nodes are with just the inputs.
                    # [batch, hidden_dim]
                    node_type_input = node_processed_agent_type[:, i, :]
                    node_edges_input = node_processed_trajectories.new_zeros(batch_size, self.node_hidden_dim)
                    node_coord_input = node_processed_trajectories[:, i, t, :]
                    if initial_state is not None:
                        init_state = initial_state[0][0][i, :, :]
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
                        lstm_input_list.append(batched_processed_agent[:, i, t, :])

                    # [batch, hidden_dim*3]
                    lstm_input = torch.cat(lstm_input_list, 1)
                    # [1, batch, hidden_dim]
                    h = init_state.unsqueeze(0)
                    c = h * 0
                    if self.params["predictor_batch_norm"]:
                        lstm_input = self.inputs_normalizer(lstm_input.clone())
                    _, lstm_hidden = self.temporal_model(
                        lstm_input.view(1, batch_size, -1), (h, c), agent_type[:, i, :]
                    )
                    # import IPython;IPython.embed()
                    # exit()

                else:
                    # For later timesteps, nodes are created by feeding neighbor edges and inputs to the LSTM.
                    node_coord_input = node_processed_trajectories[:, i, t, :]
                    node_type_input = node_processed_agent_type[:, i, :]
                    node_edges_input = node_processed_trajectories.new_zeros(batch_size, self.node_hidden_dim)
                    if not self.params["disable_gnn_edges"]:
                        for e in self.edges:
                            i1, j1, e_id = e
                            if i1 == j1 and self.params["disable_self_edges"]:
                                continue
                            if j1 == i:
                                temporal_update = self.edge_to_node_module(self.edge_states[t - 1][:, e[2], :])
                                node_edges_input = node_edges_input + temporal_update
                    lstm_input_list = [
                        node_coord_input,
                        node_type_input,
                        node_edges_input,
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
                        lstm_input_list.append(batched_processed_agent[:, i, t, :])

                        # Feed to the recurrent unit a sum of neighbor edge feeds and the current node "observation" input.
                    lstm_input = torch.cat(lstm_input_list, 1)

                    if self.params["predictor_batch_norm"]:
                        lstm_input = self.inputs_normalizer(lstm_input.clone())

                    _, lstm_hidden = self.temporal_model(
                        lstm_input.view(1, batch_size, -1),
                        (self.agent_states[t - 1][0][i : i + 1, :, :], self.agent_states[t - 1][1][i : i + 1, :, :]),
                        agent_type[:, i, :],
                    )

                if i == 0:
                    self.agent_states[t] = lstm_hidden
                else:
                    # Note that we utilize the unused layers dimension to concatenate multiple agent hidden states into one tensor
                    # If we ever use multi-layer LSTMs, this logic will have to change.
                    # agent_states[t][0] is h, shape [agents, batch, hidden_dim]
                    self.agent_states[t] = (
                        torch.cat([self.agent_states[t][0], lstm_hidden[0]], 0),
                        torch.cat([self.agent_states[t][1], lstm_hidden[1]], 0),
                    )

            # Do edge updates.
            # Inputs and outputs should look like [batch, edge, Nfeat]
            # Edges each have a to and from node, and agent types and data associated with each.
            # Indicies are computed for gather to batch all edge inputs.
            both_trajectories_t = both_trajectories[:, :, t, :]  # [batch, agent, time, 6]
            trajectory_and_type = torch.cat([both_trajectories_t, agent_type], 2)  # [batch, agent, Nfeat]

            edge_input_module_indicies_i = self.edge_input_module_indicies_i.unsqueeze(0).to(device)
            edge_input_module_indicies_i = edge_input_module_indicies_i.expand(  # [batch, edges, feature_size]
                (batch_size, self.num_edges, self.edge_input_module_indicies_i.shape[1])
            )
            edge_input_module_indicies_j = self.edge_input_module_indicies_j.unsqueeze(0).to(device)
            edge_input_module_indicies_j = edge_input_module_indicies_j.expand(  # [batch, edges, feature_size]
                (batch_size, self.num_edges, self.edge_input_module_indicies_j.shape[1])
            )
            edge_module_indicies_i = self.edge_module_indicies_i.unsqueeze(1).to(device)
            edge_module_indicies_i = edge_module_indicies_i.expand(  # [edges, batch, node_hidden_states]
                (self.num_edges, batch_size, self.edge_module_indicies_i.shape[1])
            )
            edge_module_indicies_j = self.edge_module_indicies_j.unsqueeze(1).to(device)
            edge_module_indicies_j = edge_module_indicies_j.expand(  # [edges, batch, node_hidden_states]
                (self.num_edges, batch_size, self.edge_module_indicies_j.shape[1])
            )
            ###########
            # Calculate edge inputs for input and output node
            trajectory_and_type_i = torch.gather(
                trajectory_and_type, 1, edge_input_module_indicies_i
            )  # [batch, edge, features]
            trajectory_and_type_j = torch.gather(
                trajectory_and_type, 1, edge_input_module_indicies_j
            )  # [batch, edge, features]
            # Edge_input_module should get [both_traj_i, agent_type_i, both_traj_j, agent_type_j]
            # For each edge, this means two gathers, one for agent_i and one for agent_j, and a concat
            processed_edge_inputs = self.edge_input_module(torch.cat([trajectory_and_type_i, trajectory_and_type_j], 2))

            ##########
            #   update edge state given input and output node state and edge inputs
            node_state_i_h = torch.gather(self.agent_states[t][0], 0, edge_module_indicies_i).permute(
                1, 0, 2
            )  # [batch, edge, features]
            node_state_j_h = torch.gather(self.agent_states[t][0], 0, edge_module_indicies_j).permute(
                1, 0, 2
            )  # [batch, edge, features]

            # edge_module should get [node_state_i_h[0], node_state_j_h[0], edge_input]
            self.edge_states[t] = self.edge_module(  # [batch, edge, features]
                torch.cat([node_state_i_h, node_state_j_h, processed_edge_inputs], 2)
            )

            if t == self.temporal_truncated_steps:
                self.edge_states[t] = self.edge_states[t].detach()

                # ToDo(paul.drews): This needs to be fixed to use temporal_truncated_steps.
                # for i in range(self.num_agents):
                #     self.agent_states[i][t] = tuple([x.detach() for x in self.agent_states[i][t]])

        ###########
        # Calculate output given final timestamp node hidden state
        # Get only the h state from the LSTM, [agents, batch, hidden_dim]
        agent_states_h = [i[0] for i in self.agent_states]
        all_agent_states = torch.stack(agent_states_h, 2)  # [agents, batch, time, intermediate_dim]
        res = self.layer_pred(all_agent_states).permute(1, 0, 2, 3)  # [batch, agent, time, intermediate_dim]
        # Make sure our result is not NaN.
        assert not torch.any(torch.isnan(res))

        return res, self.agent_states, {}  # ([batch, agent, time, intermediate_dim], [t][2][agent, batch, hidden_dim])

    def set_num_agents(self, num_agents, num_pastpoints):
        """
        Reset the graph structure to match the number of agents, timepoints
        :param num_agents:
        :param num_pastpoints:
        :return:
        """
        self.num_agents = num_agents
        self.past_time_points = num_pastpoints
        self.agent_states = [None] * self.past_time_points
        self.edge_states = {}
        self.edges = []
        edge_num = 0
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j and self.params["disable_self_edges"]:
                    continue
                self.edges.append((i, j, edge_num))
                edge_num = edge_num + 1

        self.num_edges = (self.num_agents * self.num_agents) - (
            self.num_agents if self.params["disable_self_edges"] else 0
        )
        assert self.num_edges == len(self.edges)

        agent_data_dim = 3 + 3 + self.type_dim  # Two trajectories plus the agent type
        # Create indicies for torch.gather() to batch edge updates, both edge input module and edge module.
        # Indicies of from node on edge.
        self.edge_input_module_indicies_i = torch.empty(self.num_edges, agent_data_dim, dtype=torch.long)
        # Indicies of to node on edge.
        self.edge_input_module_indicies_j = torch.empty(self.num_edges, agent_data_dim, dtype=torch.long)
        for edge_num, edge in enumerate(self.edges):
            for feature in range(agent_data_dim):
                self.edge_input_module_indicies_i[edge_num, feature] = edge[0]
                self.edge_input_module_indicies_j[edge_num, feature] = edge[1]

        self.edge_module_indicies_i = torch.empty(self.num_edges, self.node_hidden_dim, dtype=torch.long)
        self.edge_module_indicies_j = torch.empty(self.num_edges, self.node_hidden_dim, dtype=torch.long)
        for edge_num, edge in enumerate(self.edges):
            for feature in range(self.node_hidden_dim):
                self.edge_module_indicies_i[edge_num, feature] = edge[0]
                self.edge_module_indicies_j[edge_num, feature] = edge[1]

    def save_model(self, data, is_valid):
        # TODO(guy.rosman): Implement using torch.jit.trace.
        pass
