import torch

from intent.multiagents.additional_callback import AdditionalModelCallback


class AdditionalModelCallback_NormalizedAgents(AdditionalModelCallback):
    def update_agent_additional_inputs(
        self, batch, agent_additional_inputs: dict, transforms: torch.Tensor, normalizer
    ):
        max_relevant_agents = batch["max_relevant_agents"]

        trajectory_data = agent_additional_inputs["trajectory_data"][..., :2]
        num_agent = trajectory_data.shape[1]

        # [batch, agents, normalized_agents(normalized), timestamps, 2]
        normalized_others_positions = trajectory_data.unsqueeze(2).repeat(1, 1, num_agent, 1, 1)
        for i in range(num_agent):
            i_traj = trajectory_data[:, i : i + 1, :, :2].repeat(1, num_agent, 1, 1)
            # ith agent's position in all other agents frame.
            normalized_others_positions[:, :, i] = normalizer(i_traj, None, transforms)

        agent_additional_inputs["normalized_others_positions"] = normalized_others_positions[:, :max_relevant_agents]
