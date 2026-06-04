import unittest

import torch
from numpy import cos, deg2rad, sin

from intent.multiagents.log.logging_handlers import SaveErrorStatistics


def create_rotation_matrix(rotation_angle_rad):
    return torch.tensor(
        (
            (cos(rotation_angle_rad), -sin(rotation_angle_rad)),
            (sin(rotation_angle_rad), cos(rotation_angle_rad)),
        )
    ).float()


class TestSaveErrorStatistics(unittest.TestCase):
    @staticmethod
    def test_create_rotations_ego_scene():
        """Verifies that SaveErrorStatistics.create_rotations_ego_scene creates correct transforms

        When applied to a trajectory, ego trajectory should have a normalized orientation and all other trajectories
        should be oriented relative the ego vehicle's local frame
        """
        batch_size, num_agents = 4, 4
        ego_agent = 0
        rotation_step = float(deg2rad(30))

        ego_mask = torch.zeros((batch_size, num_agents))
        ego_mask[:, ego_agent] = 1

        # Generate agent rotations unique per batch
        rotation_angles = [torch.arange(offset, num_agents + offset) * rotation_step for offset in range(batch_size)]
        rotations_scene_local = torch.stack(
            [torch.stack([create_rotation_matrix(angle) for angle in angles_batch]) for angles_batch in rotation_angles]
        ).expand((batch_size, num_agents, 2, 2))

        # Expect transform to be the inverse of the ego rotation
        expected_rotations_ego_scene = torch.linalg.inv(rotations_scene_local[:, ego_agent, ...])

        actual_rotations_ego_scene = SaveErrorStatistics.create_rotations_ego_scene(ego_mask, rotations_scene_local)

        torch.testing.assert_close(actual_rotations_ego_scene, expected_rotations_ego_scene, check_stride=False)


if __name__ == "__main__":
    unittest.main()
