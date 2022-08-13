import unittest

import pytest
import torch
from numpy import cos, deg2rad, sin

from radutils.torch.torch_utils import (
    apply_2d_coordinate_rotation_transform,
    compute_mean_variances,
    vector_list_to_rotation_matrices,
)


def create_rotation_matrix(angle_radians: float):
    # TODO: Move to utility method
    cos_theta = cos(angle_radians)
    sin_theta = sin(angle_radians)
    return torch.tensor(
        (
            (cos_theta, -sin_theta),
            (sin_theta, cos_theta),
        )
    ).float()


class TestDiscplacementErrors(unittest.TestCase):
    def test_covariance(self):
        """Test correctness of covariance estimation."""
        batch_size = 32
        input_dim = 10
        num_samples = 5000
        X = torch.randn([batch_size, input_dim, num_samples])
        mean, cov = compute_mean_variances(X)
        assert mean.abs().mean() < 1e-1
        assert (cov[:, 0, 0] - 1).abs().mean() < 1e-1
        assert (cov[:, 0, 1]).abs().mean() < 1e-1


class TestApply2dCoordinateRotationTransform:
    # pylint: disable=attribute-defined-outside-init
    def setup_method(self):
        self.single_axis_coordinates = torch.tensor(
            (
                (0.0, 0.5),
                (0.0, 1.4),
                (0.0, -2.8),
                (0.0, -3.5),
            )
        )

        self.multi_axis_diagonal_coordinates = torch.tensor(
            (
                (1.2, 1.2),
                (-2.6, -2.6),
                (3.5, -3.5),
                (-4.7, 4.7),
            )
        )
        self.multi_axis_oblique_coordinates = torch.tensor(
            (
                (1.0, 2.0),
                (-3.0, -1.0),
                (3.0, -3.0),
                (-4.0, 4.0),
            ),
        )

    def test_identity_rotation(self):
        rotations_a_a = torch.eye(2)
        rotation_einsum_prefix = ""

        coordinates_b = torch.stack(
            (self.single_axis_coordinates, self.multi_axis_diagonal_coordinates, self.multi_axis_oblique_coordinates)
        )
        coordinate_einsum_prefix = "ab"

        actual_coordinates_a = apply_2d_coordinate_rotation_transform(
            rotations_a_a,
            coordinates_b,
            result_einsum_prefix=coordinate_einsum_prefix,
            rotation_einsum_prefix=rotation_einsum_prefix,
        )

        # Applying identity rotation should not change results
        torch.testing.assert_close(actual_coordinates_a, coordinates_b, check_stride=False)

    @pytest.mark.parametrize("angle_radians", map(deg2rad, (180, -180)))
    def test_180_rotation(self, angle_radians):
        rotations_a_b = create_rotation_matrix(angle_radians)

        coordinates_b = torch.stack(
            (self.single_axis_coordinates, self.multi_axis_diagonal_coordinates, self.multi_axis_oblique_coordinates)
        )

        actual_coordinates_a = apply_2d_coordinate_rotation_transform(
            rotations_a_b,
            coordinates_b,
            result_einsum_prefix="ab",
            rotation_einsum_prefix="",
        )

        # Applying 180 degree rotation should flip both axes
        torch.testing.assert_close(actual_coordinates_a, -1 * coordinates_b, check_stride=False)

    @pytest.mark.parametrize("angle_radians", map(deg2rad, (0, 30, -30, 45, -45, 90, 180, 200, 360)))
    def test_single_rotation(self, angle_radians):
        rotations_a_b = create_rotation_matrix(angle_radians)
        coordinates_b = torch.cat(
            (self.single_axis_coordinates, self.multi_axis_diagonal_coordinates, self.multi_axis_oblique_coordinates),
            dim=0,
        )

        expected_coordinates_a = torch.stack([rotations_a_b @ coordinate for coordinate in coordinates_b])

        actual_coordinates_a = apply_2d_coordinate_rotation_transform(
            rotations_a_b, coordinates_b, rotation_einsum_prefix="", result_einsum_prefix="a"
        )
        torch.testing.assert_close(actual_coordinates_a, expected_coordinates_a, check_stride=False)

    def test_batch_rotations(self):
        rotations_a_b = torch.stack(
            (
                # Positive Rotations
                create_rotation_matrix(deg2rad(90)),
                create_rotation_matrix(deg2rad(180)),
                create_rotation_matrix(deg2rad(270)),
                create_rotation_matrix(deg2rad(360)),
                # Negative rotations
                create_rotation_matrix(deg2rad(-90)),
                create_rotation_matrix(deg2rad(-180)),
                create_rotation_matrix(deg2rad(-270)),
                create_rotation_matrix(deg2rad(-360)),
            )
        )

        coordinates_b = torch.stack(
            (self.single_axis_coordinates, self.multi_axis_diagonal_coordinates, self.multi_axis_oblique_coordinates)
        )

        # Manually apply each rotation one coordinate at a time
        expected_coordinates_a = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([rotation @ coordinate for coordinate in coordinate_list])
                        for coordinate_list in coordinates_b
                    ]
                )
                for rotation in rotations_a_b
            ]
        )

        actual_coordinates_a = apply_2d_coordinate_rotation_transform(
            rotations_a_b,
            coordinates_b,
            rotation_einsum_prefix="a",
            coordinate_einsum_prefix="bt",
            result_einsum_prefix="abt",
        )
        torch.testing.assert_close(actual_coordinates_a, expected_coordinates_a, check_stride=False)


class TestVectorListToRotationMatrices:
    def test_valid_output(self):
        base_vector = torch.tensor((1.0, 0.0))
        cardinal_vectors = torch.stack(
            (
                base_vector,
                base_vector.flip(-1),
                base_vector * -1.0,
                base_vector.flip(-1) * -1.0,
            )
        )
        non_cardinal_vectors = torch.combinations(torch.arange(10), r=2)

        vectors = torch.cat((cardinal_vectors, non_cardinal_vectors))
        normalized_vectors = vectors / torch.linalg.norm(vectors, dim=-1).unsqueeze(-1)

        rotations_local_scene = vector_list_to_rotation_matrices(vectors)

        vectors_from_rotations = apply_2d_coordinate_rotation_transform(
            rotations_a_b=rotations_local_scene,
            coordinates_b=base_vector,
            result_einsum_prefix="a",
            coordinate_einsum_prefix="",
        )

        # Verify that the rotations preserve the same direction/angle of the source vector
        torch.testing.assert_close(normalized_vectors, vectors_from_rotations, check_stride=False)

        # Verify the structure is a valid rotation
        torch.testing.assert_close(
            # cosines should match
            rotations_local_scene[..., 0, 0],
            rotations_local_scene[..., 1, 1],
            check_stride=False,
        )
        torch.testing.assert_close(
            # sines should be inverses of each other
            rotations_local_scene[..., 0, 1],
            rotations_local_scene[..., 1, 0] * -1.0,
            check_stride=False,
        )
