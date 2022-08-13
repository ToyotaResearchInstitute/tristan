import unittest

import torch

from radutils.torch.stats import gaussian_mutual_information


class TestStats(unittest.TestCase):
    """Tests Statistics utilities."""

    def test_gaussian_mi(self):
        """Test Gaussian Mutual Information computation."""
        num_datapoints = 100
        num_dims = (5, 10)

        # Ensure test reproducibility
        torch.manual_seed(0)

        rv1 = torch.normal(torch.zeros(num_datapoints, num_dims[0]))
        rv2 = torch.normal(torch.zeros(num_datapoints, num_dims[1]))

        torch.isclose(gaussian_mutual_information(rv1, rv2), gaussian_mutual_information(rv2, rv1))


if __name__ == "__main__":
    unittest.main()
