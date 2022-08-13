""" Statitstics and Probability Utilities for PyTorch """
import torch

from radutils.torch.linalg import schur_complement
from radutils.torch.torch_utils import compute_mean_variances


def gaussian_mutual_information(rand_vec1: torch.Tensor, rand_vec2: torch.Tensor):
    """Empirical mutual information between data from different multivariate Gaussian random vectors.

    Parameters
    ----------
    rand_vec1, rand_vec2 : torch.Tensor
        Random vectors for which the mutual information will be computed of shape (num_samples, num_dims) with
        num_samples having to be equal and num_dims being allowed to be different for the two vectors.

    Returns
    -------
    torch.Tensor
        Mutual information between the two Gaussians.
    """
    dim_rv1 = rand_vec1.shape[-1]

    all_data = torch.cat((rand_vec1, rand_vec2), -1)

    # shape [total_num_dim, total_num_dim]
    cov_mat_all = compute_mean_variances(all_data.transpose(0, 1).unsqueeze(0))[1][0]

    cov_mat_rv1 = cov_mat_all[:dim_rv1, :dim_rv1]
    cond_var_rv1_given_rv2 = schur_complement(cov_mat_all, dim_rv1)

    # Our computation is based on the fact that: I(Y;X) = H(Y) - H(Y|X)
    # see https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy
    #
    # Gaussian entropy is 0.5 * ln( (2 * pi * e)^(num_dim) * det(cov_mat) )
    # The terms containing (2 * pi * e)^(num_dim) cancel eachother out.
    # Thus, we can leave them out from entropy_rv1 and cond_netropy below.
    entropy_rv1 = 0.5 * torch.logdet(cov_mat_rv1)
    cond_entropy = 0.5 * torch.logdet(cond_var_rv1_given_rv2)

    result = entropy_rv1 - cond_entropy

    if torch.isnan(result):
        raise ValueError("Invalid mutual information value.")

    return result
