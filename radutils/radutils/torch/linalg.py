""" Additional Linear Algebra Utilities not provided in PyTorch """
import torch


def mean_log_determinant(tensor: torch.Tensor, diagonal_scalar: float = 1e-5) -> torch.Tensor:
    """Compute average tensor log determinant with diagonal term

    Parameters
    ----------
    tensor: torch.Tensor
        (batch_size, N,N) tensor of matrices
    diagonal_scalar: float
        The scalar to add to the diagonal to stabilize log computations.

    Returns
    -------
    torch.Tensor
        The mean of the log-determinant of the matrices.
    """
    tensor_w_diagonal = (
        tensor
        + torch.eye(
            tensor.shape[-1],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        * diagonal_scalar
    )
    return tensor_w_diagonal.det().log().mean()


def schur_complement(mat: torch.Tensor, cut_off_dim: int):
    """Computes the Schur Complement

    The Schur complement for a matrix

    | A B |
    | C D |

    is given by A - B * D^(-1) * C.

    Parameters
    ----------
    mat : torch.Tensor
        The original matrix of shape (num_dim, num_dim)
    cut_off_dim : int
        Cut off dimension for obtaining matrices A, B, C, D assuming A to be of shape (cut_off_dim, cut_off_dim).

    Returns
    -------
    torch.Tensor
        Resulting schur complement of shape (cut_off_dim, cut_off_dim)
    """

    mat_a = mat[:cut_off_dim, :cut_off_dim]
    mat_b = mat[:cut_off_dim, cut_off_dim:]
    mat_c = mat[cut_off_dim:, :cut_off_dim]
    mat_d = mat[cut_off_dim:, cut_off_dim:]

    result = mat_a - torch.mm(mat_b, torch.mm(torch.linalg.inv(mat_d), mat_c))
    return result
