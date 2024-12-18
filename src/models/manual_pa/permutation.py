# Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html

from typing import Sequence

import numpy as np
import torch
import wandb
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def sample_gumbel(shape: Sequence[int], eps: float = 1e-20):
    """Samples arbitrary-shaped standard gumbel variables.

    Args:
        shape (Sequence[int]): Shape of the desired Gumbel sample.
        eps (float, optional): For numerical stability. Defaults to 1e-20.

    Returns:
        torch.Tensor: A sample of standard Gumbel random variables.
    """
    u = torch.rand(shape)
    return -torch.log(-torch.log(u + eps) + eps)


def log_sinkhorn(log_alpha: torch.Tensor, n_iter: int, eps: float = 1e-6):
    """By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved matrix with positive entries can be turned
        into a doubly-stochastic matrix(i.e. its rows and columns add up to one) via the successive row and column
        normalization.

    [1] Sinkhorn, Richard and Knopp, Paul. Concerning nonnegative matrices and doubly stochastic matrices. Pacific
        Journal of Mathematics, 1967

    Args:
        log_alpha (torch.Tensor): 2D tensor (a matrix of shape [N, N]) or 3D tensor (a batch of matrices of
            shape = [batch_size, N, N]).
        n_iter (int): Number of sinkhorn iterations (in practice, as little as 20 iterations are needed to
            achieve decent convergence for N~100).

    Returns:
        torch.Tensor: A 2D/3D tensor of close-to-doubly-stochastic matrices.
    """
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        with torch.no_grad():
            beta = log_alpha.exp().sum(1)
            beta_error = torch.abs(beta - torch.ones_like(beta))
            if torch.max(beta_error) <= eps:
                break
    ret = log_alpha.exp()
    # ret = F.gumbel_softmax(ret, tau=0.5, hard=False)
    with torch.no_grad():
        wandb.log(
            {
                "train/beta_error_max": beta_error.max(),
                "train/sinkhorn_second_largest_max": torch.mean(
                    torch.topk(ret, k=2, dim=-1).values[:, 1], dim=-1
                ).max(),
                "train/sinkhorn_iters": _,
            }
        )
    return ret


class GumbelSinkhornPermutation(nn.Module):
    def __init__(self, tau: float = 0.1, n_iter: int = 100):
        """Gumbel-Sinkhorn layer for learning permutations.

        Args:
            tau (float, optional): Temperature parameter, the lower the value for tau the more closely we follow a
                categorical sampling. Defaults to 0.1.
            n_iter (int, optional): Number of Sinkhorn iterations. Defaults to 100.
        """
        super().__init__()
        self.tau = tau
        self.n_iter = n_iter

    def gumbel_sinkhorn(self, alpha: torch.Tensor):
        """Sample a permutation matrix from the Gumbel-Sinkhorn distribution with parameters given by log_alpha and
            temperature tau.

        Args:
            alpha (torch.Tensor): Logarithm of assignment probabilities. In our case this is of
                dimensionality [num_pieces, num_pieces].

        Returns:
            torch.Tensor: A permutation matrix.
        """
        # Sample Gumbel noise.
        # gumbel_noise = sample_gumbel(alpha.shape).to(alpha.device)

        # Apply the Sinkhorn operator!
        # scaled_noised_alpha = (alpha + gumbel_noise) / self.tau
        scaled_noised_alpha = alpha / self.tau
        sampled_perm_mat = log_sinkhorn(scaled_noised_alpha, self.n_iter)
        return sampled_perm_mat

    @torch.no_grad()
    def matching(self, alpha: torch.Tensor):
        """Negate the probability matrix to serve as cost matrix and solve the linear sum assignment problem.

        Args:
            alpha (torch.Tensor): The N x N probability matrix.

        Returns:
            torch.Tensor: The N x N permutation matrix.
        """
        row, col = linear_sum_assignment(-alpha)

        # Create the permutation matrix.
        permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
        return torch.from_numpy(permutation_matrix).float().to(alpha.device)

    def forward(self, alpha: torch.Tensor):
        # During training, we sample from the Gumbel-Sinkhorn distribution.
        if self.training:
            return self.gumbel_sinkhorn(alpha)
        # During eval, we solve the linear assignment problem.
        else:
            return self.matching(alpha.cpu().detach()).to(alpha.device)


if __name__ == "__main__":
    log_alphas = torch.randn(3, 3, requires_grad=True)
    print(log_alphas)
    # alpha = log_sinkhorn(log_alpha)
    # print(alpha)
    # permutation_matrix = matching(alpha)
    # permutation_matrix = gumbel_sinkhorn(log_alpha, 0.01, 20)
    # permutation_matrix.sum().backward()
    # print(log_alpha.grad)
    net = GumbelSinkhornPermutation()
    net.eval()
    P = net(log_alphas)
    # P[0].sum().backward()
    print(log_alphas[0].grad)
    print(P)
