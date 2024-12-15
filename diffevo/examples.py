from torch.distributions import MultivariateNormal
import torch

def two_peak_density(x, mu1=None, mu2=None, std=0.1):
    if mu1 is None:
        mu1 = torch.tensor([-1., -1.])
    if mu2 is None:
        mu2 = torch.tensor([1., 1.])

    # Checking if the input tensor x has shape (2,) and unsqueeze to make it (*N, 2)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    # Covariance matrix for the Gaussian distributions (identity matrix, since it's a standard Gaussian)
    covariance_matrix = torch.eye(2) * (std ** 2)

    # Create two multivariate normal distributions
    dist1 = MultivariateNormal(mu1, covariance_matrix)
    dist2 = MultivariateNormal(mu2, covariance_matrix)

    max_prob = dist1.log_prob(mu1).exp() + dist2.log_prob(mu2).exp()

    # Evaluate the density functions for each distribution and sum them up
    density = dist1.log_prob(x).exp() + dist2.log_prob(x).exp()

    return density / max_prob * 2


def two_peak_density_step(x, mu1=None, mu2=None, std=0.5):
    if mu1 is None:
        mu1 = torch.tensor([-1., -1.])
    if mu2 is None:
        mu2 = torch.tensor([1., 1.])

    # compute the minimal distance to the two peaks
    d1 = torch.norm(x - mu1, dim=-1)
    d2 = torch.norm(x - mu2, dim=-1)
    d = torch.min(d1, d2)

    # if the distance is smaller than the standard deviation, return 1, otherwise 0
    p = (d < std).float()
    p = torch.clamp(p, 1e-9, 1)

    return p