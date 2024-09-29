import torch


def distance_matrix(x, y):
    """Compute the pairwise distance matrix between x and y.
    
    Args:
        x: (N, d) tensor.
        y: (M, d) tensor.
    Returns:
        (N, M) tensor, the pairwise distance matrix.
    """
    return torch.cdist(x, y)

def KDE(samples, h=0.1):
    """Modified Kernel Density Estimation (KDE) method, which only estimate the density at the given samples.
    
    Args:
        samples: (N, d) tensor, the samples to estimate the density.
        h: float, the bandwidth.
    Returns:
        (N,) tensor, the estimated density at the given samples.
    """
    distances = distance_matrix(samples, samples) # (N, N)
    weights = torch.exp(-(distances ** 2) / (2 * h**2)) # (N,)
    weights = weights.sum(dim=-1)
    return weights / sum(weights) * samples.shape[0]