import torch
import torch.nn as nn
from .kde import KDE


class BayesianEstimator:
    """Bayesian Estimator of the origin points, based on current samples and fitness values."""
    def __init__(self, x: torch.tensor, fitness: torch.tensor, alpha, density='uniform', h=0.1):
        self.x = x
        self.fitness = fitness
        self.alpha = alpha
        self.density_method = density
        self.h = h
        if not density in ['uniform', 'kde']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')

    def append(self, estimator):
        self.x = torch.cat([self.x, estimator.x], dim=0)
        self.fitness = torch.cat([self.fitness, estimator.fitness], dim=0)
    
    def density(self, x):
        if self.density_method == 'uniform':
            return torch.ones(x.shape[0]) / x.shape[0]
        elif self.density_method == 'kde':
            return KDE(x, h=self.h)
    
    @staticmethod
    def norm(x):
        if x.shape[-1] == 1:
            # for some reason, torch.norm become very slow when dim=1, so we use torch.abs instead
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)

    def gaussian_prob(self, x, mu, sigma):
        dist = self.norm(x - mu)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))

    def _estimate(self, x_t, p_x_t):
        # diffusion proability, P = N(x_t; \sqrt{α_t}x,\sqrt{1-α_t})
        mu = self.x * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        p_diffusion = self.gaussian_prob(x_t, mu, sigma)

        # estimate the origin
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.unsqueeze(1) * self.x, dim=0) / (z + 1e-9)

        return origin

    def estimate(self, x_t):
        p_x_t = self.density(x_t)
        origin = torch.vmap(self._estimate, (0, 0))(x_t, p_x_t)
        return origin

    def __call__(self, x_t):
        return self.estimate(x_t)

    def __repr__(self):
        return f'<BayesianEstimator {len(self.x)} samples>'

class LatentBayesianEstimator(BayesianEstimator):
    def __init__(self, x: torch.tensor, latent: torch.tensor, fitness: torch.tensor, alpha, density='uniform', h=0.1):
        super().__init__(x, fitness, alpha, density=density, h=h)
        self.z = latent

    def _estimate(self, z_t, p_z_t):
        # diffusion proability, P = N(x_t; \sqrt{α_t}x,\sqrt{1-α_t})
        mu = self.z * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        p_diffusion = self.gaussian_prob(z_t, mu, sigma)

        # estimate the origin
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_z_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.unsqueeze(1) * self.x, dim=0) / (z + 1e-9)

        return origin
    
    def estimate(self, z_t):
        p_z_t = self.density(self.z)
        origin = torch.vmap(self._estimate, (0, 0))(z_t, p_z_t)
        return origin


def ddim_step(xt, x0, alphas: tuple, noise: float = None):
    """One step of the DDIM algorithm.

    Args:
    - xt: torch.Tensor, shape (n, d), the current samples.
    - x0: torch.Tensor, shape (n, d), the estimated origin.
    - alphas: tuple of two floats, alpha_{t} and alpha_{t-1}.

    Returns:
    - x_next: torch.Tensor, shape (n, d), the next samples.
    """
    alphat, alphatp = alphas
    sigma = ddpm_sigma(alphat, alphatp) * noise
    eps = (xt - (alphat ** 0.5) * x0) / (1.0 - alphat) ** 0.5
    if sigma is None:
        sigma = ddpm_sigma(alphat, alphatp)
    x_next = (alphatp ** 0.5) * x0 + ((1 - alphatp - sigma ** 2) ** 0.5) * \
        eps + sigma * torch.randn_like(x0)
    return x_next

def ddpm_sigma(alphat, alphatp):
    """Compute the default sigma for the DDPM algorithm."""
    return ((1 - alphatp) / (1 - alphat) * (1 - alphat / alphatp)) ** 0.5


class BayesianGenerator:
    """Bayesian Generator for the DDIM algorithm.

    Args:
        density: legacy option for changing the density estimator.
        h: bandwidth for KDE when ``density`` is ``'kde'``. Both rarely used.
    """
    def __init__(self, x, fitness, alpha, density='uniform', h=0.1):
        self.x = x
        if torch.any(fitness < 0):
            raise ValueError('fitness must be non-negative')
        self.fitness = fitness
        self.alpha, self.alpha_past = alpha
        self.estimator = BayesianEstimator(self.x, self.fitness, self.alpha, density=density, h=h)
    
    def generate(self, noise=1.0, return_x0=False):
        x0_est = self.estimator(self.x)
        x_next = ddim_step(self.x, x0_est, (self.alpha, self.alpha_past), noise=noise)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next

    def __call__(self, noise=1.0, return_x0=False):
        return self.generate(noise=noise, return_x0=return_x0)


class LatentBayesianGenerator(BayesianGenerator):
    """Bayesian Generator for the DDIM algorithm."""
    def __init__(self, x, latent, fitness, alpha, density='uniform', h=0.1):
        # density and h are legacy options kept for backward compatibility
        self.x = x
        self.latent = latent
        if torch.any(fitness < 0):
            raise ValueError('fitness must be non-negative')
        self.fitness = fitness
        self.alpha, self.alpha_past = alpha
        self.estimator = LatentBayesianEstimator(self.x, self.latent, self.fitness, self.alpha, density=density, h=h)
    
    def generate(self, noise=1.0, return_x0=False):
        x0_est = self.estimator(self.latent)
        x_next = ddim_step(self.x, x0_est, (self.alpha, self.alpha_past), noise=noise)
        if return_x0:
            return x_next, x0_est
        else:
            return x_next