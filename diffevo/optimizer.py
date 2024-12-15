from .ddim import DDIMScheduler
from .generator import BayesianGenerator
from .fitnessmapping import Identity
import torch
from tqdm import tqdm


class DiffEvo:
    """Diffusion evolution algorithm for optimization.

    Args:
        - num_step: int, the number of steps to evolve the population.
        - alpha: str or torch.Tensor, the alpha schedule for the diffusion process.
        - density: str, the mode of the density function, only support 'uniform' and 'kde'.
        - sigma: str, the mode of the sigma, only support 'ddpm' and 'zero'.
        - sigma_scale: float, the scaling factor for the sigma.
        - sample_steps: list of int, the steps to evaluate the fitness.
        - scaling: float, the scaling factor for the population.
        - fitness_mapping: str, the mapping function from fitness to probability, only support 'identity' and 'energy'.
        - temperature: float or list of float, the temperature for the fitness mapping.
        - method: str, the method to estimate the density, only support 'bayesian' and 'nn'.
        - kde_bandwidth: float, the bandwidth for the KDE density estimator.
        - nn: nn.Module, the neural network for the density estimator.

    Methods:
        - step(gt, t): evolve the population by one step.
            outputs:
                - gt: torch.Tensor, the evolved population.
                - density: torch.Tensor, the estimated density of the evolved population.

        - optimize(fit_fn, initial_population, trace=False): optimize the population.
            outputs:
                - population: torch.Tensor, the optimized population.
                - population_trace: list of torch.Tensor, the population trace during the optimization.
                - fitness_count: list of float, the fitness count during the optimization.

    Example:
        ```python
        optimizer = DiffEvo(num_step=100, sigma='ddpm', density='kde')
        sampled, trace, fitness = optimizer.optimize(fitness_function_2d, torch.randn(512, 2), trace=True)
        ```
    """

    def __init__(self,
                 num_step: int = 100,
                 density='kde',
                 noise:float=1.0,
                 scaling: float=1,
                 fitness_mapping=None,
                 kde_bandwidth=0.1):
        self.num_step = num_step

        if not density in ['uniform', 'kde']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')
        self.density = density
        self.kde_bandwidth = kde_bandwidth
        self.scaling = scaling
        self.noise = noise
        if fitness_mapping is None:
            self.fitness_mapping = Identity()
        else:
            self.fitness_mapping = fitness_mapping
        self.scheduler = DDIMScheduler(self.num_step)
    
    def optimize(self, fit_fn, initial_population, trace=False):
        x = initial_population

        fitness_count = []
        if trace:
            population_trace = [initial_population]

        for t, alpha in tqdm(self.scheduler):
            fitness = fit_fn(x * self.scaling)
            generator = BayesianGenerator(x, self.fitness_mapping(fitness), alpha)
            x = generator(noise=self.noise)
            if trace:
                population_trace.append(x)
            fitness_count.append(fitness)
        
        if trace:
            population_trace = torch.stack(population_trace) * self.scaling
        
        if trace:
            return x, population_trace, fitness_count
        else:
            return x