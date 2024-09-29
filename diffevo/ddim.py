import torch
import numpy as np


class DDIMScheduler:
    """
    DDIMScheduler is a scheduler for the DDIM algorithm.

    Args:
        num_step: int, the number of steps for the DDIM algorithm
    
    Iters:
        t: int, the current time step.
        alpha: float, the current value of alpha.
        alpha_past: float, the previous value of alpha
    
    Example:
        scheduler = DDIMScheduler(num_step=100)
        for t, alpha, alpha_past in scheduler:
            # do something with t, alpha, and alpha_past
    """
    def __init__(self, num_step, power=1, eps=1e-4):
        self.num_step = num_step
        self.power = power
        self.alpha = torch.linspace(1 - eps, (eps*eps) ** (1 / self.power), num_step) ** self.power
        self.index = 0
    
    def __next__(self):
        if self.index >= self.num_step - 1:
            raise StopIteration
        
        t = self.num_step - self.index - 1
        alpha = self.alpha[t]
        alpha_past = self.alpha[t - 1]
        self.index += 1
        return t, (alpha, alpha_past)
    
    def __len__(self):
        return self.num_step - 1
    
    def __iter__(self):
        self.index = 0
        return self


class DDIMSchedulerCosine(DDIMScheduler):
    """
    DDIMSchedulerCosine is a scheduler for the DDIM algorithm with cosine alpha schedule.
    Ref: https://arxiv.org/abs/2102.09672
    
    Args:
        num_step: int, the number of steps for the DDIM algorithm
    
    Iters:
        t: int, the current time step.
        alpha: float, the current value of alpha.
        alpha_past: float, the previous value of alpha
    
    Example:
        scheduler = DDIMSchedulerCosine(num_step=100)
        for t, alpha, alpha_past in scheduler:
            # do something with t, alpha, and alpha_past
    """

    def __init__(self, num_step):
        super().__init__(num_step)
        alpha = torch.cos(torch.linspace(0, torch.pi, num_step)) + 1
        self.alpha = alpha / 2

class DDPMScheduler(DDIMScheduler):
    """
    DDPMScheduler is a scheduler for the DDPM algorithm.
    """
    def __init__(self, num_step, eps=1e-4):
        r"""Approximate the alpha schedule of DDPM.

        Approximation:
            Given β_t = β₀ + γt, and α_t = prod_{s=1}^t (1 - β_s), we can approximate ln(α_t) as follows:

            ln(α_t) ≈ ∫[s=1 to t] ln(1-β₀-γs) ds ≈ T ∫[0 to t/T] (-β₀-γu) du

            --> α_t ≈ exp(-β₀ t - γ t² / 2T)

            Convert t from [1, num_step] to [1/num_step, 1], we got a general form:
            --> α_t ≈ exp(-β t - γ t²)

        Args:
            num_step: int, the number of steps for the DDPM algorithm
            alpha_min: float, the minimum value of alpha
        """
        super().__init__(num_step)
        # ensure alpha[0] = 1 - eps, and alpha[-1] = eps
        beta = ((num_step ** 2) * np.log(1 / (1 - eps)) + np.log(eps)) / (num_step - 1)
        gamma = - num_step * (num_step * np.log(1 / (1-eps)) + np.log(eps)) / (num_step - 1)
        t = torch.linspace(1.0 / num_step, 1.0, num_step)
        self.alpha = torch.exp(-beta * t - gamma * t.square())
        