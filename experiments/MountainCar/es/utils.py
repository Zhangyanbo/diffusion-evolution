import torch
import numpy as np
from functools import partial


def tensor_to_numpy(t: torch.Tensor):
    t = t.detach()
    try:
        return t.numpy()
    except RuntimeError:  # grad
        return t.detach().numpy()
    except TypeError:  # gpu
        return t.cpu().numpy()


class Optimizer(object):
    def __init__(self, mu, num_params, epsilon=1e-08):
        self.mu = mu
        self.dim = num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, mu, num_params, stepsize, momentum=0.9, epsilon=1e-08):
        Optimizer.__init__(self, mu, num_params, epsilon=epsilon)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, mu, num_params, stepsize, beta1=0.99, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, mu, num_params, epsilon=epsilon)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list, reg='l2'):
    if isinstance(model_param_list, torch.Tensor):
        mean = partial(torch.mean, dim=1)
    else:
        mean = partial(np.mean, axis=1)

    if reg == 'l1':
        return - weight_decay * mean(torch.abs(model_param_list))

    return - weight_decay * mean(model_param_list * model_param_list)


class ScheduledSelectionPressure:
    """ Scheduled Selection Pressure. """
    def __init__(self, selection_pressure, num_steps, rate, mu, offset=1.):
        """ Initialize the ScheduledSelectionPressure.

        :param selection_pressure: float, final selection pressure value
        :param num_steps: int, number of steps for the scheduling
        :param rate: float, rate of the sigmoid function
        :param mu: float, center of the sigmoid function
        """
        self.selection_pressure = selection_pressure
        self.offset = offset
        self.mu = mu
        self.num_steps = num_steps
        self.rate = rate

        self.current_step = 0

    def reset(self):
        self.current_step = 0

    @property
    def scaling_factor(self):
        """ return sigmoid scaling factor based on current step and total steps """
        # alpha = self.current_step / self.num_steps
        x_adjusted = (self.current_step - self.mu) / self.num_steps
        return 1 / (1 + np.exp(-x_adjusted * self.rate))

    def get_value(self):
        value = (self.selection_pressure - self.offset) * self.scaling_factor + self.offset
        self.current_step += 1
        return value

    # override multiplication with numpy array
    def __mul__(self, other):
        return self.get_value() * other

    # override right-side multiplication with numpy array
    def __rmul__(self, other):
        return self.get_value() * other

    # override left-side multiplication with numpy array
    def __lmul__(self, other):
        return self.get_value() * other


def roulette_wheel(f, s=3., eps=1e-12, assume_sorted=False, normalize=False):
    """ Roulette wheel fitness transformation.

    We transform the fitness values f to probabilities p by applying the roulette wheel fitness transformation.
    The roulette wheel fitness transformation is a monotonic transformation that maps the fitness values to
    probabilities. The selection pressure s controls the degree of selection. The higher the selection pressure,
    the more the probabilities are concentrated on the best solutions (s can be positive or negative).

    :param f: torch.Tensor of shape (popsize,), fitness values of the sampled solutions
    :param s: float, selection pressure
    :param eps: float, epsilon to avoid division by zero
    :param assume_sorted: bool, whether to disable sorting of the fitness values and assume that they are already sorted
    :param normalize: bool, whether to normalize the probabilities to sum to 1 (default False, i.e., the sum over
                      the returned scaled probabilities is equal to the sum over the fitness absolute values)
    :return: torch.Tensor of shape (popsize,), indices of the selected solutions
    """
    if not isinstance(f, (torch.Tensor, np.ndarray)):
        f = torch.tensor(f)

    if isinstance(f, torch.Tensor):
        exp = torch.exp
        indices = torch.arange(len(f))
    else:
        exp = np.exp
        indices = np.arange(len(f))

    if not assume_sorted:
        # sort fitness in ascending order
        if isinstance(f, torch.Tensor):
            asc = torch.argsort(f.flatten(), descending=False, dim=0)
            where = torch.where
        else:  # numpy
            asc = f.flatten().argsort()
            where = np.where

        indices = where(asc[None, :] == indices[:, None])[1]  # original order
        f = f[asc]

    if isinstance(f, torch.Tensor):
        total_weight = torch.abs(f).sum()
    else:
        total_weight = np.abs(f).sum()

    fs = (f - f.min()) / (f.max() - f.min() + eps)  # normalize fitness values to [0, 1], and sort
    fs = exp(s*fs)  # apply selection pressure, s can be positive or negative

    if isinstance(f, torch.Tensor):
        fs = fs.cumsum(dim=0)  # compute cumulative sum
    else:
        fs = np.cumsum(fs)  # compute cumulative sum

    fs /= fs.sum()
    if not normalize:
        fs *= total_weight
    return fs[indices]


def parameter_crowding(parameters, weight=1., sharpness=1., similarity_metric="euclidean"):
    from sklearn.metrics.pairwise import pairwise_distances
    parameter_similarity_matrix = pairwise_distances(parameters.reshape(len(parameters), -1), metric=similarity_metric)
    loss = np.exp(-parameter_similarity_matrix * sharpness)
    return loss.mean(axis=-1) * weight
