import torch
import numpy as np
from torch import Tensor
from . import utils


class CMAES:
    """
    Covariance Matrix Adaptation Evolutionary Strategy (CMAES)
    """

    def __init__(self, num_params,
                 sigma_init=1.0,
                 popsize=255,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 inopts=None
                 ):
        """Constructs a CMA-ES solver, based on Hannsen's `cma` module.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param popsize: population size.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param inopts: dict-like CMAOptions, forwarded to cma.CMAEvolutionStrategy constructor).
        :param x0: (Optional) either (i) a single or (ii) several initial guesses for a good solution,
                   defaults to None (initialize via `np.zeros(num_parameters)`).
                   In case (i), the population is seeded with x0.
                   In case (ii), the population is seeded with mean(x0, axis=0) and x0 is subsequently injected.
        """

        self.popsize = popsize

        inopts = inopts or {}
        inopts['popsize'] = self.popsize

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.weight_decay = weight_decay
        self.reg = reg
        self.solutions = None
        self.fitness = None

        # HANDLE INITIAL SOLUTIONS
        inject_solutions = None
        if x0 is None:
            x0 = np.zeros(self.num_params)

        elif isinstance(x0, np.ndarray):
            x0 = np.atleast_2d(x0)
            inject_solutions = x0
            x0 = np.mean(x0, axis=0)

        # INITIALIZE
        import cma
        self.cma = cma.CMAEvolutionStrategy(x0, self.sigma_init, inopts)

        if inject_solutions is not None:
            if len(inject_solutions) == self.popsize:
                self.flush(inject_solutions)
            else:
                self.inject(inject_solutions)  # INJECT POTENTIALLY PROVIDED SOLUTIONS

    def inject(self, solutions=None):
        if solutions is not None:
            self.cma.inject(solutions, force=True)

    def flush(self, solutions):
        self.cma.ary = solutions
        self.solutions = solutions

    def rms_stdev(self):
        sigma = self.cma.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        self.solutions = np.array(self.cma.ask())
        return torch.tensor(self.solutions)

    def tell(self, reward_table_result):
        if not isinstance(reward_table_result, Tensor):
            reward_table = torch.tensor(reward_table_result)
        else:
            reward_table = reward_table_result.clone()

        if self.weight_decay > 0:
            reg = utils.compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        try:
            reward_table = reward_table.numpy()
        except:
            reward_table = reward_table.cpu().numpy()

        self.cma.tell(self.solutions, (-reward_table).tolist())  # convert minimizer to maximizer.

        fitness_argsort = np.argsort(reward_table)[::-1]  # sort in descending order
        self.fitness = reward_table[fitness_argsort]
        self.solutions = self.solutions[fitness_argsort]

    def current_param(self):
        return self.cma.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.cma.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.cma.result
        return r[0], -r[1], -r[1], r[6]