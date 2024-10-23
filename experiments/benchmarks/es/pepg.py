import torch
import numpy as np
from torch import Tensor
from . import utils


class PEPG:
    '''
    Extension of PEPG with bells and whistles.

    From HADES package, author: Benedikt Hartl
    '''
    def __init__(self, num_params,
                 sigma_init=1.0,
                 sigma_alpha=0.20,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 sigma_max_change=0.2,
                 learning_rate=0.01,
                 learning_rate_decay=0.9999,
                 learning_rate_limit=0.01,
                 elite_ratio=0,
                 popsize=256,
                 average_baseline=True,
                 weight_decay=0.01,
                 reg='l2',
                 rank_fitness=True,
                 forget_best=True,
                 x0=None,
                 ):  #
        """ Constructs a `PEPG` solver instance.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param sigma_alpha: learning rate for standard deviation.
        :param sigma_decay: anneal standard deviation.
        :param sigma_limit: stop annealing if less than this.
        :param sigma_max_change: clips adaptive sigma to 20%.
        :param learning_rate: learning rate for standard deviation.
        :param learning_rate_decay: annealing the learning rate.
        :param learning_rate_limit: stop annealing learning rate.
        :param elite_ratio: if > 0, then ignore learning_rate.
        :param popsize: population size.
        :param average_baseline: set baseline to average of batch.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param rank_fitness: use rank rather than fitness numbers.
        :param forget_best: don't keep the historical best solution.
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.batch_size = int(self.popsize / 2)
        else:
            assert (self.popsize & 1), "Population size must be odd"
            self.batch_size = int((self.popsize - 1) / 2)

        # option to use greedy es method to select next mu, rather than using drift param
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.use_elite = False
        if self.elite_popsize > 0:
            self.use_elite = True

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)

        # BH: ADDING option to start from prior solution
        self.mu = np.zeros(self.num_params) if x0 is None else np.asarray(x0)  # np.zeros(self.num_params)
        self.best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)
        self.curr_best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)

        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.reg = reg
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = utils.Adam(mu=self.best_mu, num_params=num_params, stepsize=learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            # first population is mu, then positive epsilon, then negative epsilon
            epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            reward_table = utils.compute_centered_ranks(reward_table)

        if self.weight_decay > 0:
            reg = utils.compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline

        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_popsize]
        else:
            idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        if (best_reward > b or self.average_baseline):
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # short hand
        epsilon = self.epsilon
        sigma = self.sigma

        # update the mean

        # move mean to the average of the best idx means
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0)
        else:
            rT = (reward[:self.batch_size] - reward[self.batch_size:])
            change_mu = np.dot(rT, epsilon)
            self.optimizer.stepsize = self.learning_rate
            update_ratio = self.optimizer.update(-change_mu)  # adam, rmsprop, momentum, etc.
            # self.mu += (change_mu * self.learning_rate) # normal SGD method

        # adaptive sigma
        # normalization
        if (self.sigma_alpha > 0):
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
            reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

            # adjust sigma according to the adaptive sigma calculation
            # for stability, don't let sigma move more than 10% of orig value
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
            self.sigma += change_sigma

        if (self.sigma_decay < 1):
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def flush(self, solutions):
        self.solutions = solutions

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)
