"""Acrobat experiments with CMA-ES"""
import torch
import numpy as np
from tqdm import tqdm
from es import CMAES
import os

from acrobat_latent import compute_rewards_list

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def experiment(num_step, T=1, population_size=256, scaling=100, weight_decay=0, sigma_init=1, **args):
    es = CMAES(num_params=83, popsize=population_size, weight_decay=weight_decay, sigma_init=sigma_init, inopts={'seed': np.nan, 'CMA_elitist': 2})

    population = []
    reward_history = []
    observations = []

    for _ in tqdm(range(num_step)):
        pop = es.ask()
        population.append(pop * scaling)

        rewards, obs = compute_rewards_list(6, 3, 8, pop * scaling)
        fitness = rewards

        es.tell(fitness)
        reward_history.append(rewards)
        observations.append(obs)

    population = torch.from_numpy(np.stack(population)).float()
    reward_history = torch.stack(reward_history)

    return pop, reward_history, population, observations, None

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("./data/cmaes", exist_ok=True)

    num_experiment = 10

    all_reward_history = []

    for i in range(num_experiment):
        x, reward_history, population, observations, random_map = experiment(
            10, 
            population_size=256, 
            T=10, 
            scaling=100, 
            sigma_init=1,
            weight_decay=1e-3)
        
        all_reward_history.append(reward_history)
        if i == 0:
            print('saving the data from the first experiment ...')
            torch.save(population, "./data/cmaes/population_cmaes.pt")
            torch.save(observations, "./data/cmaes/observations_cmaes.pt")

    torch.save(all_reward_history, "./data/cmaes/reward_history_cmaes.pt")