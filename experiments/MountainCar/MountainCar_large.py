import torch
import numpy as np
import matplotlib.pyplot as plt
from diffevo import LatentBayesianGenerator, RandomProjection, DDIMSchedulerCosine
from tqdm import tqdm
import os

from experiments.MountainCar.MountainCar_latent import compute_rewards_list

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def experiment(num_step, T=1, population_size=512, scaling=0.1, noise=1, weight_decay=0):
    scheduler = DDIMSchedulerCosine(num_step=num_step)

    x = torch.randn(population_size, 17283)

    reward_history = []
    population = [x * scaling]
    x0_population = [x * scaling]
    observations = []

    random_map = RandomProjection(17283, 2, normalize=True)

    for t, alpha in tqdm(scheduler, total=scheduler.num_step-1):
        rewards, obs = compute_rewards_list(2, 3, 128, x * scaling, n_hidden_layers=2)
        l2 = torch.norm(population[-1], dim=-1) ** 2
        fitness = torch.exp((rewards - rewards.max()) / T - l2 * weight_decay)

        reward_history.append(rewards)

        generator = LatentBayesianGenerator(x, random_map(x).detach(), fitness, alpha)
        x, x0 = generator(noise=noise, return_x0=True)
        x0_population.append(x0 * scaling)
        population.append(x * scaling)
        observations.append(obs)
    
    rewards, obs = compute_rewards_list(2, 3, 128, x * scaling, n_hidden_layers=2)
    reward_history.append(rewards)
    observations.append(obs)

    reward_history = torch.stack(reward_history)
    population = torch.stack(population)
    x0_population = torch.stack(x0_population)

    return x, reward_history, population, x0_population, observations, random_map

def make_plot(reward_history):
    plt.plot(reward_history.median(dim=-1).values, label="median", color='#46B3D5')
    plt.fill_between(
        range(reward_history.size(0)),
        reward_history.quantile(0.1, dim=-1),
        reward_history.quantile(0.9, dim=-1),
        alpha=0.3, label=r"10%-90% quantile", color='#46B3D5')
    plt.xlabel("generation")
    plt.ylabel("rewards")
    plt.legend()

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("./data/large", exist_ok=True)
    os.makedirs("./figures", exist_ok=True)

    num_experiment = 10
    all_reward_history = []

    for i in range(num_experiment):
        x, reward_history, population, x0_population, observations, random_map = experiment(
            10, 
            population_size=256, 
            T=10, 
            scaling=100, 
            noise=1)
        
        all_reward_history.append(reward_history)
        
        if i == 0:
            print('saving the data from the first experiment ...')

            # save the data
            torch.save(population, "./data/large/population_large.pt")
            torch.save(x0_population, "./data/large/x0_population_large.pt")
            torch.save(observations, "./data/large/observations_large.pt")
            torch.save(random_map.state_dict(), "./data/large/random_map_large.pt")

            make_plot(reward_history)
            plt.savefig("./figures/fitness_large.png")
            plt.savefig("./figures/fitness_large.pdf")
            plt.close()

    # save all reward histories
    torch.save(all_reward_history, "./data/large/reward_history_large.pt")