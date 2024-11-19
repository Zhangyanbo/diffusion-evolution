"""acrobat experiments without latent diffusion"""
import torch
import numpy as np
from diffevo import DDIMScheduler, BayesianGenerator

from models import ControllerMLP, ContinuousController
import matplotlib.pyplot as plt

import gym
from tqdm import tqdm
import os


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def compute_rewards(dim_in, dim_out, dim_hidden, param, n_hidden_layers=1):
    env = gym.make("MountainCarContinuous-v0", render_mode='rgb_array')

    model = ControllerMLP.from_parameter(dim_in, dim_out, dim_hidden, param, n_hidden_layers=n_hidden_layers)
    controller = ContinuousController(model, env.action_space)

    observation, info = env.reset(seed=42)
    total_reward = 0
    observations = []

    for i in range(200):
        action = controller(torch.from_numpy(observation).float())
        observation, reward, terminated, truncated, info = env.step([action])
        observations.append(observation)

        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close()
    return total_reward, torch.from_numpy(np.stack(observations)).float()

def compute_rewards_list(dim_in, dim_out, dim_hidden, params, n_hidden_layers=1):
    rewards = []
    observations = []
    for p in params:
        reward, obs = compute_rewards(dim_in, dim_out, dim_hidden, p, n_hidden_layers=n_hidden_layers)
        rewards.append(reward)
        observations.append(obs)
    return torch.Tensor(rewards), observations



def experiment(num_step, T=1, population_size=512, scaling=0.1, noise=1, weight_decay=0):
    scheduler = DDIMScheduler(num_step=num_step)

    x = torch.randn(population_size, 33)

    reward_history = []
    population = [x * scaling]
    x0_population = [x * scaling]
    observations = []

    for t, alpha in tqdm(scheduler, total=scheduler.num_step-1):
        rewards, obs = compute_rewards_list(2, 1, 8, x * scaling)
        l2 = torch.norm(population[-1], dim=-1) ** 2
        fitness = torch.exp((rewards - rewards.max()) / T - l2 * weight_decay)

        reward_history.append(rewards)

        generator = BayesianGenerator(x, fitness, alpha)
        x, x0 = generator(noise=noise, return_x0=True)
        population.append(x * scaling)
        x0_population.append(x0 * scaling)
        observations.append(obs)
    
    rewards, obs = compute_rewards_list(2, 1, 8, x * scaling)
    reward_history.append(rewards)
    observations.append(obs)

    reward_history = torch.stack(reward_history)
    population = torch.stack(population)
    x0_population = torch.stack(x0_population)

    return x, reward_history, population, x0_population, observations, None

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("./data/raw", exist_ok=True)
    

    num_experiment = 10

    all_reward_history = []

    for i in range(num_experiment):
        x, reward_history, population, x0_population, observations, random_map = experiment(
            num_step=10, 
            population_size=256, 
            T=10, 
            scaling=100, 
            noise=1.0)
    
        all_reward_history.append(reward_history)
        if i == 0:
            print('saving the data in the first experiment ...')
            torch.save(population, "./data/raw/population.pt")
            torch.save(x0_population, "./data/raw/x0_population.pt")
            torch.save(observations, "./data/raw/observations.pt")

    torch.save(all_reward_history, "./data/raw/reward_history.pt")