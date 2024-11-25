from .models import ControllerMLP, DiscreteController, ContinuousController
import torch
import numpy as np
import gym
from tqdm import tqdm
from .es import CMAES
from diffevo import LatentBayesianGenerator, RandomProjection, DDIMSchedulerCosine, BayesianGenerator


def compute_rewards(dim_in, dim_out, dim_hidden, param, env_name, n_hidden_layers=1, controller_type="discrete", factor=1):
    env = gym.make(env_name, render_mode='rgb_array')

    model = ControllerMLP.from_parameter(dim_in, dim_out, dim_hidden, param, n_hidden_layers=n_hidden_layers)

    if controller_type == "discrete":
        controller = DiscreteController(model, env.action_space)
    elif controller_type == "continuous":
        controller = ContinuousController(model, env.action_space, factor=factor)

    observation, info = env.reset(seed=42)
    total_reward = 0
    observations = []
    ending = {'terminated': False, 'truncated': False}

    for i in range(500):
        action = controller(torch.from_numpy(observation).float())
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation)

        total_reward += reward

        if terminated or truncated:
            ending['terminated'] = terminated
            ending['truncated'] = truncated
            observation, info = env.reset()
            break

    env.close()
    observations = torch.from_numpy(np.stack(observations)).float()
    return total_reward, observations, ending

def compute_rewards_list(dim_in, dim_out, dim_hidden, params, env_name, n_hidden_layers=1, controller_type="discrete", factor=1):
    rewards = []
    observations = []
    endings = []
    for p in params:
        reward, obs, ending = compute_rewards(dim_in, dim_out, dim_hidden, p, env_name, n_hidden_layers=n_hidden_layers, controller_type=controller_type, factor=factor)
        rewards.append(reward)
        observations.append(obs)
        endings.append(ending)
    return torch.Tensor(rewards), observations, endings

def calculate_dim(dim_in, dim_out, dim_hidden, n_hidden_layers):
    # calculate the total dimension of the controller
    return (dim_in + 1) * dim_hidden + (dim_hidden + 1) * dim_hidden * (n_hidden_layers-1) + (dim_hidden + 1) * dim_out

def experiment(num_step, T=1, population_size=512, latent_dim=None, scaling=0.1, noise=1, dim_in=4, dim_out=2, dim_hidden=8, n_hidden_layers=1, weight_decay=0, env_name="CartPole-v1", controller_type="discrete", factor=1):
    scheduler = DDIMSchedulerCosine(num_step=num_step)

    dim = calculate_dim(dim_in, dim_out, dim_hidden, n_hidden_layers)
    x = torch.randn(population_size, dim)

    reward_history = []
    population_history = [x * scaling]
    x0_population = [x * scaling]
    observations = []

    if latent_dim is not None:
        random_map = RandomProjection(dim, latent_dim, normalize=True)

    for t, alpha in tqdm(scheduler, total=scheduler.num_step-1):
        rewards, obs, endings = compute_rewards_list(dim_in, dim_out, dim_hidden, x * scaling, env_name, n_hidden_layers=n_hidden_layers, controller_type=controller_type, factor=factor)
        l2 = torch.norm(population_history[-1], dim=-1) ** 2
        fitness = torch.exp((rewards - rewards.max()) / T - l2 * weight_decay)

        reward_history.append(rewards)

        if latent_dim is not None:
            generator = LatentBayesianGenerator(x, random_map(x).detach(), fitness, alpha)
        else:
            generator = BayesianGenerator(x, fitness, alpha)

        x, x0 = generator(noise=noise, return_x0=True)
        population_history.append(x * scaling)
        x0_population.append(x0 * scaling)
        observations.append(obs)
    
    rewards, obs, endings = compute_rewards_list(dim_in, dim_out, dim_hidden, x * scaling, env_name, n_hidden_layers=n_hidden_layers, controller_type=controller_type, factor=factor)
    reward_history.append(rewards)
    observations.append(obs)

    reward_history = torch.stack(reward_history)
    population_history = torch.stack(population_history)
    x0_population = torch.stack(x0_population)

    if latent_dim is not None:
        return x, reward_history, population_history, x0_population, observations, random_map, endings
    else:
        return x, reward_history, population_history, x0_population, observations, None, endings

def experiment_cmaes(num_step, T=1, population_size=512, latent_dim=None, scaling=0.1, noise=1, sigma_init=1, dim_in=4, dim_out=2, dim_hidden=8, n_hidden_layers=1, weight_decay=0, env_name="CartPole-v1", controller_type="discrete", factor=1):
    dim = calculate_dim(dim_in, dim_out, dim_hidden, n_hidden_layers)
    es = CMAES(num_params=dim, popsize=population_size, weight_decay=weight_decay, sigma_init=sigma_init, inopts={'seed': np.nan, 'CMA_elitist': 2})

    population_history = []
    reward_history = []
    observations = []

    for _ in tqdm(range(num_step)):
        x = es.ask()
        population_history.append(x * scaling)

        rewards, obs, endings = compute_rewards_list(dim_in, dim_out, dim_hidden, x * scaling, env_name, n_hidden_layers=n_hidden_layers, controller_type=controller_type, factor=factor)
        fitness = rewards

        es.tell(fitness)
        reward_history.append(rewards)
        observations.append(obs)

    population_history = torch.from_numpy(np.stack(population_history)).float()
    reward_history = torch.stack(reward_history)

    return x, reward_history, population_history, None, observations, None, endings