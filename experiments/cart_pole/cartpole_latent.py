from models import ControllerMLP, DiscreteController
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffevo import LatentBayesianGenerator, RandomProjection, DDIMSchedulerCosine
import gym
from tqdm import tqdm

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def compute_rewards(dim_in, dim_out, dim_hidden, param, n_hidden_layers=1):
    env = gym.make("CartPole-v1", render_mode='rgb_array')

    model = ControllerMLP.from_parameter(dim_in, dim_out, dim_hidden, param, n_hidden_layers=n_hidden_layers)
    controller = DiscreteController(model, env.action_space)

    observation, info = env.reset(seed=42)
    total_reward = 0
    observations = []

    for i in range(500):
        action = controller(torch.from_numpy(observation).float())
        observation, reward, terminated, truncated, info = env.step(action)
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
    scheduler = DDIMSchedulerCosine(num_step=num_step)

    x = torch.randn(population_size, 58)

    reward_history = []
    population = [x * scaling]
    x0_population = [x * scaling]
    observations = []

    random_map = RandomProjection(58, 2, normalize=True)

    for t, alpha in tqdm(scheduler, total=scheduler.num_step-1):
        rewards, obs = compute_rewards_list(4, 2, 8, x * scaling)
        l2 = torch.norm(population[-1], dim=-1) ** 2
        fitness = torch.exp((rewards - rewards.max()) / T - l2 * weight_decay)

        reward_history.append(rewards)

        generator = LatentBayesianGenerator(x, random_map(x).detach(), fitness, alpha)
        x, x0 = generator(noise=noise, return_x0=True)
        population.append(x * scaling)
        x0_population.append(x0 * scaling)
        observations.append(obs)
    
    rewards, obs = compute_rewards_list(4, 2, 8, x * scaling)
    reward_history.append(rewards)
    observations.append(obs)

    reward_history = torch.stack(reward_history)
    population = torch.stack(population)
    x0_population = torch.stack(x0_population)

    return x, reward_history, population, x0_population, observations, random_map

def compute_reward_history(population, dim_in, dim_out, dim_hidden, n_hidden_layers=1):
    rewards = []
    print("Computing reward history for estimated x0 ...")
    for x in tqdm(population):
        reward, obs = compute_rewards_list(dim_in, dim_out, dim_hidden, x, n_hidden_layers=n_hidden_layers)
        rewards.append(reward)

    return torch.stack(rewards)

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

def make_video(para):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env=env, video_folder="./figures/", name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)

    model = ControllerMLP.from_parameter(4, 2, 8, para)
    controller = DiscreteController(model, env.action_space)

    observation, info = env.reset(seed=42)
    rewards = []
    infos = []

    # Start the recorder
    env.start_video_recorder()

    for i in range(500):
        action = controller(torch.from_numpy(observation).float())
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if len(info) > 0:
            infos.append(info)

        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close_video_recorder()
    env.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    num_experiment = 100

    all_reward_history = []

    for i in range(num_experiment):
        x, reward_history, population, x0_population, observations, random_map = experiment(
            num_step=10, 
            population_size=256, 
            T=10, 
            scaling=1, 
            noise=1)
        
        all_reward_history.append(reward_history)
        if i == 0:
            print('saving the data in the first experiment ...')
            torch.save(population, "./data/latent/population.pt")
            torch.save(x0_population, "./data/latent/x0_population.pt")
            torch.save(observations, "./data/latent/observations.pt")
            torch.save(random_map.state_dict(), "./data/latent/random_map.pt")

            make_plot(reward_history)
            plt.savefig("./figures/fitness.png")
            plt.savefig("./figures/fitness.pdf")
            plt.close()

            best_para = population[-1][reward_history[-1].argmax().item()]
            make_video(best_para)

    # save the data
    torch.save(all_reward_history, "./data/latent/reward_history.pt")