import torch
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .models import ControllerMLP, DiscreteController, ContinuousController
from .utils import normalize_observation

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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

def make_video(folder, para, controller_type="discrete", env_name="CartPole-v1", dim_in=4, dim_out=2, dim_hidden=8, n_hidden_layers=1, factor=1):
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env=env, video_folder=folder, name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)

    model = ControllerMLP.from_parameter(dim_in, dim_out, dim_hidden, para, n_hidden_layers=n_hidden_layers)
    if controller_type == "discrete":
        controller = DiscreteController(model, env.action_space)
    elif controller_type == "continuous":
        controller = ContinuousController(model, env.action_space, factor=factor)
    
    seed = np.random.randint(0, np.iinfo(np.int32).max)
    observation, info = env.reset(seed=seed)
    rewards = []
    infos = []

    # Start the recorder
    env.start_video_recorder()

    for i in range(500):
        action = controller(torch.from_numpy(normalize_observation(observation, env.observation_space)).float())
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if len(info) > 0:
            infos.append(info)

        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close_video_recorder()
    env.close()