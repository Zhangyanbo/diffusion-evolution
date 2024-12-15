# Making the figure of the similarity between diffusion and evolution
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from diffevo import DiffEvo
from experiment import two_peak_density
from matplotlib.colors import LinearSegmentedColormap

colors = ["#C8C7C7", "#FF6D3D", "#E93A01"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


def diffuse(num_population, num_step):
    optimizer_ddpm = DiffEvo(num_step=num_step, scaling=1.0)
    x0 = torch.randn(num_population, 2)

    fitness_func = lambda x: two_peak_density(x, std=0.5)

    pop, trace, fitnesses = optimizer_ddpm.optimize(fitness_func, initial_population=x0, trace=True)

    return pop, trace, fitnesses

def make_plot(trace, fitnesses):
    steps = [0, 80, 98]
    fig, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 3, 3))

    for i, t in enumerate(steps):
        ax = axes[i]
        ax.scatter(trace[t, :, 0], trace[t, :, 1], s=1, c=fitnesses[t], cmap=custom_cmap, vmin=0, vmax=1)
        ax.set_title(f'T={t}')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal', adjustable='box')
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = plt.colorbar(ax.collections[0], orientation='vertical')
        cbar.set_label('Fitness')

if __name__ == '__main__':
    torch.manual_seed(42)
    num_population = 512
    num_step = 100

    pop, trace, fitnesses = diffuse(num_population, num_step)
    make_plot(trace, fitnesses)
    plt.tight_layout()
    plt.savefig('./images/diffuse.png')
    plt.savefig('./images/diffuse.pdf')
    plt.close()