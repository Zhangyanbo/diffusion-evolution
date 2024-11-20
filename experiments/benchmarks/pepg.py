from benchmarks import plot_background, get_obj
from es import PEPG
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from matplotlib.patches import Ellipse
from color_plate import *

# https://www.sciencedirect.com/science/article/pii/S0893608009003220


def PEPG_experiment(obj, num_steps=10, sigma_init=1):
    es = PEPG(
        num_params=2, 
        popsize=512, 
        sigma_init=sigma_init, 
        sigma_decay=0.01**(1/num_steps), 
        elite_ratio=0.1 # Elite ratio can lead to multiple solutions
        )
    
    populations = []
    mus = []
    sigmas = []
    fitnesses = []
    for i in range(num_steps):
        pop = es.ask()
        mus.append(deepcopy(es.mu))
        sigmas.append(deepcopy(es.sigma))
        populations.append(deepcopy(pop))
        fitness = obj(pop)
        fitnesses.append(fitness)
        es.tell(fitness)
    
    populations = torch.from_numpy(np.stack(populations)).float()
    fitnesses = torch.from_numpy(np.stack(fitnesses)).float()

    return es, populations, np.stack(mus), np.stack(sigmas), fitnesses

def prepare_data(trace, arg, fitnesses):
    info = {
        "arguments": arg,
        "trace": trace,
        "fitnesses": fitnesses
    }
    return info

def PEPG_plot(obj, es, mus, sigmas, ax=None, traces=16):
    if ax is None:
        fig, ax = plt.subplots()

    plot_background(obj, ax=ax, title='')
    pop = es.ask()
    plt.scatter(pop[:, 0], pop[:, 1], c=x0_color, alpha=0.5, marker='.', zorder=10, edgecolors='none')
    plt.plot(mus[:, 0], mus[:, 1], '.-', c=traj_color, alpha=1, zorder=0)

    # plot sigma ranges
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        alpha = (i + 1) / len(mus)
        ellipse = Ellipse(
            xy=mu,
            width=sigma[0] * 2,
            height=sigma[1] * 2,
            linewidth=2,
            edgecolor=traj_color,
            facecolor='none',
            alpha=0.5 * alpha ** 0.5,
        )
        ax.add_patch(ellipse)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

def PEPG_benchmark(objs, num_steps, row=0, total_row=4, total_col=5, sigma_init=4, plot=False, **kwargs):
    arg = {
        "num_step": num_steps,
        "sigma_init": sigma_init
    }

    record = dict()

    for i, foo_name in enumerate(objs):
        obj, obj_rescaled = get_obj(foo_name, **kwargs)

        es, traj, mus, sigmas, fitnesses = PEPG_experiment(obj_rescaled, num_steps=num_steps, sigma_init=sigma_init)
        record[foo_name] = prepare_data(traj, arg, fitnesses)
        if plot:
            ax = plt.subplot(total_row, total_col, i + 1 + row * total_col)
            PEPG_plot(obj, es, mus, sigmas, ax=ax)
            if i == 0:
                ax.set_ylabel(f"PEPG")
    
    return record


if __name__ == '__main__':

    objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin"]

    plt.figure(figsize=(12, 3))

    record = PEPG_benchmark(objs, 10, 0, total_row=1, plot=True)
    torch.save(record, './data/pepg.pt')
    plt.tight_layout()

    plt.savefig('./images/PEPG.png')
    plt.close()