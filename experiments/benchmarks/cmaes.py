from hades.es import CMAES
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import torch
from benchmarks import plot_background, get_obj
from color_plate import *


def CMAES_experiment(obj, num_steps=10, sigma_init=1):
    es = CMAES(num_params=2, popsize=512, sigma_init=sigma_init, weight_decay=1e-3, inopts={'seed': np.nan}) # ensure reproducibility
    
    populations = []
    fitnesses = []
    mu = [np.zeros(2)]
    cor = [np.eye(2) * sigma_init ** 2]
    for i in range(num_steps):
        pop = es.ask()
        populations.append(pop)
        mu.append(es.cma.mean)
        cor.append(es.cma.C)
        fitness = obj(pop)
        es.tell(fitness)
        fitnesses.append(fitness)

    mu = np.stack(mu)
    cor = np.stack(cor)
    populations = np.stack(populations)
    fitnesses = np.stack(fitnesses)

    populations = torch.from_numpy(populations).float()
    fitnesses = torch.from_numpy(fitnesses).float()

    return es, mu, cor, populations, fitnesses

def CMAES_plot(obj, es, mu, cor, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    plot_background(obj, ax=ax, title='')

    plt.plot(mu[:, 0], mu[:, 1], '.-', color=traj_color, label='Mean', zorder=5)

    population = es.ask()
    plt.scatter(population[:, 0], population[:, 1], c=x0_color, marker='o', alpha=0.25, zorder=10, edgecolors='none')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    for i, (m, c) in enumerate(zip(mu, cor)):
        eigenvalues, eigenvectors = np.linalg.eigh(c)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width, height = np.sqrt(eigenvalues) * 2

        alpha = (i + 1) / len(cor)
        ellipse = Ellipse(
            xy=m,
            width=width * 1,
            height=height * 1,
            angle=np.degrees(angle),
            linewidth=2,
            edgecolor=traj_color,
            facecolor='none',
            alpha=alpha ** 0.5,
            label='Covariance Ellipsoid'
        )
        if i % 2 == 0:
            ax.add_patch(ellipse)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

def prepare_data(obj, trace, arg, fitnesses):
    info = {
        "arguments": arg,
        "trace": trace,
        "fitnesses": fitnesses
    }
    return info

def CMAES_benchmark(objs, num_steps, row=0, total_row=4, total_col=5, sigma_init=4, limit_val=100, plot=False, **kwargs):
    arg = {
        "num_step": num_steps,
        "sigma_init": sigma_init,
        "limit_val": limit_val
    }

    trace = []
    record = dict()

    for i, foo_name in enumerate(objs):
        obj, obj_rescaled = get_obj(foo_name)

        es, mu, cor, trace, fitnesses = CMAES_experiment(obj_rescaled, num_steps=num_steps, sigma_init=sigma_init)
        record[obj.foo_name] = prepare_data(obj, trace, arg, fitnesses)
        if plot:
            ax = plt.subplot(total_row, total_col, i + 1 + row * total_col)
            CMAES_plot(obj, es, mu, cor, ax=ax)
            if i == 0:
                ax.set_ylabel('CMAES')
    
    return record


if __name__ == '__main__':
    import random

    # set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin"]

    plt.figure(figsize=(12, 3))

    record = CMAES_benchmark(objs, num_steps=10, row=0, total_row=1, limit_val=100, plot=True, sigma_init=4)
    torch.save(record, './data/cmaes.pt')

    # remove xy ticks and labels
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[], xlabel='', ylabel='')
    plt.tight_layout()
    plt.savefig('./images/cmaes.png')
    plt.close()