from .benchmarks import plot_background, get_obj
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from .color_plate import *


def MapEliteExperiment(obj, init_num_pop=100, num_iter=256, sigma_mut=0.1, sigma_init=1, grid_size=10):
    # https://arxiv.org/pdf/1504.04909
    assert num_iter > init_num_pop
    populations = []
    maps = dict()
    def feature_descriptor(x):
        cls = tuple(torch.round(x * grid_size).long().tolist())
        return cls

    # generate initial population
    pop_init = torch.randn(init_num_pop, 2) * sigma_init
    rewards = obj(pop_init)
    for p, r in zip(pop_init, rewards):
        cls = feature_descriptor(p)
        if cls not in maps:
            maps[cls] = (p, r)
            populations.append(p)
        elif r > maps[cls][1]:
            maps[cls] = (p, r)
            populations.append(p)
    # iterate
    for i in range(num_iter - init_num_pop):
        # random select a population to mutate
        idx = np.random.randint(0, len(maps))
        p_old = list(maps.values())[idx][0]
        p_new = p_old + torch.randn(2) * sigma_mut
        r_new = obj(p_new.unsqueeze(0)).squeeze(0)
        cls = feature_descriptor(p_new)
        if cls not in maps:
            maps[cls] = (p_new, r_new)
            populations.append(p_new)
        elif r_new > maps[cls][1]:
            maps[cls] = (p_new, r_new)
            populations.append(p_new)
    
    populations = torch.stack(populations)
    fitnesses = torch.stack([r for p, r in maps.values()])
    return populations, maps, fitnesses

def MAPElite_plot(obj, maps, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    plot_background(obj, ax=ax, title='')
    pop_elite = torch.stack([p for p, r in maps.values()])
    rewards = torch.stack([r for p, r in maps.values()])

    plt.scatter(pop_elite[:, 0], pop_elite[:, 1], c=x0_color, alpha=0.8, marker='.', zorder=10, edgecolors='none', s=(rewards + 0.1)*100)

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

def prepare_data(trace, arg, fitnesses, maps):
    info = {
        "arguments": arg,
        "trace": trace,
        "fitnesses": fitnesses,
        "maps": maps
    }
    return info

def MAPElite_benchmark(objs, num_steps, row=0, grid_size=1, sigma_mut=0.5, total_row=4, total_col=5, sigma_init=4, plot=False, **kwargs):
    init_num_pop = 256
    arg = {
        "num_step": num_steps,
        "init_num_pop": init_num_pop,
        "sigma_init": sigma_init,
        "sigma_mut": sigma_mut,
        "grid_size": grid_size
    }

    record = dict()

    for i, foo_name in enumerate(objs):
        obj, obj_rescaled = get_obj(foo_name, **kwargs)

        # es, traj, mus, sigmas, fitnesses = PEPG_experiment(obj_rescaled, num_steps=num_steps, sigma_init=sigma_init)
        populations, maps, fitnesses = MapEliteExperiment(obj_rescaled, 
            init_num_pop=init_num_pop, 
            num_iter=num_steps*init_num_pop, 
            sigma_mut=sigma_mut, 
            sigma_init=sigma_init, 
            grid_size=grid_size
        )
        record[foo_name] = prepare_data(populations, arg, fitnesses, maps)
        if plot:
            ax = plt.subplot(total_row, total_col, i + 1 + row * total_col)
            MAPElite_plot(obj, maps, ax=ax)
            if i == 0:
                ax.set_ylabel(f"MAP-Elite")
    
    return record

if __name__ == '__main__':

    objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin"]

    plt.figure(figsize=(12, 3))

    record = MAPElite_benchmark(objs, 10, 0, total_row=1, plot=True)
    torch.save(record, './data/map_elite.pt')
    plt.tight_layout()

    plt.savefig('./images/MAPElite.png')
    plt.close()