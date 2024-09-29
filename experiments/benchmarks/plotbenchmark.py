import matplotlib.pyplot as plt
from cmaes import CMAES_benchmark
from diff_evo import DiffEvo_benchmark
from pepg import PEPG_benchmark
from openes import OpenES_benchmark
import torch
import numpy as np
import random

objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin"]

if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_benchmark = 4
    plt.figure(figsize=(12, 1 + num_benchmark * 2))

    # DiffEvo
    record = DiffEvo_benchmark(objs, num_steps=25, row=0, total_row=num_benchmark, plot=True, num_pop=512)
    torch.save(record, './data/diff_evo.pt')

    # CMAES
    record = CMAES_benchmark(objs, num_steps=25, row=1, total_row=num_benchmark, limit_val=100, plot=True)
    torch.save(record, './data/cmaes.pt')

    # OpenES
    record = OpenES_benchmark(objs, num_steps=1000, row=2, total_row=num_benchmark, plot=True)
    torch.save(record, './data/openes.pt')

    # PEPG
    record = PEPG_benchmark(objs, num_steps=25, row=3, total_row=num_benchmark, plot=True)
    torch.save(record, './data/pepg.pt')

    # save the plot
    plt.tight_layout()
    plt.savefig('./images/benchmark.png')
    plt.savefig('./images/benchmark.pdf')
    plt.close()