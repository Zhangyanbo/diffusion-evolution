import matplotlib.pyplot as plt
from methods import *
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse


objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin", "rastrigin_4d", "rastrigin_32d", "rastrigin_256d"]

experiments = {
    "diffevo":{
        "method": DiffEvo_benchmark,
        "num_steps": 25
    },
    "latentdiffevo": {
        "method": LatentDiffEvo_benchmark,
        "num_steps": 25
    },
    "cmaes": {
        "method": CMAES_benchmark, 
        "num_steps": 25
    },
    "openes": {
        "method": OpenES_benchmark,
        "num_steps": 1000
    },
    "pepg": {
        "method": PEPG_benchmark,
        "num_steps": 25
    },
    "mapelite": {
        "method": MAPElite_benchmark,
        "num_steps": 25
    }
}


def get_all_records(num_experiments, exp_names):
    all_records = dict()

    pop_size = 512
    methods = [experiments[name]['method'] for name in exp_names]
    num_steps = [experiments[name]['num_steps'] for name in exp_names]

    assert len(methods) == len(num_steps)

    for method, step in zip(methods, num_steps):
        name = method.__name__
        records = []
        print(f"Running {name}...")

        for i in tqdm(range(num_experiments)):
            r = method(objs, num_steps=step, disable_bar=True, limit_val=100, num_pop=pop_size, init_num_pop=pop_size)
            records.append(r)
        
        # save to ./data/records/
        torch.save(records, f'./data/records/{name}.pt')
        all_records[name] = records
    
    return all_records


if __name__ == '__main__':
    num_experiments = 100
    top_k = 64

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run optimization benchmarks')
    parser.add_argument('--experiments', nargs='+', default=['all'],
                       help='List of experiments to run. Use experiment names or "all" for all experiments. '
                            'Valid names: diffevo, latentdiffevo, cmaes, openes, pepg, mapelite')
    args = parser.parse_args()

    # Determine which experiments to run
    if 'all' in args.experiments:
        exp_names = list(experiments.keys())
    else:
        # Validate experiment names
        valid_names = set(experiments.keys())
        exp_names = []
        for name in args.experiments:
            if name not in valid_names:
                raise ValueError(f'Invalid experiment name: {name}. '
                               f'Valid names are: {", ".join(valid_names)}')
            exp_names.append(name)

    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    get_all_records(num_experiments, exp_names)