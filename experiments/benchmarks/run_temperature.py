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

method = DiffEvo_benchmark
num_steps = 25


def get_records(num_experiments, temperature):
    all_records = dict()

    pop_size = 512

    name = method.__name__
    records = []
    print(f"Running {name}...")

    for i in tqdm(range(num_experiments)):
        r = method(objs, num_steps=num_steps, disable_bar=True, limit_val=100, num_pop=pop_size, init_num_pop=pop_size, temperature=temperature)
        records.append(r)
    
    result = {
        "temperature": temperature,
        "num_experiments": num_experiments,
        "records": records
    }
        
    return result


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run optimization benchmarks')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_experiments', type=int, default=100)
    args = parser.parse_args()

    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    records = get_records(args.num_experiments, args.temperature)
    # save to ./data/temperatures/
    torch.save(records, f'./data/temperatures/temperature_{args.temperature}.pt')
