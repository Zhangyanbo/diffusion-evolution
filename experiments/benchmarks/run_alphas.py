import matplotlib.pyplot as plt
from methods import *
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import argparse
from diffevo import DDIMScheduler, DDIMSchedulerCosine, DDPMScheduler

# experiment with different alphas on different total steps

obj = "rastrigin"
steps = [10, 25, 50, 100, 250, 500, 1000]

num_steps = 25


def get_records(num_experiments, scheduler, scheduler_name):
    all_records = dict()

    pop_size = 512

    records = []
    print(f"Running DiffEvo_benchmark...")

    for i in tqdm(range(num_experiments)):
        records_per_exp = []
        for step in steps:
            r = DiffEvo_benchmark([obj], num_steps=step, disable_bar=True, limit_val=100, num_pop=pop_size, init_num_pop=pop_size, scheduler=scheduler)
            records_per_exp.append(r)
        records.append(records_per_exp)
    
    result = {
        "scheduler": scheduler_name,
        "num_experiments": num_experiments,
        "records": records
    }
        
    return result


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run optimization benchmarks')
    parser.add_argument('--scheduler', type=str, default='DDIMSchedulerCosine')
    parser.add_argument('--num_experiments', type=int, default=100)
    args = parser.parse_args()

    schedulers = {
        "DDIMSchedulerCosine": DDIMSchedulerCosine,
        "DDPMScheduler": DDPMScheduler,
        "DDIMScheduler": DDIMScheduler
    }

    if args.scheduler not in schedulers:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    records = get_records(args.num_experiments, schedulers[args.scheduler], args.scheduler)
    # save to ./data/schedulers/
    torch.save(records, f'./data/schedulers/{args.scheduler}.pt')
