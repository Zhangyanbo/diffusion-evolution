import matplotlib.pyplot as plt
from cmaes import CMAES_benchmark
from diff_evo import DiffEvo_benchmark
from pepg import PEPG_benchmark
from openes import OpenES_benchmark
from map_elite import MAPElite_benchmark
from diff_evo_latent import LatentDiffEvo_benchmark
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import pandas as pd


objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin", "rastrigin_4d", "rastrigin_32d", "rastrigin_256d"]


def statistics(func):
    """apply the func to each record of a list of experiments

    Args of decorated function:
        records: list of records of experiments
            structure: experiments[experiment_1[fitness_func_1, ...], ...]
    
    Returns:
        list of statistics of each experiment
            structure: [num_experiments, num_fitness_funcs, *statistics]
    """
    def wrapper(records, *args, **kwargs):
        results = []
        for record in records:
            result_temp = {}
            for fitness_func in record.keys():
                result_temp[fitness_func] = func(record[fitness_func], *args, **kwargs)
            results.append(result_temp)
        return results
    return wrapper

def group(statistics:list):
    results = {}
    for measure in statistics:
        for fitness_func in measure.keys():
            if fitness_func not in results:
                results[fitness_func] = []
            results[fitness_func].append(measure[fitness_func])
    return results

def avg_group(statistics:list):
    grouped = group(statistics)
    for k, v in grouped.items():
        grouped[k] = np.mean(v, axis=0)
    
    return grouped

def std_group(statistics:list):
    grouped = group(statistics)
    for k, v in grouped.items():
        grouped[k] = np.std(v, axis=0)
    
    return grouped

def get_top_values(fitness, x, n):
    idx = np.argsort(-fitness)[:n]
    return x[idx]

@statistics
def top_rewards(record, n=None, use_x0=False):
    if use_x0:
        fitnesses = record['x0_fitness']
    else:
        fitnesses = record['fitnesses']
    
    if n is not None:
        if len(fitnesses.shape) == 1:
            fitnesses = fitnesses.unsqueeze(0)
        fitnesses = fitnesses[-1]
        fitnesses = get_top_values(fitnesses, fitnesses, n)
    else:
        fitnesses = fitnesses[-1]
    return fitnesses.mean().item()

def prob(x, scale=10):
    classification = torch.round(x * scale).long()
    # count the number of points in each class, return [class, num]
    classes, num = torch.unique(classification, return_counts=True, dim=0)
    prob = num.float() / num.sum()
    return prob

def entropy(x, scale=10):
    p = prob(x, scale)
    return torch.sum(-p * torch.log2(p))

@statistics
def point_entropy(record, n=None, scale=10, use_x0=False, name=None):
    if name != 'MAPElite_benchmark':
        x = record['trace'][-1]
        if use_x0:
            fitnesses = record['x0_fitness']
        else:
            fitnesses = record['fitnesses']
    else:
        x = [p for p, r in record['maps'].values()]
        # print(x)
        x = torch.stack(x)
        # print(record['fitnesses'])
        fitnesses = record['fitnesses'].unsqueeze(0)
    
    if n is not None:
        x = get_top_values(fitnesses[-1], x, n)
    return entropy(x, scale).item()

def get_all_records(num_experiments):
    all_records = dict()

    pop_size = 512
    methods = [DiffEvo_benchmark, LatentDiffEvo_benchmark, CMAES_benchmark, OpenES_benchmark, PEPG_benchmark, MAPElite_benchmark]
    # benchmarks = [CMAES_benchmark]
    num_steps = [25, 25, 25, 1000, 25, 25]
    # num_steps = [25]

    assert len(methods) == len(num_steps)

    for method, step in zip(methods, num_steps):
        name = method.__name__
        records = []

        for i in tqdm(range(num_experiments)):
            r = method(objs, num_steps=step, disable_bar=True, limit_val=100, num_pop=pop_size, init_num_pop=pop_size)
            records.append(r)
        
        # save to ./data/records/
        torch.save(records, f'./data/records/{name}.pt')
        all_records[name] = records
    
    return all_records


if __name__ == '__main__':
    num_experiments = 10#100
    top_k = 64
    cache = True

    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if cache:
        all_records = {}
        for name in os.listdir('./data/records/'):
            if name.endswith('.pt'):
                all_records[name.split('.')[0]] = torch.load(f'./data/records/{name}')
    else:
        all_records = get_all_records(num_experiments)

    # save to a .md file

    # add title
    with open('./data/results.md', 'w') as f:
        f.write('# Benchmark Results\n\n')
    
    # entropy
    entropy_table = pd.DataFrame()
    for name, records in all_records.items():
        for k, v in avg_group(point_entropy(all_records[name], n=top_k, use_x0=(name=='DiffEvo_benchmark'), name=name)).items():
            entropy_table.loc[k, name.replace("_benchmark", "")] = v
    
    # save to ./data/entropy_top_<top_k>.csv
    entropy_table.to_csv(f'./data/entropy_top_{top_k}.csv')
    
    # fitness
    fitness_table = pd.DataFrame()
    for name, records in all_records.items():
        for k, v in avg_group(top_rewards(all_records[name], n=top_k, use_x0=(name=='DiffEvo_benchmark'))).items():
            fitness_table.loc[k, name.replace("_benchmark", "")] = v
    
    # save to ./data/fitness_top_<top_k>.csv
    fitness_table.to_csv(f'./data/fitness_top_{top_k}.csv')

    # merge two tables together, each cell is "entropy (fitness)"
    # use string to format
    merged_table = pd.DataFrame()
    for i in range(len(entropy_table)):
        for j in range(len(entropy_table.columns)):
            merged_table.loc[i, j] = f"{entropy_table.iloc[i, j]:.2f} ({fitness_table.iloc[i, j]:.2f})"
    # add row and column index
    merged_table.index = entropy_table.index
    merged_table.columns = entropy_table.columns
    
    with open('./data/results.md', 'a') as f:
        f.write('## Result Table\n\n')
        f.write('Each cell is entropy (fitness)\n\n')
        f.write(merged_table.to_markdown(floatfmt=".2f") + '\n\n')