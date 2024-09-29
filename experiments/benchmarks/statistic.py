import matplotlib.pyplot as plt
from cmaes import CMAES_benchmark
from diff_evo import DiffEvo_benchmark
from pepg import PEPG_benchmark
from openes import OpenES_benchmark
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd


objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin"]


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
def point_entropy(record, n=None, scale=10, use_x0=False):
    x = record['trace'][-1]
    if use_x0:
        fitnesses = record['x0_fitness']
    else:
        fitnesses = record['fitnesses']
    
    if n is not None:
        x = get_top_values(fitnesses[-1], x, n)
    return entropy(x, scale).item()

def get_all_records(num_experiments):
    all_records = dict()

    benchmarks = [DiffEvo_benchmark, CMAES_benchmark, OpenES_benchmark, PEPG_benchmark]
    num_steps = [25, 25, 1000, 25]

    for benchmark, step in zip(benchmarks, num_steps):
        name = benchmark.__name__
        records = []

        for i in tqdm(range(num_experiments)):
            r = benchmark(objs, num_steps=step, disable_bar=True, limit_val=100, num_pop=512)
            records.append(r)
        
        all_records[name] = records
    
    return all_records


if __name__ == '__main__':
    num_experiments = 100
    top_k = 64
    cache = False

    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if cache:
        all_records = torch.load('./data/all_records.pt')
    else:
        all_records = get_all_records(num_experiments)
        torch.save(all_records, './data/all_records.pt')

    # save to a .md file

    # add title
    with open('./data/results.md', 'w') as f:
        f.write('# Benchmark Results\n\n')
    
    # entropy
    entropy_table = pd.DataFrame()
    for name, records in all_records.items():
        for k, v in avg_group(point_entropy(all_records[name], n=top_k, use_x0=(name=='DiffEvo_benchmark'))).items():
            entropy_table.loc[k, name.replace("_benchmark", "")] = v
    
    # fitness
    fitness_table = pd.DataFrame()
    for name, records in all_records.items():
        for k, v in avg_group(top_rewards(all_records[name], n=top_k, use_x0=(name=='DiffEvo_benchmark'))).items():
            fitness_table.loc[k, name.replace("_benchmark", "")] = v

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