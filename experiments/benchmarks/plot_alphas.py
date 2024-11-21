import torch
import matplotlib.pyplot as plt
import numpy as np
import os

name_table = {
    'DDIMSchedulerCosine': 'Cosine',
    'DDPMScheduler': 'DDPM',
    'DDIMScheduler': 'Linear',
}

def get_avg_fitness(record, idx_exp, idx_step, top_n=1e9):
    # Extract fitness values for rastrigin benchmark
    fitnesses = record['records'][idx_exp][idx_step]['rastrigin']['fitnesses'][-1]
    num_steps = record['records'][idx_exp][idx_step]['rastrigin']['arguments']['num_step']
    
    # Calculate mean of top-n fitnesses
    return fitnesses[:int(min(len(fitnesses), top_n))].mean(), num_steps

def get_step_fitness(record, top_n=1e9):
    total_step_fitness = []
    x = []
    for idx_exp in range(len(record['records'])):
        y = []
        for idx_step in range(len(record['records'][idx_exp])):
            avg, num_steps = get_avg_fitness(record, idx_exp, idx_step, top_n)
            if idx_exp == 0:  # Only need to collect x once
                x.append(num_steps)
            y.append(avg)
        total_step_fitness.append(y)

    total_step_fitness = np.array(total_step_fitness).mean(axis=0)
    return total_step_fitness, x, record['scheduler']

def get_step_fitness_std(record, top_n=1e9):
    total_step_fitness = []
    x = []
    for idx_exp in range(len(record['records'])):
        y = []
        for idx_step in range(len(record['records'][idx_exp])):
            std, num_steps = get_avg_fitness(record, idx_exp, idx_step, top_n)
            if idx_exp == 0:
                x.append(num_steps)
            y.append(std)
        total_step_fitness.append(y)
    total_step_fitness = np.array(total_step_fitness).std(axis=0)
    return total_step_fitness, x, record['scheduler']

def main():
    # Load data
    folder = './data/schedulers'
    schedulers = os.listdir(folder)
    all_records = []

    for scheduler in schedulers:
        records = torch.load(f'{folder}/{scheduler}')
        all_records.append(records)

    # Create plot
    plt.figure()
    for idx_scheduler in range(len(all_records)):
        total_step_fitness, x, scheduler = get_step_fitness(all_records[idx_scheduler], top_n=64)
        total_step_fitness_std, x_std, scheduler_std = get_step_fitness_std(all_records[idx_scheduler])
        plt.errorbar(x, total_step_fitness, yerr=total_step_fitness_std, label=name_table[scheduler])

    # Configure plot
    plt.semilogx()
    plt.legend(loc='lower right')
    plt.xlabel('Number of steps')
    plt.ylabel('Average fitness (top 64 elites)')
    
    # Save and show plot
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/alpha.png', dpi=300)

if __name__ == "__main__":
    main()
