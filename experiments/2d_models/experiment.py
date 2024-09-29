from two_peaks.experiment import plot_diffusion as plot_diffusion_two_peaks
import torch
import matplotlib.pyplot as plt
from diffevo import DiffEvo, BayesianGenerator, DDIMScheduler
from two_peaks.experiment import two_peak_density
from two_peaks_step.experiment import two_peak_density as two_peak_density_step
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def optimizer(fit_fn, initial_population, scaling=1.0, noise=0.1, num_step=100):
    x = initial_population

    fitness_count = []
    population_trace = [initial_population]
    x0_trace = []
    scheduler = DDIMScheduler(num_step)

    for t, alpha in tqdm(scheduler):
        fitness = fit_fn(x * scaling)
        generator = BayesianGenerator(x, fitness, alpha)
        x, x0 = generator(noise=noise, return_x0=True)
        x0_trace.append(x0)
        population_trace.append(x)
        fitness_count.append(fitness)
    
    population_trace = torch.stack(population_trace) * scaling
    x0_trace = torch.stack(x0_trace) * scaling
    
    return (x * scaling, population_trace, fitness_count), x0_trace, scheduler

def make_plot(alpha, trace, fitnesses, method:str, focus_id=20, row=0, plot_diffusion=plot_diffusion_two_peaks, draw_circle=False):
    time_steps = [20, 45, 70, 95]
    for i, t in enumerate(time_steps):
        plt.subplot(2, 4, i+1+row*4)
        past_ts = time_steps[:i] if i > 0 else []
        alpha_t = alpha[len(alpha) - t - 1]
        plot_diffusion(alpha_t, trace, fitnesses, focus_id=focus_id, T=t, num_sample=100, dt=23, past_ts=past_ts)
        # set aspect ratio to be equal
        plt.gca().set_aspect('equal', adjustable='box')

        if row == 0:
            plt.title(f'$t={100-t}$')
        
        if i == 0:
            plt.ylabel(method)
        
        # add a y = x line
        plt.axline((-5, -5), (5, 5), color='black', linestyle='--', alpha=0.25)

        if draw_circle:
            circle = plt.Circle([-1, -1], 0.5, color='black', fill=False, zorder=2, linestyle='--', alpha=0.5)
            plt.gca().add_artist(circle)
            circle = plt.Circle([1, 1], 0.5, color='black', fill=False, zorder=2, linestyle='--', alpha=0.5)
            plt.gca().add_artist(circle)

def project_to_1d(trace):
    return trace.mean(dim=-1) * (2 ** 0.5)

def plot_distance_histogram(x0_trace, population_trace, t, ax=None, total_step=100, label=True, ylabel=True, title=False, xlabel=True):
    if ax is None:
        ax = plt.gca()
    
    T = total_step - t

    plt.hist(project_to_1d(x0_trace[t]).numpy(), bins=32, density=True, alpha=0.75, range=(-3,3), label='$\hat{x}_0$', color='#E93A01')
    plt.hist(project_to_1d(population_trace[t]).numpy(), bins=32, density=True, alpha=0.5, range=(-3,3), label='$x$', color='#6F6E6E')

    # remove x, y ticks
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0, 1.5)

    # add vertical line at +- sqrt(2)
    plt.axvline(x=np.sqrt(2), color='black', linestyle='--')
    plt.axvline(x=-np.sqrt(2), color='black', linestyle='--')

    if label:
        ax.legend()
    if title:
        ax.set_title(f't = {T}')
    if ylabel:
        ax.set_ylabel('(b) density')

def make_plot_distance_histogram(x0_trace, population_trace, row=0, total_step=100):
    time_steps = [20, 45, 70, 95]
    for i, t in enumerate(time_steps):
        plt.subplot(2, 4, i+1+row*4)
        plot_distance_histogram(x0_trace, population_trace, t, total_step=total_step, label=(i==3), ylabel=(i==0))
        # set aspect ratio to be a standard rectangle
        plt.gca().set_aspect('auto', adjustable='box')

if __name__ == '__main__':
    torch.manual_seed(42)

    x0 = torch.randn(512, 2)
    result_two_peak, x0_trace, scheduler_two_peak = optimizer(two_peak_density, initial_population=x0, scaling=1.5, noise=0.1)

    # save results
    torch.save([result_two_peak, x0_trace, scheduler_two_peak.alpha], './data/two_peak.pt')

    # make plots
    plt.figure(figsize=(8, 4))
    pop, trace, fitnesses = result_two_peak
    make_plot(scheduler_two_peak.alpha, trace, fitnesses, '(a) evolution', row=0, focus_id=7, plot_diffusion=plot_diffusion_two_peaks)
    make_plot_distance_histogram(x0_trace, trace, row=1, total_step=100)

    plt.tight_layout()
    plt.savefig(f'./figures/process.png')
    plt.savefig(f'./figures/process.pdf')
    plt.close()