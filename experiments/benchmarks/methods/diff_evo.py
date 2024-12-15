import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from diffevo import DDIMScheduler, BayesianGenerator, DDIMSchedulerCosine, DDPMScheduler
from .benchmarks import plot_background, get_obj
from .color_plate import *



def experiment(obj, num_pop=256, num_step=100, scaling=4.0, temperatures=None, disable_bar=False, dim=2, scheduler=None):
    
    if scheduler is None:
        scheduler = DDIMSchedulerCosine(num_step=num_step)
    else:
        scheduler = scheduler(num_step=num_step)

    x = torch.randn(num_pop, dim)

    trace = []
    x0_trace = []
    fitnesses = []
    x0_fitness = []

    for t, alpha in tqdm(scheduler, total=num_step-1, disable=disable_bar):
        fitness = obj(x * scaling)
        fitnesses.append(fitness)
        generator = BayesianGenerator(x, fitness, alpha, density='uniform')
        x, x0 = generator(noise=0.1, return_x0=True)
        x0_fit = obj(x0 * scaling)
        x0_fitness.append(x0_fit)
        trace.append(x.clone() * scaling)
        x0_trace.append(x0.clone() * scaling)
    fitness = obj(x * scaling)
    fitnesses.append(fitness)
    x0_fitness.append(x0_fit)
    
    pop = x * scaling
    trace = torch.stack(trace)
    x0_trace = torch.stack(x0_trace)
    fitnesses = torch.stack(fitnesses)
    x0_fitness = torch.stack(x0_fitness)
    return pop, trace, x0_trace, fitnesses, x0_fitness

def make_plot(obj, pop, ax=None, traj=None, x0_trace=None, num_trace=64, title=None):
    plot_background(obj, ax=ax, title=title)

    x0 = x0_trace[-1]
    plt.scatter(x0[:, 0], x0[:, 1], c=x0_color, marker='o', alpha=0.1, zorder=10, edgecolors='none')

    if traj is not None:
        t = traj[:, :num_trace]
        plt.plot(t[:, :, 0], t[:, :, 1], c=traj_color, alpha=0.25, zorder=5)

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

def prepare_data(obj, trace, x0_trace, arg, fitnesses, x0_fitness, benchmark_fitness=None):
    info = {
        "arguments": arg,
        "trace": trace,
        "x0_trace": x0_trace,
        "fitnesses": fitnesses,
        "x0_fitness": x0_fitness,
        "benchmark_fitness": benchmark_fitness
    }
    return info

def DiffEvo_benchmark(objs, num_steps, row=0, total_row=4, total_col=5, num_pop=256, scaling=4.0, plot=False, disable_bar=False, benchmark_temperature=1.0, dim=2, scheduler=None, **kwargs):
    arg = {
        "limit_val": 100,
        "num_pop": num_pop,
        "num_step": num_steps,
        "scaling": scaling,
    }

    shift = row * total_col

    record = dict()

    for i, name in enumerate(objs):
        obj, obj_rescaled = get_obj(name, **kwargs)
        pop, trace, x0_trace, fitnesses, x0_fitness = experiment(
            obj_rescaled, 
            num_pop=num_pop, 
            num_step=num_steps, 
            scaling=scaling, 
            disable_bar=disable_bar,
            dim=dim,
            scheduler=scheduler
            )
        
        _, obj_benchmark = get_obj(name, temperature=benchmark_temperature)
        benchmark_fitness = obj_benchmark(pop)

        if plot:
            ax = plt.subplot(total_row, total_col, i + 1 + shift)
            plot_name = name[0].upper() + name[1:]
            make_plot(obj, pop, ax=ax, traj=trace, x0_trace=x0_trace, title=plot_name)
            if i == 0:
                ax.set_ylabel('DiffEvo')

        arg['limit_val'] = obj.limit_val
        record[name] = prepare_data(obj, trace, x0_trace, arg, fitnesses, x0_fitness, benchmark_fitness)
    
    return record

if __name__ == '__main__':
    # set random seed for reproducibility
    torch.manual_seed(42)

    obj_names = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin"]
    
    plt.figure(figsize=(12, 3))
    record = DiffEvo_benchmark(obj_names, num_steps=100, row=0, total_row=1, num_pop=512, scaling=4, plot=True)
    torch.save(record, './data/diff_evo.pt')

    plt.tight_layout()
    plt.savefig('./images/diff_evo.png')
    plt.close()