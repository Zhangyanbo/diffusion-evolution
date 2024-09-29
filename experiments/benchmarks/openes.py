from benchmarks import plot_background, get_obj
import numpy as np
import matplotlib.pyplot as plt
import torch
from color_plate import *


class OpenES:
    def __init__(self, num_params, popsize, sigma_init=1, learning_rate=1e-3, learning_rate_decay=1, sigma_decay=1, momentum=0.9):
        self.num_params = num_params
        self.popsize = popsize
        self.sigma = np.ones(num_params) * sigma_init
        self.learning_rate = learning_rate
        self.sigma_decay = sigma_decay
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum

        self.theta = np.zeros(num_params)
        self.velocity = np.zeros(num_params)
        self.eps = None
    
    def ask(self):
        self.eps = np.random.randn(self.popsize, self.num_params)
        return self.theta + self.sigma * self.eps
    
    def tell(self, fitnesses):
        fitnesses = np.array(fitnesses).reshape(-1, 1)
        dmu = (fitnesses * self.eps).mean(axis=0) / self.sigma #* (self.popsize ** 0.5)
        
        # Apply momentum
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * dmu
        self.theta += self.learning_rate * self.velocity
        
        self.sigma = self.sigma * self.sigma_decay
        self.learning_rate = self.learning_rate * self.learning_rate_decay


def OpenES_experiment(obj, num_steps=100, sigma_init=1):
    es = OpenES(
        num_params=2, 
        popsize=512, 
        sigma_init=sigma_init, 
        learning_rate=1000,
        learning_rate_decay=0.00001**(1/num_steps), 
        sigma_decay=0.01**(1/num_steps), 
        )
    
    populations = []
    fitnesses = []
    mu = []

    for i in range(num_steps):
        pop = es.ask()
        populations.append(pop)

        fitness = obj(pop)
        fitnesses.append(fitness)
        mu.append(es.theta.copy())
        es.tell(fitness)
    
    populations = torch.from_numpy(np.stack(populations)).float()
    fitnesses = torch.from_numpy(np.stack(fitnesses)).float()
    mu = torch.from_numpy(np.stack(mu)).float()
    return es, populations, fitnesses, mu

def prepare_data(trace, arg, fitnesses):
    info = {
        "arguments": arg,
        "trace": trace,
        "fitnesses": fitnesses
    }
    return info

def OpenES_plot(obj, es, traj, mu, ax=None, traces=32):
    if ax is None:
        fig, ax = plt.subplots()

    plot_background(obj, ax=ax, title='')
    pop = traj[-1]
    plt.scatter(pop[:, 0], pop[:, 1], c=x0_color, alpha=0.5, marker='.', zorder=10, edgecolors='none')
    for i, history in enumerate(traj[-10:]):
        alpha = 0.5 * (i / len(traj))
        plt.plot(history[:traces, 0], history[:traces, 1], '.', c='white', alpha=alpha, zorder=5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.plot(mu[::1, 0], mu[::1, 1], '-', color=traj_color, zorder=4, alpha=0.5)

def OpenES_benchmark(objs, num_steps, row=0, total_row=4, total_col=5, sigma_init=4, plot=False, **kwargs):
    arg = {
        "num_step": num_steps,
        "sigma_init": sigma_init
    }

    record = dict()

    for i, foo_name in enumerate(objs):
        obj, obj_rescaled = get_obj(foo_name)

        es, traj, fitnesses, mu = OpenES_experiment(obj_rescaled, num_steps=num_steps, sigma_init=sigma_init)
        record[obj.foo_name] = prepare_data(traj, arg, fitnesses)
        if plot:
            ax = plt.subplot(total_row, total_col, i + 1 + row * total_col)
            OpenES_plot(obj, es, traj, mu, ax=ax)
            if i == 0:
                ax.set_ylabel(f"OpenES")
    
    return record


if __name__ == '__main__':

    objs = ["rosenbrock", "beale", "himmelblau", "ackley", "rastrigin"]

    plt.figure(figsize=(12, 3))

    record = OpenES_benchmark(objs, 1000, 0, total_row=1, plot=True, sigma_init=4)
    torch.save(record, './data/OpenES.pt')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[], xlabel='', ylabel='')
    plt.tight_layout()

    plt.savefig('./images/OpenES.png')
    plt.close()