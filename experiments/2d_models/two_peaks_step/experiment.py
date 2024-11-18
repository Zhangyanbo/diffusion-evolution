import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from diffevo import DiffEvo


def two_peak_density(x, mu1=None, mu2=None, std=0.5):
    if mu1 is None:
        mu1 = torch.tensor([-1., -1.])
    if mu2 is None:
        mu2 = torch.tensor([1., 1.])

    # compute the minimal distance to the two peaks
    d1 = torch.norm(x - mu1, dim=-1)
    d2 = torch.norm(x - mu2, dim=-1)
    d = torch.min(d1, d2)

    # if the distance is smaller than the standard deviation, return 1, otherwise 0
    p = (d < std).float()
    p = torch.clamp(p, 1e-9, 1)

    return p

def add_circle(mu, r, alpha=0.1):
    circle = plt.Circle(mu, r, color='#46B3D5', alpha=alpha, zorder=2)
    plt.gca().add_artist(circle)

def plot_diffusion(alpha_t, trace, fitnesses, focus_id, T, num_sample=100, dt=25, past_ts=[]):
    r = 3 * torch.sqrt((1 - alpha_t) / alpha_t)

    # plot trace
    for t in trace.transpose(0, 1)[:num_sample]:
        plt.plot(t[:, 0], t[:, 1], '-', color='#E3E3E3', alpha=0.5)

    # select samples in distance to the focus point, within r
    selected_p = trace[T, focus_id, :]
    mu = selected_p / alpha_t ** 0.5
    d = torch.norm(trace[T, :num_sample] - mu, dim=1)
    inrange = torch.where(d <= r)[0]
    outrange = torch.where(d > r)[0]

    next_t = min(T + dt, len(trace) - 1)
    plt.plot(trace[:next_t, focus_id, 0], trace[:next_t, focus_id, 1], '-', color='#F5851E', zorder=3)

    # plot selected

    size = torch.stack(fitnesses)[T, :num_sample] ** 0.5 * 50 + 1
    size = size / size.max() * 10
    plt.scatter(trace[T, outrange, 0], trace[T, outrange, 1], color='#C6C6C6', s=size[outrange], alpha=1, zorder=10)
    plt.scatter(trace[T, inrange, 0], trace[T, inrange, 1], color='#46B3D5', s=size[inrange] * 1, alpha=1, zorder=9)
    plt.scatter(selected_p[0], selected_p[1], color='black', zorder=10, marker='*')
    for pt in past_ts:
        _sp = trace[pt, focus_id, :]
        plt.scatter(_sp[0], _sp[1], color='gray', zorder=10, marker='*')

    # draw a disk around selected_p
    # filling the circle with transparent blue
    add_circle(mu, r * 3/3, alpha=0.1)
    add_circle(mu, r * 2/3, alpha=0.2)
    add_circle(mu, r * 1/3, alpha=0.3)

    fit = torch.stack(fitnesses)[T].unsqueeze(1)
    x = trace[T]
    d = torch.norm(alpha_t.sqrt() * x - x[focus_id], dim=1).unsqueeze(1)
    pd = torch.exp(-(d ** 2) / (1 - alpha_t) / 2)
    w = fit * pd
    w = w / w.sum()

    x0 = torch.sum(x * w, dim=0)
    eps = (selected_p - alpha_t.sqrt() * x0) / torch.sqrt(1-alpha_t)
    x_next = trace[next_t, focus_id]
    plt.scatter(x0[0], x0[1], color='#E93A01', zorder=13, marker='.')
    # add text
    plt.text(x0[0], x0[1], '$x_0$', fontsize=12, color='black', ha='left', va='top', zorder=13)
    plt.scatter(x_next[0], x_next[1], color='#F5851E', zorder=15, marker='*')

    # draw a dashed arrow from selected_p to x0
    v = x0 - selected_p
    u = v / torch.norm(v)
    plt.arrow(selected_p[0] + 0.2 * u[0], selected_p[1] + 0.2 * u[1],
            v[0] - 0.4 * u[0],
            v[1] - 0.4 * u[1],
            head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=14, alpha=0.25)

    # set limits
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    # remove ticks
    plt.xticks([])
    plt.yticks([])

def make_plot(optimizer, trace, fitnesses, method:str, time_steps = [20, 45, 70, 95]):
    plt.figure(figsize=(8, 2.5))
    for i, t in enumerate(time_steps):
        plt.subplot(1, 4, i+1)
        past_ts = time_steps[:i] if i > 0 else []
        alpha_t = optimizer.scheduler.alpha[optimizer.num_step - t - 1]
        plot_diffusion(alpha_t, trace, fitnesses, focus_id=20, T=t, num_sample=100, dt=23, past_ts=past_ts)
        # set aspect ratio to be equal
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'T={t}')
        ## draw two dashed circles, no filling
        circle = plt.Circle([-1, -1], 0.5, color='black', fill=False, zorder=2, linestyle='--', alpha=0.5)
        plt.gca().add_artist(circle)
        circle = plt.Circle([1, 1], 0.5, color='black', fill=False, zorder=2, linestyle='--', alpha=0.5)
        plt.gca().add_artist(circle)

    plt.tight_layout()
    plt.savefig(f'./figures/process_bayesian_{method}.png')
    plt.savefig(f'./figures/process_bayesian_{method}.pdf')
    plt.close()

    # do a simple scatter plot of the final population
    plt.scatter(trace[-1, :, 0], trace[-1, :, 1], s=1)
    ## draw two circles
    circle = plt.Circle([-1, -1], 0.5, color='black', alpha=0.1, zorder=2)
    plt.gca().add_artist(circle)
    circle = plt.Circle([1, 1], 0.5, color='black', alpha=0.1, zorder=2)
    plt.gca().add_artist(circle)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'./figures/final_population_{method}.png')
    plt.savefig(f'./figures/final_population_{method}.pdf')
    plt.close()


if __name__ == '__main__':
    torch.manual_seed(7)
    optimizer_zero = DiffEvo(num_step=100, scaling=100.5, noise=0.0)
    optimizer_ddpm = DiffEvo(num_step=100, scaling=100.5, noise=0.1)

    x0 = torch.randn(512, 2)

    result_zero = optimizer_zero.optimize(two_peak_density, initial_population=x0, trace=True)
    result_ddpm = optimizer_ddpm.optimize(two_peak_density, initial_population=x0, trace=True)

    pop, trace, fitnesses = result_zero
    make_plot(optimizer_zero, trace, fitnesses, 'zero')
    pop, trace, fitnesses = result_ddpm
    make_plot(optimizer_ddpm, trace, fitnesses, 'ddpm')