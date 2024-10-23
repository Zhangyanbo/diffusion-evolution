import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from diffevo import RandomProjection
from matplotlib.ticker import LogLocator, LogFormatter

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# change default font size
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 10

# colors = ["#C8C7C7", "#E93A01"]
colors = ['#efefef', '#f6d8cd', '#f9c0ab', '#faa98b', '#f8906b', '#f5774c', '#f05c2c', '#e93a01']
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


def cartpole_plot(angles, positions, box_size=1, y0_shift=0, pole_length=1, cart_size=0.1, ang_scale=2.4, max_alpha=1, decay=10, color='black'):
    total_time = len(angles)
    x0 = torch.linspace(0, total_time * box_size, total_time)
    x0 = x0 + positions
    y0 = x0 * 0 + y0_shift

    x1 = x0 + torch.sin(angles * ang_scale) * pole_length
    y1 = y0 + torch.cos(angles * ang_scale) * pole_length

    alpha = 1
    i = len(x0) - 1
    plt.arrow(x0[i], y0[i], x1[i] - x0[i], y1[i] - y0[i], head_width=0.0, head_length=0.0, alpha=alpha * max_alpha, color=color)
    # add a line to represent the cart
    plt.plot([x0[i]-cart_size,x0[i]+cart_size], [y0[i], y0[i]], color=color, alpha=alpha * max_alpha)

def plot_cartpole(observations, generations, rewards, ax=None, box_size=1.5, box_size_y=2, dt=25, color_bar=False):
    if ax is None:
        ax = plt.gca()

    for g, t in enumerate(generations):
        ax.axhline(g * 2, color='black', alpha=0.05)
        for i in range(len(observations[0])):
            c = custom_cmap(rewards[t][i] / 500)
            ang = observations[t][i][::dt, 2]
            pos = observations[t][i][::dt, 0] / 4.8
            cartpole_plot(ang, pos, max_alpha=0.5, decay=2, y0_shift=g * box_size_y, box_size=box_size, color=c)

    x = np.arange(0, 501, 50)
    x_corr = x / dt * box_size
    ax.set_xticks(x_corr, x)
    ax.set_yticks(np.arange(0, len(generations) * box_size_y, box_size_y), generations+1)
    ax.set_ylim(-0.5, None)
    ax.set_xlabel('time steps')
    ax.set_ylabel('generation')

    if color_bar:
        # Adding the horizontal color bar inside the plot using inset_axes
        cbar_ax = inset_axes(ax, width="20%", height="3%", loc='lower right',
                            bbox_to_anchor=(0.05, 0.15, 0.9, 0.95),
                            bbox_transform=ax.transAxes, borderpad=0)
        
        # Correctly referencing the figure associated with ax
        cbar = ax.figure.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), cax=cbar_ax, orientation='horizontal')

        # Remove color bar ticks
        cbar.ax.set_xticks([])
        cbar.ax.set_yticks([])

        # Set color bar label
        cbar.set_label('reward')

def prepare_reward(rewards):
    # rewards.shape = [num_experiment, num_generation, num_population]
    # merge each experiment into one for each generation
    if isinstance(rewards, list):
        rewards = torch.stack(rewards)

    rewards = rewards.permute(1, 0, 2).reshape(rewards.shape[1], -1)
    return rewards

def range_plot(x, color=None, label=None):
    x = prepare_reward(x)
    center = x.quantile(0.5, dim=-1)
    lower = x.quantile(0.25, dim=-1)
    upper = x.quantile(0.75, dim=-1)
    X = np.arange(len(center)) + 1
    plt.plot(X, center, color=color, label=label)
    plt.fill_between(X, lower, upper, alpha=0.25, color=color, edgecolor='none')


def reward_compare_plot(*rewards, labels=None, colors=None, ax=None):
    if ax is None:
        ax = plt.gca()
    for i, w in enumerate(rewards):
        range_plot(w, color=colors[i] if colors else f'C{i}', label=labels[i] if labels else None)

    ax.axhline(500, color='gray', linestyle='--',
                label='max reward', alpha=0.5)
    ax.legend(fontsize='small')
    ax.set_ylim(5, 570)
    # ax.set_xlim(None, len(rewards[0]))
    ax.set_xlabel('generation')
    ax.set_ylabel('reward')
    
    # set x-axis as reversed log scale
    ax.set_yscale('log')
    
    major_ticks = [10, 100, 300, 500]
    plt.yticks(major_ticks, [f'{tick}' for tick in major_ticks])

    # Set the minor ticks locator and formatter
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    ax.yaxis.set_minor_formatter(LogFormatter(base=10.0, labelOnlyBase=False))


def latent_plot(z, ax, color=None, alpha=1, label=None, zorder=1):
    ax.scatter(z[:, 0], z[:, 1], zorder=zorder, marker='o', color=color, alpha=alpha, label=label, edgecolors='none')

def compare_latent_plot(pop, pop_raw, pop_cmaes, random_map, pop_large, random_map_large, ax=None):
    if ax is None:
        ax = plt.gca()
    latent_plot(random_map(pop).detach()[-1], ax, color='#E93A01', label='latent diffusion evolution', alpha=0.5)
    latent_plot(random_map(pop_raw).detach()[-1], ax, color='#46B3D5', alpha=0.1, label='diffusion evolution')
    latent_plot(random_map(pop_cmaes).detach()[-1], ax, color='#6F6E6E', alpha=0.5, label='CMA-ES')
    latent_plot(random_map_large(pop_large).detach()[-1], ax, color='#F5851E', alpha=0.25, label='latent DiffEvo (high-d)')

    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Create custom legend markers with alpha=1 using Line2D
    latent_line = Line2D([0], [0], marker='.', color='#E93A01', linestyle='None', markersize=10, label='latent DiffEvo')
    raw_line = Line2D([0], [0], marker='.', color='#46B3D5', linestyle='None', markersize=10, label='DiffEvo')
    cmaes_line = Line2D([0], [0], marker='.', color='#6F6E6E', linestyle='None', markersize=10, label='CMA-ES')
    large_line = Line2D([0], [0], marker='.', color='#F5851E', linestyle='None', markersize=10, label='latent DiffEvo (high-d)')
    # Set the legend with the custom lines
    # ax.legend(handles=[raw_line, latent_line, large_line, cmaes_line])

def draw_cartpole_demo(ax):
    ax.axhline(y=0, color='gray', linestyle='--')
    # ax.axvline(x=0, color='gray', linestyle='--')

    def draw_cart_and_pole(ax, center=(0, 0), angle=0.25, alpha=1):
        width=1
        height=0.25
        length=1
        # add a black rectangle representing the cart
        ax.add_patch(plt.Rectangle((center[0]-width/2, center[1]-height/2), width, height, color='black', alpha=alpha, linewidth=0))

        # draw a pole with the given angle
        ax.plot(
            [center[0], center[0]+length*np.sin(angle)], 
            [center[1], center[1]+length*np.cos(angle)], 
            color='#CC9965',
            linewidth=4,
            alpha=alpha)

    # Call the sub-function
    draw_cart_and_pole(ax) # main plot
    # add left and right arrows
    ax.arrow(0.1, -0.25, 1, 0, head_width=0.1, head_length=0.1, color='black', zorder=10)
    ax.arrow(-0.1, -0.25, -1, 0, head_width=0.1, head_length=0.1, color='black', zorder=10)
    for i in range(10):
        # random x and angle
        x = np.random.uniform(-3, 3)
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        draw_cart_and_pole(ax, center=(x, 0), angle=angle, alpha=0.1)
    
    # remove y-axis
    ax.set_yticks([])
    ax.set_xlabel('$x$')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 1.7)


if __name__ == '__main__':
    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    obs = torch.load('./data/latent/observations.pt') # [time, num_pop, [T, 4]]

    pop_latent = torch.load('./data/latent/population.pt')
    rewards_latent = torch.load('./data/latent/reward_history.pt')
    random_map = RandomProjection(58, 2, normalize=True)
    random_map.load_state_dict(torch.load('./data/latent/random_map.pt'))

    rewards_raw = torch.load('./data/raw/reward_history.pt')
    pop_raw = torch.load('./data/raw/population.pt')

    rewards_cmaes = torch.load('./data/cmaes/reward_history_cmaes.pt')
    pop_cmaes = torch.load('./data/cmaes/population_cmaes.pt')

    rewards_large = torch.load('./data/large/reward_history_large.pt')
    pop_large = torch.load('./data/large/population_large.pt')
    random_map_large = RandomProjection(17410, 2, normalize=True)
    random_map_large.load_state_dict(torch.load('./data/large/random_map_large.pt'))

    # generations = np.array([1, 40, 70, 90, 100])-1
    generations = np.array([2, 4, 6, 8, 10])-1

    # Create a figure
    fig = plt.figure(figsize=(10, 6))

    # Create a GridSpec with 2 rows and 3 columns
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # Top plot, merged across both columns
    ax1 = fig.add_subplot(gs[0, :])
    plot_cartpole(obs, generations, rewards_latent[0], ax=ax1)
    ax1.set_title('(a) evolution process')

    # Bottom left plot
    ax2 = fig.add_subplot(gs[1, 0])
    reward_compare_plot(rewards_raw, rewards_latent, rewards_large, rewards_cmaes,
                        labels=['DiffEvo',
                                'latent DiffEvo', 'latent DiffEvo (high-d)', 'CMA-ES'],
                        colors=['#46B3D5', '#E93A01', '#F5851E', '#6F6E6E'], ax=ax2)
    ax2.set_title('(b) reward comparison')

    # Bottom middle plot
    ax3 = fig.add_subplot(gs[1, 1])
    compare_latent_plot(pop_latent, pop_raw, pop_cmaes, random_map, pop_large, random_map_large, ax=ax3)
    ax3.set_title('(c) latent space comparison')

    # Bottom right plot
    ax4 = fig.add_subplot(gs[1, 2])
    ## a placeholder plot
    draw_cartpole_demo(ax4)
    ax4.set_title('(d) cart-pole system')

    # add margin between the subplots
    plt.tight_layout()

    plt.savefig('./figures/cartpole.png', bbox_inches='tight')
    # save as pdf with transparent background
    plt.savefig('./figures/cartpole.pdf', bbox_inches='tight', transparent=True)
    plt.close()