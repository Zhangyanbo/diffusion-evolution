import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from statistic import point_entropy, avg_group, std_group

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Constants
experiment_names = ['rosenbrock', 'beale', 'himmelblau', 'ackley', 'rastrigin', 'rastrigin_4d', 'rastrigin_32d', 'rastrigin_256d']
name_display = {
    'rosenbrock': 'Rosenbrock',
    'beale': 'Beale',
    'himmelblau': 'Himmelblau',
    'ackley': 'Ackley',
    'rastrigin': r'Rastrigin$^{2}$',
    'rastrigin_4d': r'Rastrigin$^{4}$',
    'rastrigin_32d': r'Rastrigin$^{32}$',
    'rastrigin_256d': r'Rastrigin$^{256}$'
}
colors = ['#F5851E', '#E93A01', '#6F6E6E', '#800080', '#2B9BBF', '#46B3D5', '#73C5DF', '#94D3E7']

def QD_score_from_trace(X, fitness, grid_size=1):
    """Calculate QD-score from population trace and fitness values"""
    X = torch.clamp(X, -4, 4)
    grid_indices = (X * grid_size).long()
    
    unique_cells = {}
    for idx, fit in zip(grid_indices, fitness):
        cell_key = tuple(idx.tolist())
        if cell_key not in unique_cells or fit > unique_cells[cell_key]:
            unique_cells[cell_key] = fit.item()
    
    return sum(unique_cells.values())

def load_data(folder='./data/temperatures/'):
    files = [f for f in os.listdir(folder) if f.endswith('.pt')]
    return [torch.load(os.path.join(folder, f)) for f in files]

def plot_boxplots(records, ax=None, savefig=True, legend=True):
    # Process data for boxplots
    temperature_data = {}
    for record in records:
        temp = record['temperature']
        if temp not in temperature_data:
            temperature_data[temp] = {}
        for run in record['records']:
            for exp_name in experiment_names:
                results = run[exp_name]
                if exp_name not in temperature_data[temp]:
                    temperature_data[temp][exp_name] = []
                temperature_data[temp][exp_name].append(results['benchmark_fitness'].mean().item())
    if ax is None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
    
    positions = []
    data = []
    spacing = 1.5
    curr_pos = 0
    colors = ['#F5851E', '#E93A01', '#6F6E6E', '#800080', '#2B9BBF', '#46B3D5', '#73C5DF', '#94D3E7']

    # Plot boxplots for each temperature
    for temp in sorted(temperature_data.keys()):
        temp_positions = [curr_pos + i for i in range(len(experiment_names))]
        positions.extend(temp_positions)
        data.extend([temperature_data[temp][exp] for exp in experiment_names])
        curr_pos += len(experiment_names) + spacing

    # Create boxplots with different colors
    bp = ax.boxplot(data, positions=positions, patch_artist=True)

    # Color the boxplots
    num_experiments = len(experiment_names)
    for i in range(len(bp['boxes'])):
        color_idx = i % num_experiments
        bp['boxes'][i].set_facecolor(colors[color_idx])
        # set line color to the same color
        bp['boxes'][i].set_edgecolor(colors[color_idx])
        # set whiskers color
        bp['whiskers'][2*i].set_color(colors[color_idx])
        bp['whiskers'][2*i+1].set_color(colors[color_idx])
        # set cap color
        bp['caps'][2*i].set_color(colors[color_idx])
        bp['caps'][2*i+1].set_color(colors[color_idx])
        # set outlier color
        bp['fliers'][i].set_markeredgecolor(colors[color_idx])
        # set median line color to white
        bp['medians'][i].set_color('white')

    # ax.set_xlabel('Temperature')
    ax.set_ylabel('(a) Final Fitness')

    # Add temperature labels and vertical lines
    temp_positions = []
    curr_pos = 0
    for temp in sorted(temperature_data.keys()):
        center = curr_pos + (len(experiment_names) - 1) / 2
        temp_positions.append(center)
        if curr_pos > 0:  # Add vertical line before each temperature group except first
            plt.axvline(x=curr_pos - spacing/2, color='gray', linestyle='--', alpha=0.5)
        curr_pos += len(experiment_names) + spacing

    ax.set_xticks(temp_positions, [f'$T={t:.1f}$' for t in sorted(temperature_data.keys())])

    # add joint lines for each experiment
    for exp_idx, exp_name in enumerate(experiment_names):
        # Get positions of box centers for this experiment across temperatures
        exp_positions = []
        exp_medians = []
        curr_pos = 0
        for temp in sorted(temperature_data.keys()):
            exp_pos = curr_pos + exp_idx
            exp_positions.append(exp_pos)
            exp_medians.append(np.median(temperature_data[temp][exp_name]))
            curr_pos += len(temperature_data[temp]) + spacing
        ax.plot(exp_positions, exp_medians, color=colors[exp_idx], linestyle='-', alpha=0.5)

    # Add legend for experiments
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[experiment_names.index(exp)]) 
                      for exp in experiment_names]
    if legend:
        ax.legend(legend_elements, [name_display[exp] for exp in experiment_names],
                 loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    # plt.tight_layout()
    if savefig:
        plt.savefig('./figures/temperature_boxplot.png', dpi=300)
        plt.savefig('./figures/temperature_boxplot.pdf', bbox_inches='tight')
        plt.close()

def plot_qd_scores(records, ax=None, savefig=True, legend=True):
    if ax is None:
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
    
    qd_scores = {}
    for record in records:
        temp = record['temperature']
        if temp not in qd_scores:
            qd_scores[temp] = {}
        for run in record['records']:
            for exp_name in experiment_names:
                if exp_name not in qd_scores[temp]:
                    qd_scores[temp][exp_name] = []
                X = run[exp_name]['trace'][-1]
                fitness = run[exp_name]['benchmark_fitness']
                qd_score = QD_score_from_trace(X, fitness)
                qd_scores[temp][exp_name].append(qd_score)

    for exp_name in experiment_names:
        temps = sorted(qd_scores.keys())
        scores = [np.mean(qd_scores[t][exp_name]) for t in temps]
        # normalize scores by the best score and the worst score
        # best_score = max(scores)
        # worst_score = min(scores)
        # scores = [(s - worst_score) / (best_score - worst_score) for s in scores]
        std_scores = [np.std(qd_scores[t][exp_name]) for t in temps]
        
        ax.plot(temps, scores, '.-', label=name_display[exp_name],
                color=colors[experiment_names.index(exp_name)])
        # plt.fill_between(temps, 
        #                 [s - std for s, std in zip(scores, std_scores)],
        #                 [s + std for s, std in zip(scores, std_scores)],
        #                 color=colors[experiment_names.index(exp_name)],
        #                 alpha=0.2)

    ax.set_xlabel('Temperature')
    ax.set_ylabel('(c) QD-Score')
    if legend:
        ax.legend()
    ax.semilogx()
    if savefig:
        plt.savefig('./figures/temperature_qd_scores.png', dpi=300)
        plt.savefig('./figures/temperature_qd_scores.pdf', bbox_inches='tight')
        plt.close()

def plot_entropy(records, ax=None, savefig=True, legend=True):
    if ax is None:
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
    
    # Calculate entropy for each temperature
    entropy_table = []
    temperature_list = []
    for record in records:
        avg_entropy = avg_group(point_entropy(record['records'], n=64))
        temperature_list.append(record['temperature'])
        entropy_table.append(list(avg_entropy.values()))

    entropy_table = np.array(entropy_table)

    # Sort entropy and temperature together
    temperature_list = np.array(temperature_list)
    sorted_indices = np.argsort(temperature_list)
    temperature_list = temperature_list[sorted_indices]
    entropy_table = entropy_table[sorted_indices]

    # Create entropy plot
    for i in range(entropy_table.shape[1]):
        ax.plot(temperature_list, entropy_table[:, i], '.-',
                label=name_display[experiment_names[i]], 
                color=colors[i])
    if legend:
        ax.legend()
    ax.set_xlabel('Temperature')
    ax.set_ylabel('(b) Entropy')
    ax.semilogx()
    if savefig:
        plt.savefig('./figures/temperature_entropy.png', dpi=300)
        plt.savefig('./figures/temperature_entropy.pdf', bbox_inches='tight')
        plt.close()

def combined_plot(records, ax=None, savefig=True):
    """combine boxplot, entropy, and qd-score plots
    Structure:
    - 1st row: boxplot (75%) + legend (25%)
    - 2nd row: entropy + qd-score (each takes half of the second row)
    """
    # Create figure with 2x2 grid
    scale = 0.85
    fig = plt.figure(figsize=(10 * scale, 5 * scale))
    
    # Import gridspec if not already imported
    from matplotlib import gridspec
    
    # Create main gridspec with proper spacing
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Create sub-gridspecs for each row
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[6, 1], wspace=0.05)
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[1, 1], wspace=0.2)
    
    # First row - boxplot on left (75%)
    ax_box = fig.add_subplot(gs_top[0])
    plot_boxplots(records, ax=ax_box, savefig=False, legend=False)
    
    # First row - legend on right (25%)
    ax_legend = fig.add_subplot(gs_top[1])
    ax_legend.axis('off')
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[experiment_names.index(exp)]) 
                      for exp in experiment_names]
    ax_legend.legend(legend_elements, [name_display[exp] for exp in experiment_names],
                    loc='center', fontsize='small')
    
    # Second row - entropy plot on left (50%)
    ax_entropy = fig.add_subplot(gs_bottom[0])
    plot_entropy(records, ax=ax_entropy, savefig=False, legend=False)
    
    # Second row - QD score plot on right (50%)
    ax_qd = fig.add_subplot(gs_bottom[1])
    plot_qd_scores(records, ax=ax_qd, savefig=False, legend=False)

    if savefig:
        plt.savefig('./figures/temperature_combined.png', dpi=300, bbox_inches='tight')
        plt.savefig('./figures/temperature_combined.pdf', bbox_inches='tight')
        plt.close()

def main():
    # Create figures directory if it doesn't exist
    os.makedirs('./figures', exist_ok=True)
    
    # Load data
    records = load_data()
    
    
    
    # Create all plots
    plot_boxplots(records)
    plot_entropy(records)
    plot_qd_scores(records)
    combined_plot(records)

if __name__ == "__main__":
    main()