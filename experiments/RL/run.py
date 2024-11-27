from diffRL import experiment, make_plot, make_video, experiment_cmaes
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import gym


def save_experiment_data(folder, population, x0_population, observations, random_map, reward_history, controller_params):

    # Save experiment data
    torch.save(population[-1].clone(), f"{folder}/population.pt") # only save the last step
    if x0_population is not None:
        torch.save(x0_population[-1].clone(), f"{folder}/x0_population.pt")
    torch.save(observations, f"{folder}/observations.pt") # [num_step, population_size, (t_last, dim_in)]
    if random_map is not None:
        torch.save(random_map.state_dict(), f"{folder}/random_map.pt")

    # Generate and save plots
    make_plot(reward_history)
    plt.savefig(f"{folder}/fitness.png")
    plt.savefig(f"{folder}/fitness.pdf")
    plt.close()

    # Generate video with best parameters
    best_para = population[-1][reward_history[-1].argmax().item()]
    make_video(
        folder,
        best_para, 
        controller_type=controller_params["controller_type"],
        env_name=controller_params["env_name"],
        dim_in=controller_params["dim_in"],
        dim_out=controller_params["dim_out"],
        dim_hidden=controller_params["dim_hidden"],
        n_hidden_layers=controller_params["n_hidden_layers"],
        factor=controller_params["factor"]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL experiment runner')
    parser.add_argument('--method', type=str, default='diff_evo', choices=['diff_evo', 'cmaes'],
                      help='Training method to use')
    parser.add_argument('--env_name', type=str, default='CartPole-v1',
                      help='Environment name')
    parser.add_argument('--latent_dim', type=int, default=None,
                      help='Dimension of latent space')
    parser.add_argument('--dim_in', type=int, default=4,
                      help='Input dimension')
    parser.add_argument('--dim_out', type=int, default=2,
                      help='Output dimension')
    parser.add_argument('--dim_hidden', type=int, default=8,
                      help='Hidden layer dimension')
    parser.add_argument('--n_hidden_layers', type=int, default=1,
                      help='Number of hidden layers')
    parser.add_argument('--factor', type=float, default=1.0,
                      help='Factor parameter')
    parser.add_argument('--controller_type', type=str, default='discrete',
                      help='Type of controller, discrete or continuous')
    parser.add_argument('--T', type=int, default=10,
                      help='Temperature')
    parser.add_argument('--scaling', type=float, default=100,
                      help='Scaling factor')
    parser.add_argument('--num_experiment', type=int, default=1,
                      help='Number of experiments')
    # required arguments
    parser.add_argument('--exp_name', type=str, required=True,
                      help='Experiment name')

    args = parser.parse_args()

    # print the arguments
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # set seed
    torch.manual_seed(42)
    np.random.seed(42)

    all_reward_history = []
    all_endings = []


    controller_params = {
        "dim_in": args.dim_in,
        "dim_out": args.dim_out,
        "dim_hidden": args.dim_hidden,
        "n_hidden_layers": args.n_hidden_layers,
        "factor": args.factor,
        "controller_type": args.controller_type,
        "env_name": args.env_name,
    }

    if args.method == "diff_evo":
        experiment_func = experiment
    elif args.method == "cmaes":
        experiment_func = experiment_cmaes

    folder = f'./results/{str(args.scaling)}/{args.env_name}/{args.exp_name}'
    os.makedirs(folder, exist_ok=True)

    for i in range(args.num_experiment):
        x, reward_history, population, x0_population, observations, random_map, endings = experiment_func(
            num_step=10, 
            population_size=256, 
            T=args.T, 
            scaling=args.scaling, 
            latent_dim=args.latent_dim,
            noise=1,
            **controller_params
        )
        
        all_reward_history.append(reward_history)
        all_endings.append(endings)
        if i == 0:
            save_experiment_data(folder, population, x0_population, observations, random_map, reward_history, controller_params)
    # save the data
    torch.save(all_reward_history, f"{folder}/reward_history.pt")
    torch.save(all_endings, f"{folder}/endings.pt")
    # save all the arguments
    with open(f"{folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)