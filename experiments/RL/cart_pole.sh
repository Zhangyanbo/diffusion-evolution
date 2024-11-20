python run.py --exp_name DiffEvoRaw --method diff_evo --env_name CartPole-v1 --num_experiment 1    
python run.py --exp_name DiffEvoLatent --method diff_evo --env_name CartPole-v1 --latent_dim 2 --num_experiment 1
python run.py --exp_name DiffEvoLargeLatent --method diff_evo --env_name CartPole-v1 --latent_dim 2 --dim_in 4 --dim_out 2 --dim_hidden 128 --n_hidden_layers 2 --num_experiment 1
python run.py --exp_name CMAES --method cmaes --env_name CartPole-v1 --num_experiment 1