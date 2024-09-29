# run all experiments step by step

# raw
echo "Running raw experiment..."
python cartpole_raw.py

# latent
echo "Running latent experiment..."
python cartpole_latent.py

echo "Running large experiment..."
python cartpole_large.py

# cmaes
echo "Running CMA-ES experiment..."
python cartpole_cmaes.py

# do visualization
echo "Running visualization..."
python visulization.py