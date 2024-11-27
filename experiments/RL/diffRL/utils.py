import numpy as np

def normalize_observation(observation, observation_space, extreme_threshold=1e3):
    # Replace inf/-inf with threshold values
    low = np.where(observation_space.low < -extreme_threshold, -1, observation_space.low)
    high = np.where(observation_space.high > extreme_threshold, 1, observation_space.high)
    
    # Normalize to [-1, 1] range
    rescaled = 2 * (observation - low) / (high - low) - 1
    return rescaled * np.sqrt(3) # scale to unit variance