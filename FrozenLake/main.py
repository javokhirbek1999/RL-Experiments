import gymnasium as gym
from stable_baselines3 import PPO

# Load the trained model
model = PPO.load("ppo_frozen_lake_v1_expert")

# Create a human-viewable environment
env = gym.make('FrozenLake-v1', render_mode='human')

obs, info = env.reset()

while True:
    # The model predicts the best action based on the observation
    action, _states = model.predict(obs, deterministic=True)
    action = action.item()  # Extracts the raw integer from the array
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()