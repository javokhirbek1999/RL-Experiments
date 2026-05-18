import gymnasium as gym
from stable_baselines3 import PPO


# Load the trained model
model = PPO.load('ppo_taxi_v3_expert')

# Create a human-viewable environment
env = gym.make('Taxi-v3', render_mode='human')

obs, info = env.reset()

while True:
    # The model predicts the best action based on the observation
    action, _states = model.predict(obs, deterministic=True)
    action = action.item() # Extracts the raw integer from the array
    obs, reward, terminated, truncated, info = env.step(action=action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()