import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


# 1. Create a Vectorized Environment 
env_id = "FrozenLake-v1"
num_envs = 8
env = make_vec_env(env_id, n_envs=num_envs)

# 2. Define Hyperparameters 
# These are the "secret sauce" for LunarLander. 
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    learning_rate=1e-4,
    tensorboard_log='./ppo_frozen_lake_tensorboard'
)


# 3. Kick off the training
print("Training started...")
model.learn(total_timesteps=500000, progress_bar=True)

# 4. Save the trained brain
model.save('ppo_frozen_lake_v1_expert')
print("Training complete! Model saved.")