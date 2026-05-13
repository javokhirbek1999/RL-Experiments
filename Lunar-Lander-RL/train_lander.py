import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# 1. Create a Vectorized Environment (Runs 4 landers in parallel for speed)
env_id = "LunarLander-v3"
num_envs = 4
env = make_vec_env(env_id, n_envs=num_envs)

# 2. Define Hyperparameters 
# These are the "secret sauce" for LunarLander. 
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,           # Number of steps to run per update
    batch_size=64,          # Minibatch size for gradient descent
    n_epochs=10,            # Number of epochs when optimizing the surrogate loss
    gamma=0.999,            # Discount factor (looking far ahead)
    gae_lambda=0.98,        # Factor for trade-off of bias vs variance
    ent_coef=0.01,          # Entropy coefficient (encourages exploration)
    verbose=1,
    learning_rate=5e-4,
    tensorboard_log="./ppo_lunar_tensorboard/"
)

# 3. Kick off the training
# 200,000 steps is usually enough to see a "competent" pilot.
print("Training started... Sit back and relax.")
model.learn(total_timesteps=500000, progress_bar=True)

# 4. Save the trained brain
model.save("ppo_lunar_lander_v3_expert")
print("Training complete! Model saved.")