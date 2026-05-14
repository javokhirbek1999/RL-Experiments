import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


# 1. Define the Wrapper correctly
class HumanoidGaitWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = np.zeros(4)

    def step(self, action):
        self.last_action = action
        return super().step(action)

    def reward(self, reward):
        # Reduced penalty so it actually dares to move
        torque_penalty = 0.0005 * np.sum(np.square(self.last_action))
        
        # We only apply the penalty if the reward isn't a massive "Fall" penalty
        # This keeps the agent from being double-punished for falling
        if reward > -50:
            return reward - torque_penalty
        return reward

# 1. Create a Vectorized Environment (Runs 4 Walkers in parallel for speed)
env_id = "BipedalWalker-v3"
num_envs = 4
env = make_vec_env(env_id=env_id, n_envs=num_envs, wrapper_class=HumanoidGaitWrapper)

# 2. Define Hyperparameters
model = PPO(
    policy='MlpPolicy',
    env=env,
    n_steps=2048,    # Number of steps to run per update
    batch_size=128,   # Minibatch size for gradient descent
    n_epochs=20,     # Number of epochs when optimizing the surrogate loss
    gamma=0.999,     # Discount factor (looking far ahead)
    gae_lambda=0.95, # Factor for trade-off of bias vs variance
    ent_coef=0.05,   # Entropy coefficient (encourages exploration)
    verbose=1,
    learning_rate=3e-4,
    tensorboard_log="./ppo_bipedal_tensorboard"
)


# 3. Kick off the training
print("Training started...")
model.learn(total_timesteps=5_000_000, progress_bar=True)

# 4. Save the trained brain
model.save("ppo_bipedal_walker_v3_5M__with_wrapper")
print("Training completed! Model saved.")
