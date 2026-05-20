import gymnasium as gym
import panda_gym, time
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. Configuration
env_id = "PandaReachDense-v3"
model_name = "a2c-PandaReachDense-v3"
stats_path = "vec_normalize.pkl"

# 2. Create the evaluation environment with render_mode="human"
# render_mode="human" is what enables the graphics window
eval_env = gym.make(env_id, render_mode="human")

# 3. Wrap it in VecNormalize and load the saved statistics
# This is CRITICAL. If you don't load the stats, the agent will receive 
# observations that it doesn't recognize (wrong scaling).
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecNormalize.load(stats_path, eval_env)
eval_env.training = False  # Disable training/updating of stats
eval_env.norm_reward = False # Turn off reward normalization for testing

# 4. Load the model
model = A2C.load(model_name)

# 5. Run the loop
obs = eval_env.reset()
done = False

print("Starting evaluation with graphics...")

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)

    time.sleep(0.1)
    
    if dones:
        obs = eval_env.reset()