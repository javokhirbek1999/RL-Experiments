import time
import gymnasium as gym
import ale_py
import shimmy

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Explicitly register the ALE environments
gym.register_envs(ale_py)

def main():
    # 1. Create the environment with visual human rendering active
    env_id = "ALE/SpaceInvaders-v5"
    print(f"Loading environment {env_id} in human render mode...")
    
    # Passing render_mode="human" inside wrapper_kwargs opens the GUI game window
    env = make_atari_env(env_id, n_envs=1, seed=42, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    # 2. Load your trained model weights
    model_path = "dqn_space_invaders_colab"
    print(f"Loading trained model from {model_path}.zip...")
    
    # We set device="cpu" for testing to avoid wasting GPU VRAM on simple rendering
    model = DQN.load(model_path, env=env, device="cpu")

    # 3. Run the evaluation game loop
    print("Starting game loop. Press Ctrl+C in your terminal to exit.")
    obs = env.reset()
    
    while True:
        # Predict the best action. deterministic=True removes exploratory random actions.
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment forward using the model's action
        obs, rewards, dones, infos = env.step(action)

        # Slow down emulation slightly so human eyes can comfortably follow the ball
        time.sleep(0.02) 
        
        # The environment automatically resets inside VecFrameStack when dones is true

if __name__ == '__main__':
    main()
