import gymnasium as gym
import ale_py
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

def main():
    env_id = "ALE/SpaceInvaders-v5"
    
    # "human" render mode spins up the game window to visually see the agent
    env = make_atari_env(env_id, n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)
    
    # Load the laptop-trained PPO model weights
    print("Loading local PPO model...")
    model = PPO.load("ppo_space_invaders_laptop", env=env, device="cpu")
    
    print("Starting evaluation loop. Close the window to exit.")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        env.render()
        time.sleep(0.02)  # Adjust delay to change game visual speed

if __name__ == '__main__':
    main()
