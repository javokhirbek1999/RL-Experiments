import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

def main():
    env_id = "ALE/SpaceInvaders-v5"
    
    # 4 parallel environments balances modern laptop CPU structures perfectly 
    # without causing system lag or thermal throttling.
    env = make_atari_env(env_id, n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)
    
    # --- PPO OPTIMIZATIONS FOR RTX 2050 (4GB VRAM) ---
    model = PPO(
        policy='CnnPolicy',
        env=env,
        verbose=1,
        learning_rate=2.5e-4,       # Standard highly stable Atari learning rate
        
        # 128 steps per env * 4 envs = 512 total steps collected per rollout phase.
        # This keeps the temporary tensor buffers exceptionally small and fast.
        n_steps=128,                
        
        # Batch size of 128 forces the RTX 2050 to process data efficiently 
        # while keeping matrix allocation well inside your 4GB limit.
        batch_size=128,             
        
        # 4 mini-epochs optimizes the data reuse without baking your laptop GPU.
        n_epochs=4,                 
        
        device="cuda"
    )
    
    print("Starting optimized local PPO training on RTX 2050...")
    model.learn(total_timesteps=1_000_000, log_interval=10, progress_bar=True)
    
    model.save("ppo_space_invaders_laptop")
    print("Training complete! Model saved locally.")

if __name__ == '__main__':
    main()
