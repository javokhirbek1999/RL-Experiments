import gymnasium as gym
import ale_py 
import shimmy 
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

def main():
    env_id = "ALE/SpaceInvaders-v5"
    
    # 1. n_envs=2 provides the optimal CPU-to-GPU throughput balance on modern laptops
    env = make_atari_env(env_id, n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)

    # 2. Optimized DQN Hyperparameters for RTX 2050
    model = DQN(
        policy='CnnPolicy',
        env=env,
        verbose=1,
        learning_rate=1e-4,
        
        # --- GPU SPEED OPTIMIZATIONS ---
        buffer_size=50000,          # Reduced slightly to free VRAM for massive batch sizes
        batch_size=512,             # Maximize tensor cores (multiples of 64/128/256/512)
        train_freq=256,             # Collect more experience between training steps
        gradient_steps=128,         # Massive sequential batch updates per training cycle
        # -------------------------------
        
        learning_starts=10000,     # Pre-fills the buffer quickly before optimization starts
        target_update_interval=1000,
        device="cuda"
    )

    print("Starting optimized GPU training...")
    model.learn(total_timesteps=1_000_000, log_interval=10, progress_bar=True)
    
    model.save("dqn_space_invaders_v5_fast")
    print("Training complete! Model saved.")

if __name__ == '__main__':
    main()
