import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from gym_aloha.env import AlohaEnv

# Custom callback to handle physics errors
class PhysicsErrorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.unstable_steps = 0
        self.max_unstable_steps = 5  # Maximum number of consecutive unstable steps before stopping

    def _on_step(self):
        # Check if the environment has become unstable
        try:
            # Try to access physics data to check for NaN values
            qpos = self.training_env.get_attr('_env')[0].physics.data.qpos
            if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
                self.unstable_steps += 1
                print(f"Warning: Unstable physics detected (step {self.unstable_steps}/{self.max_unstable_steps})")
                if self.unstable_steps >= self.max_unstable_steps:
                    print("Stopping training due to persistent physics instability")
                    return False
            else:
                self.unstable_steps = 0  # Reset counter if stable
        except Exception as e:
            print(f"Error checking physics stability: {e}")
        return True

# Initialize the environment
env = AlohaEnv(task="insertion", 
               obs_type="pixels",
               render_mode="rgb_array")
check_env(env)  # Check if the environment follows Gym's API

# Initialize the PPO agent with more conservative hyperparameters
model = PPO(
    "MultiInputPolicy",
    env,
    n_steps=1024,  # Reduced from 2048 to be more conservative
    batch_size=64,
    n_epochs=5,    # Reduced from 10 to be more conservative
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=1e-4,  # Reduced from 3e-4 to be more conservative
    clip_range=0.1,      # Reduced from 0.2 to be more conservative
    clip_range_vf=0.1,   # Reduced from 0.2 to be more conservative
    ent_coef=0.005,      # Reduced from 0.01 to be more conservative
    vf_coef=0.5,
    max_grad_norm=0.3,   # Reduced from 0.5 to be more conservative
    use_sde=False,       # Disabled SDE which can cause instability
    target_kl=0.01,
    verbose=1
)

# Create the callback
callback = PhysicsErrorCallback()

TIMESTEPS = 10000  # Reduced for testing, increase for actual training

try:
    # Train the agent with error handling
    print(f"Training for {TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=callback, progress_bar=True)
    print("Learning completed successfully!")
    # Save the trained model
    model.save("ppo_aloha_insertion")
    print("Model saved as ppo_aloha_insertion")
    print("You can now increase the timesteps value for longer training.")
    
except Exception as e:
    print(f"Training interrupted due to error: {e}")
    # Try to save the model if possible
    try:
        print("Attempting to save partial model...")
        model.save("ppo_aloha_insertion_interrupted")
        print("Partial model saved as ppo_aloha_insertion_interrupted")
    except:
        print("Could not save partial model")
