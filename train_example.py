# ###### Fix for error in Colab ######
# # Error:
# #  File "/root/.cache/pypoetry/virtualenvs/gym-aloha-ENReEZCE-py3.11/lib/python3.11/site-packages/matplotlib/__init__.py", line 771, in __setitem__
# #     raise ValueError(f"Key {key}: {ve}") from None
# # ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend; supported values are ['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook', 'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

# import os
# import matplotlib

# # Unset any incorrect backend settings from the environment
# os.environ.pop("MPLBACKEND", None)

# # Reset Matplotlib settings
# matplotlib.rcdefaults()

# # Force a valid backend (Agg in this case)
# matplotlib.use('Agg', force=True)

# # Import other necessary libraries after setting the backend
# import matplotlib.pyplot as plt

# # Verify that the backend is set correctly
# print("Matplotlib Backend:", matplotlib.get_backend())  # Should print 'Agg' 

# ###### Fix for error in Colab ######

import wandb  # Added for experiment tracking
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gym_aloha.env import AlohaEnv

TIMESTEPS = 30000  # Reduced for testing, increase for actual training
CHECKPOINT_FREQ = 5000  # Save model every 200 steps
MODEL_PATH = "ppo_aloha_insertion"

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

# Custom callback to track and log rewards to wandb
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self):
        # Get the reward from the most recent step
        # The infos list contains dictionaries for each environment in a vectorized env
        # Since we're using a single environment, we access the first element
        infos = self.locals.get('infos')
        if infos:
            # Extract reward from the step
            reward = self.locals.get('rewards')[0]
            self.current_episode_reward += reward
            
            # Check if episode is done
            done = self.locals.get('dones')[0]
            if done:
                # Log the episode reward to wandb
                wandb.log({
                    "episode_reward": self.current_episode_reward,
                    "episode": len(self.episode_rewards)
                })
                
                # Store the episode reward and reset for next episode
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                
                # Log running statistics
                if len(self.episode_rewards) > 0:
                    wandb.log({
                        "mean_episode_reward": np.mean(self.episode_rewards),
                        "median_episode_reward": np.median(self.episode_rewards),
                        "min_episode_reward": np.min(self.episode_rewards),
                        "max_episode_reward": np.max(self.episode_rewards)
                    })
        
        return True

# Custom callback to save model checkpoints at regular intervals
class ModelCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        
    def _init_callback(self):
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = f"{self.save_path}/model_step_{self.n_calls}"
            self.model.save(checkpoint_path)
            print(f"Saving model checkpoint at step {self.n_calls} to {checkpoint_path}")
            
            # Log the model to wandb
            wandb.save(f"{checkpoint_path}.zip")
            
            # Log additional info to wandb
            wandb.log({"checkpoint_step": self.n_calls})
            
        return True

# Initialize the environment
env = AlohaEnv(task="insertion", 
               obs_type="pixels",
               render_mode="rgb_array")
check_env(env)  # Check if the environment follows Gym's API

# Initialize Weights & Biases
wandb.init(
    project="gym-aloha",
    config={
        "algorithm": "PPO",
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 1e-4,
        "clip_range": 0.1,
        "clip_range_vf": 0.1,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.3,
        "target_kl": 0.01,
        "timesteps": 1000
    }
)

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

# Create the callbacks
physics_callback = PhysicsErrorCallback()
reward_callback = RewardTrackingCallback()
checkpoint_callback = ModelCheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path="checkpoints")
callback = CallbackList([physics_callback, reward_callback, checkpoint_callback])

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)



try:
    # Train the agent with error handling
    print(f"Training for {TIMESTEPS} timesteps with checkpoints every {CHECKPOINT_FREQ} steps...")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback,
        progress_bar=True,
        tb_log_name="ppo_aloha",
        log_interval=10
    )
    print("Learning completed successfully!")
    
    # Save the final trained model
    model.save(MODEL_PATH)
    wandb.save(f"{MODEL_PATH}.zip")  # Save model to wandb
    print(f"Final model saved as {MODEL_PATH} and uploaded to wandb")
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
