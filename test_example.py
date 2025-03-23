import imageio
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gym_aloha.env import AlohaEnv

# Load the trained model
model_path = "ppo_aloha_insertion.zip"
model = PPO.load(model_path)
print(f"Loaded model from {model_path}")

# Initialize the environment with the same parameters as in training
env = AlohaEnv(task="insertion", 
               obs_type="pixels",
               render_mode="rgb_array")

# Reset the environment
observation, info = env.reset(seed=42)  # Using a seed for reproducibility
frames = []

# Number of steps to run the simulation
n_steps = 500
print(f"Running model for {n_steps} steps...")

# Run the model
for i in range(n_steps):
    print(f"Step {i+1}/{n_steps}")
    
    # Get the model's action based on the current observation
    action, _ = model.predict(observation, deterministic=True)
    
    # Take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the current frame
    image = env.render()
    frames.append(image)
    
    # Check if the episode is done
    if terminated or truncated:
        print("Episode finished early")
        if info.get("is_success", False):
            print("Task completed successfully!")
        break

# Close the environment
env.close()

# Save the video
output_file = "test_example.mp4"
print(f"Saving video to {output_file}...")
imageio.mimsave(output_file, np.stack(frames), fps=25)
print(f"Video saved to {output_file}")
