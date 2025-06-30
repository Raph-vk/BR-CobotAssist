import math
import time
import minari
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import gymnasium_robotics
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
import torch.nn as nn
from bc import BehaviorCloningNet
import numpy as np
import os
import json
import glfw

from rl_models import ResidualMLPPolicy

# Initialize GLFW to prevent errors
if not glfw.init():
    raise RuntimeError("GLFW initialization failed!")

# 🔹 Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def eval_bc_model(model, env, trials=5):
    # print("Evaluating BC model:")
    start_time = time.time()
    total_reward = 0
    successes = 0
    model.eval()

    for _ in range(trials):
        obs, info = env.reset()
        done = False
        trial_reward = 0.0
        step = 0
        action_buffer = []  # Stores the current action sequence

        while not done:
            if len(action_buffer) == 0:  # If we need a new chunk
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    action_chunk = model(obs_tensor).squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
                    # print(f"Action chunk shape {action_chunk.shape} and full chunk:\n{action_chunk}")

                action_buffer.extend(action_chunk)  # Store actions

            # Execute the next action in the buffer
            action = action_buffer.pop(0)  # Take first action in the sequence
            obs, reward, done, truncated, info = env.step(action)
            trial_reward += reward
            step += 1

            if done or truncated:
                total_reward += trial_reward
                if (trial_reward > 20):
                    successes += 1
                break
        # print("Total reward from evaluation:", round(trial_reward))
    print(f"Total reward from trials: {round(total_reward)}, AVG reward from trials: {round(total_reward / trials)} with success rate: {round(successes / trials * 100, 1)}%")
    print(f"Time to compute BC eval: {round(time.time()-start_time, 1)}")


def eval_rl_model(bc_model, rl_model, env, trials):
    # print("Evaluating RL model:")
    total_reward = 0
    successes = 0

    bc_model.eval()
    rl_model.eval()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Pre-allocate GPU tensors
    obs_tensor = torch.empty((1, obs_dim), device=device)
    action_tensor = torch.empty((1, action_dim), device=device)

    # Evaluate over multiple trials
    start_time = time.time()
    for _ in range(trials):
        obs, info = env.reset()
        done = False
        trial_reward = 0.0

        # Overwrite obs_tensor
        obs_tensor[0].copy_(torch.from_numpy(obs))

        # Get first BC chunk
        with torch.no_grad():
            bc_actions = bc_model(obs_tensor.unsqueeze(0))  # shape: (1, chunk_size, action_dim)
        bc_actions = bc_actions.squeeze(0)   # shape: (chunk_size, action_dim)
        action_chunk_ptr = 0
        chunk_size = bc_actions.shape[0]

        while not done:
            # If we used up the chunk, get a new chunk
            if action_chunk_ptr >= chunk_size:
                with torch.no_grad():
                    bc_actions = bc_model(obs_tensor.unsqueeze(0))
                bc_actions = bc_actions.squeeze(0)
                action_chunk_ptr = 0
                chunk_size = bc_actions.shape[0]
            
            # current BC action: 
            action_tensor[0].copy_(bc_actions[action_chunk_ptr])
            action_chunk_ptr += 1

            # Form residual observation
            residual_obs = torch.cat([obs_tensor, action_tensor], dim=1)
            
            # RL residual action
            with torch.no_grad():
                _, _, _, _, residual_action = rl_model.get_action_and_value(residual_obs)
            
            # Combine BC + residual
            improved_action = action_tensor + 0.1 * residual_action  # shape (1, action_dim)
            
            # Step env (CPU expects numpy)
            action_cpu = improved_action[0].cpu().numpy()  # or improved_action.squeeze(0).cpu().numpy()
            obs, reward, done, truncated, info = env.step(action_cpu)

            trial_reward += reward

            # Overwrite obs_tensor
            obs_tensor[0].copy_(torch.from_numpy(obs))

            if done or truncated:
                total_reward += trial_reward
                if (trial_reward > 20):
                    successes += 1
                break
    
    print(
        f"Total reward from trials: {round(total_reward)}, "
        f"AVG reward: {round(total_reward / trials)}, "
        f"Success rate: {round(successes / trials * 100, 1)}%"
    )
    print(f"Time to compute RL eval: {round(time.time()-start_time, 1)}")

def load_bc_model(model_path):
    if os.path.exists(f"{model_path}.json"):
        with open(f"{model_path}.json", "r") as f:
            metadata = json.load(f)
    else:
        return None

    input_dim = metadata["input_dim"]
    output_dim = metadata["output_dim"]
    chunk_size = metadata["chunk_size"]

    # Recreate model and move to GPU
    bc_model = BehaviorCloningNet(input_dim, output_dim, chunk_size).to(device)

    if os.path.exists(f"{model_path}.pth"):
        bc_model.load_state_dict(torch.load(f"{model_path}.pth", weights_only=True, map_location=device))
        # print("Model weights and metadata loaded successfully!")
        return bc_model
    return None

def load_rl_model(model_path):
    residual_policy = ResidualMLPPolicy(67, 28).to(device)
    if os.path.exists(f"{model_path}.pth"):
        residual_policy.load_state_dict(torch.load(f"{model_path}.pth", weights_only=True, map_location=device))
        # print("Model weights and metadata loaded successfully!")
        return residual_policy
    return None

def make_adroit_door_env(max_episode_steps=400, render_mode=None):
    """
    Returns a function that, when called, creates a fresh instance
    of 'AdroitHandDoorSparse-v1' with the given settings.
    """
    def _init():
        # If you need any extra wrappers, you can wrap them here
        env = gym.make('AdroitHandDoorSparse-v1',
                       max_episode_steps=max_episode_steps,
                       render_mode=render_mode)
        return env
    return _init

if __name__ == "__main__":
    base_model_name = "65_run"
    rl_model_names = [11500]
    chunk_size = 10
    max_episode_steps = 200
    trials = 100

    render_mode = "human"  # "human" or "rgb_array"

    # Setup environment
    gym.register_envs(gymnasium_robotics)
    env = gym.make('AdroitHandDoorSparse-v1', max_episode_steps=max_episode_steps, render_mode=render_mode)

    # 1) Load or train BC
    bc_model_path = os.path.join(os.getcwd(), f"models/{base_model_name}")
    bc_model = load_bc_model(bc_model_path)
    print(f"Loaded BC model '{base_model_name}'")
    # eval_bc_model(bc_model, env, trials)

    for checking_rl_model in rl_model_names:
        rl_model_path = os.path.join(os.getcwd(), f"models/{base_model_name}_rl_{checking_rl_model}")
        rl_model = load_rl_model(rl_model_path)
        print(f"Loaded RL model '{checking_rl_model}'")
        eval_rl_model(bc_model, rl_model, env, trials)