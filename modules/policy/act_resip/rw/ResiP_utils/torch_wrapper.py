import gymnasium as gym
import torch

class TorchWrapper(gym.Wrapper):
    """
    A wrapper that casts the environment's output to torch tensors on a given device.
    """
    def __init__(self, env, device="cuda"):
        super().__init__(env)
        self.device = device

    def reset(self, **kwargs):
        """
        Gymnasium's reset can return (obs, info) in recent versions.
        """
        obs, info = super().reset(**kwargs)
        obs = self.to_torch(obs)
        return obs

    def step(self, action):
        """
        Gymnasium's step can return 5-tuple (obs, reward, terminated, truncated, info).
        """
        action_cpu = action.cpu().detach().numpy()
        obs, reward, terminated, truncated, info = super().step(action_cpu)
        obs = self.to_torch(obs)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.float32)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool)
        return *obs, reward, terminated

    def to_torch(self, observation):
        qpos = torch.as_tensor(observation["state"][:30], device=self.device, dtype=torch.float32)
        env_state = None
        vision_data = torch.as_tensor(observation["pixels"], device=self.device, dtype=torch.float32).permute(2, 0, 1) / 255
        return (qpos.unsqueeze(0), env_state, vision_data.unsqueeze(0))