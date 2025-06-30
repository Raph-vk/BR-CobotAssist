import torch

class RunningNorm:
    def __init__(self, obs_space, config):
        self.mean = torch.zeros(obs_space, device=config.device)
        self.var  = torch.ones(obs_space,  device=config.device)
        self.eps = config.normalize_eps
        self.count = config.normalize_eps
        self.clip = config.normalize_clip

    def update(self, obs, dims=(0)):
        # x: batch of observations, shape [batch_size, *obs_shape]
        batch_mean = torch.mean(obs, dim=dims)
        batch_var  = torch.var(obs,  dim=dims)
        batch_count = 1

        # combine existing and new stats (Welford’s trick)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x, dims=None):
        if (dims == None):
            std = torch.sqrt(self.var) + self.eps
            x   = (x - self.mean) / std
        else:
            mean_ex = self.mean.reshape((1, -1, 1, 1))
            std_ex  = (torch.sqrt(self.var) + self.eps).reshape((1, -1, 1, 1))
            x = (x - mean_ex) / std_ex
        return torch.clip(x, -self.clip, self.clip)
    
    def state_dict(self):
        return {
            'mean': self.mean.cpu(),
            'var': self.var.cpu(),
            'eps': self.eps,
            'count': self.count,
            'clip': self.clip
        }

    def load_state_dict(self, state):
        self.mean = state['mean'].to(self.mean.device)
        self.var = state['var'].to(self.var.device)
        self.eps = state['eps']
        self.count = state['count']
        self.clip = state['clip']