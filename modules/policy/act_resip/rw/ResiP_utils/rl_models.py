import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )  # Use relu for Swish
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)

    return layer

def build_mlp(
    input_dim,
    hidden_sizes,
    output_dim,
    activation,
    output_std=1.0,
    bias_on_last_layer=True,
    last_layer_bias_const=0.0,
):
    act_func = getattr(nn, activation)
    layers = []
    layers.append(
        layer_init(nn.Linear(input_dim, hidden_sizes[0]), nonlinearity=activation)
    )
    layers.append(act_func())
    for i in range(1, len(hidden_sizes)):
        layers.append(
            layer_init(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nonlinearity=activation
            )
        )
        layers.append(act_func())
    layers.append(
        layer_init(
            nn.Linear(hidden_sizes[-1], output_dim, bias=bias_on_last_layer),
            std=output_std,
            nonlinearity="Tanh",
            bias_const=last_layer_bias_const,
        )
    )
    return nn.Sequential(*layers)

class ResidualMLPPolicy(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor_mean = build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=[config.actor_hidden_dim] * config.actor_num_layers,
            output_dim=np.prod(self.action_dim),
            activation="ReLU",
            output_std=config.actor_output_std,
            bias_on_last_layer=config.actor_bias_on_last_layer,
        )

        self.critic = build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=[config.critic_hidden_dim] * config.critic_num_layers,
            output_dim=1,
            activation="ReLU",
            output_std=config.critic_output_std,
            bias_on_last_layer=config.critic_bias_on_last_layer,
            last_layer_bias_const=config.critic_last_layer_biast_const,
        )

        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.action_dim) * -1,
            requires_grad=False
        )

    def get_state_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        action_mean: torch.Tensor = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # print("action mean", action_mean, "action std", action_std)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            return (
                action,
                probs.log_prob(action).sum(dim=1),
                probs.entropy().sum(dim=1),
                self.critic(obs),
                action_mean)

        return (
            probs.log_prob(action).sum(dim=1),
            probs.entropy().sum(dim=1),
            self.critic(obs),
            action_mean)
    
    def compute_loss(self, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, loss_buffers, cfg):
        newlogprob, entropy, newvalue, action_mean = self.get_action_and_value(mb_obs, mb_actions)

        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
        approx_kl_value = approx_kl.item()

        if mb_advantages.shape[0] > 1:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-16
            )

        # Lambda function to convert to detached numpy array on CPU
        to_numpy = lambda t: t.cpu().detach().numpy()

        pg_loss1 = -mb_advantages * ratio
        loss_buffers["pg_loss_1"].append(to_numpy(pg_loss1))
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
        )
        loss_buffers["pg_loss_2"].append(to_numpy(pg_loss2))
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        loss_buffers["pg_loss"].append(to_numpy(pg_loss))

        newvalue = newvalue.view(-1)
        if (cfg.clip_v_loss): # Clip vloss
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -cfg.clip_coef, cfg.clip_coef)
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else: # Not clip vloss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        loss_buffers["v_loss"].append(to_numpy(v_loss))

        entropy_loss = entropy.mean() * cfg.ent_coef    # To-Do check out multiplication factor 0
        loss_buffers["entropy_loss"].append(to_numpy(entropy_loss))

        ppo_loss = pg_loss - entropy_loss
        loss_buffers["ppo_loss"].append(to_numpy(ppo_loss))

        residual_l1_loss = torch.mean(torch.abs(action_mean))
        residual_l2_loss = torch.mean(torch.square(action_mean))
        loss_buffers["residual_l1_loss"].append(to_numpy(residual_l1_loss))
        loss_buffers["residual_l2_loss"].append(to_numpy(residual_l2_loss))

        policy_loss = ppo_loss + cfg.residual_l1_coef * residual_l1_loss + cfg.residual_l2_coef * residual_l2_loss # To-Do try removing regularization losses
        loss_buffers["policy_loss"].append(to_numpy(policy_loss))

        loss = policy_loss + (v_loss * cfg.v_loss_factor)
        loss_buffers["total_loss"].append(to_numpy(loss))

        return loss, approx_kl_value

        
    
    @property
    def actor_parameters(self):
        return [p for n, p in self.named_parameters() if "actor" in n]
    
    @property
    def critic_parameters(self):
        return [p for n, p in self.named_parameters() if "critic" in n]